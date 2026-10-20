#include "forecaster.hpp"
#include "decode_state.hpp"
#include "refine.hpp"
#include "common.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
namespace ts {
namespace {
size_t product(size_t a,size_t b){if(b&&a>SIZE_MAX/b)throw Error("forecast dimensions overflow");return a*b;}
template<class T> std::vector<T> slice(const std::vector<T>& x,size_t rows,size_t width,size_t start,size_t n){
 if(x.empty())return {};if(x.size()!=product(rows,width)||start>width||n>width-start)throw Error("forecast slice shape");
 std::vector<T> out(product(rows,n));for(size_t r=0;r<rows;++r)std::copy_n(x.begin()+r*width+start,n,out.begin()+r*n);return out;
}
}
std::vector<float> Forecaster::decode(const AssembledRows& a,size_t H,size_t cap,size_t shot,std::vector<DecodeBlock>* trace,const DecodeProgress& progress){
 return decode_impl(a,H,cap,shot,trace,progress,nullptr);
}
std::vector<float> Forecaster::decode_impl(const AssembledRows& a,size_t H,size_t cap,size_t shot,std::vector<DecodeBlock>* trace,const DecodeProgress& progress,ForecastAnchors* anchors){
 const auto& c=model_.config();size_t ps=c.patch_size,Q=c.num_quantiles(),R=a.rows_count,T=a.length;
 cap=cap?cap:c.context_length;shot=shot?shot:c.per_shot_horizon_patches;shot=std::min(shot,size_t(c.block_size));
 if(!H||!R||!T||T%ps||!shot||!cap||cap>c.context_length||H>SIZE_MAX-(ps-1))throw Error("invalid decode dimensions");
 if(a.rows.size()!=product(R,T)||a.mask.size()!=a.rows.size()||a.groups.size()!=R||(!a.roles.empty()&&a.roles.size()!=R))throw Error("invalid decode rows");
 const size_t patches=(H+ps-1)/ps,full=product(patches,ps);
 if(a.horizon!=full||a.future.size()!=a.future_mask.size()||(!a.future.empty()&&a.future.size()!=product(R,full)))throw Error("invalid decode future");
 if(trace)trace->clear();
 auto context=a.rows;auto mask=a.mask;std::vector<int8_t> targets;
 if(!a.roles.empty()){targets.resize(R);for(size_t r=0;r<R;++r){if(a.roles[r]<0||a.roles[r]>2)throw Error("invalid decode role");targets[r]=a.roles[r]==0;}}
 std::vector<float> output(product(product(R,H),Q));DecodeState state=DecodePending{patches,0};
 while(auto pending=std::get_if<DecodePending>(&state)){
  size_t block=std::min(shot,pending->remaining),n=product(block,ps),offset=product(pending->offset,ps);
  auto future=slice(a.future,R,full,offset,n);auto fm=slice(a.future_mask,R,full,offset,n);
  if(progress)progress(offset,block,T);
  auto pred=predictor_.predict(context,mask,R,T,block,a.groups,a.roles,future,fm);
  if(anchors && offset==0) {
   if(!anchors->batch || !anchors->targets || R%anchors->batch || R/anchors->batch<anchors->targets || pred.loc.size()!=R || pred.scale.size()!=R)throw Error("initial anchor shape mismatch");
   size_t all=R/anchors->batch,count=product(anchors->batch,anchors->targets);
   anchors->loc.resize(count);anchors->scale.resize(count);
   for(size_t b=0;b<anchors->batch;++b)for(size_t v=0;v<anchors->targets;++v) {
    size_t from=b*all+v,to=b*anchors->targets+v;
    float loc=pred.loc[from],scale=pred.scale[from];
    if(!std::isfinite(loc)||!std::isfinite(scale)||scale<=0)throw Error("invalid initial anchor");
    anchors->loc[to]=loc;anchors->scale[to]=scale;
   }
  }
  auto q=refine_block(c,pred.output,context,mask,R,T,block,targets);
  if(trace)trace->push_back({offset,block,T,context,future,pred.output,q,mask,fm});
  size_t used=std::min(n,H-offset);for(size_t r=0;r<R;++r)std::copy_n(q.begin()+r*n*Q,used*Q,output.begin()+(r*H+offset)*Q);
  auto transition=reduce_decode(state,{block});state=transition.state;
  if(std::holds_alternative<DecodeDone>(state))break;
  if(T>SIZE_MAX-n)throw Error("decode context overflow");
  size_t joined=T+n,keep=joined>cap?std::max(ps,(cap/ps)*ps):joined,start=joined-keep;
  std::vector<float> next(product(R,keep));std::vector<int8_t> nm(next.size());
  for(size_t r=0;r<R;++r)for(size_t j=0;j<keep;++j){size_t t=start+j,i=r*keep+j;
   if(t<T){next[i]=context[r*T+t];nm[i]=mask[r*T+t];continue;}
   size_t k=t-T;float value=q[(r*n+k)*Q+c.median_index()];int8_t observed=1;
   if(!std::isfinite(value)||std::abs(value)>=c.extreme_value_threshold)value=0;
   if(!a.roles.empty()){
    if(a.roles[r]==2&&!future.empty()){value=future[r*n+k];observed=fm[r*n+k];}
    if(a.roles[r]==1){value=0;observed=0;}
   }
   next[i]=value;nm[i]=observed;
  }
  context=std::move(next);mask=std::move(nm);T=keep;
 }
 return output;
}
std::vector<float> Forecaster::raw(const AssembledRows& a,const ForecastRequest& r,std::vector<DecodeBlock>* trace,const DecodeProgress& progress,ForecastAnchors* anchors){
 const auto& c=model_.config();size_t Q=c.num_quantiles(),all=a.rows_count/r.batch;
 auto decoded=decode_impl(a,r.horizon,r.max_context,r.per_shot_horizon_patches,trace,progress,anchors);
 std::vector<float> out(product(product(product(r.batch,r.targets),r.horizon),Q));
 double cap=r.quantile_real_cap_k.value_or(c.quantile_real_cap_k);
 if(!std::isfinite(cap))throw Error("invalid quantile cap");
 ScaleResult frames;if(cap>0)frames=scale_rows(c,a.rows,a.mask,a.rows_count,a.length);
 size_t nan=0;
 for(size_t b=0;b<r.batch;++b)for(size_t v=0;v<r.targets;++v){size_t row=b*all+v;
  float lo=std::numeric_limits<float>::infinity(),hi=-lo;
  if(cap>0){for(size_t t=0;t<a.length;++t)if(a.mask[row*a.length+t]){lo=std::min(lo,a.rows[row*a.length+t]);hi=std::max(hi,a.rows[row*a.length+t]);}
   if(!std::isfinite(lo))lo=0;if(!std::isfinite(hi))hi=0;
   float width=frames.scale[row*a.length+a.length-1]*float(cap);lo=lo-width;hi=hi+width;
  }
  for(size_t t=0;t<r.horizon*Q;++t){float value=decoded[row*r.horizon*Q+t];if(cap>0)value=std::min(std::max(value,lo),hi);nan+=std::isnan(value);out[(b*r.targets+v)*r.horizon*Q+t]=value;}
 }
 if(nan)throw Error("forecast produced "+std::to_string(nan)+" NaN values; the input is degenerate");
 return out;
}
Forecast Forecaster::forecast(const ForecastRequest& r,ForecastTrace* trace){
 if(trace){trace->initial_anchors.reset();trace->positive.clear();trace->negative.clear();}
 std::optional<ForecastAnchors> anchors;
 if(trace && trace->capture_initial_anchors)anchors=ForecastAnchors{r.batch,r.targets,{},{}};
 const auto& c=model_.config();
 auto a=assemble_rows(c,r.context,r.observation_mask,r.batch,r.targets,r.length,r.horizon,r.past,r.past_count,r.known,r.known_count,r.max_context,r.categorical_past,r.categorical_past_count,r.categorical_known,r.categorical_known_count);
 auto out=raw(a,r,trace && trace->capture_blocks?&trace->positive:nullptr,trace?trace->before_block:DecodeProgress{},anchors?&*anchors:nullptr);
 size_t Q=c.num_quantiles();
 if(r.symmetric_averaging.value_or(c.symmetric_averaging)){
  for(size_t i=0;i<Q;++i)if(std::abs(c.quantile_knots[i]+c.quantile_knots[Q-1-i]-1)>=1e-9)throw Error("symmetric averaging needs quantile levels symmetric about 0.5");
  for(auto& x:a.rows)x=-x;for(auto& x:a.future)x=-x;
  auto mirrored=raw(a,r,trace && trace->capture_blocks?&trace->negative:nullptr,trace?trace->before_block:DecodeProgress{});
  for(size_t i=0;i<out.size();++i)out[i]=0.5f*(out[i]+(-mirrored[(i/Q)*Q+Q-1-i%Q]));
 }
 if(r.clip_nonnegative.value_or(c.clip_nonnegative))for(size_t row=0;row<r.batch*r.targets;++row){bool any=false,eligible=true;
  for(size_t t=0;t<r.length;++t){size_t i=row*r.length+t;bool observed=std::isfinite(r.context[i])&&(r.observation_mask.empty()||r.observation_mask[i]);if(observed){any=true;eligible&=r.context[i]>=0;}}
  if(any&&eligible)for(size_t i=row*r.horizon*Q;i<(row+1)*r.horizon*Q;++i)out[i]=std::max(out[i],0.f);
 }
 Forecast result{r.batch,r.targets,r.horizon,c.quantile_knots,std::move(out)};
 if(trace && anchors)trace->initial_anchors=std::move(anchors);
 return result;
}
}
