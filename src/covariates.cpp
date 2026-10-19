#include "covariates.hpp"
#include "numpy_sum.hpp"
#include "common.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
namespace ts {
namespace {
size_t mul(size_t a,size_t b){if(b&&a>SIZE_MAX/b)throw Error("row dimensions overflow");return a*b;}
size_t add(size_t a,size_t b){if(a>SIZE_MAX-b)throw Error("row dimensions overflow");return a+b;}
float finite_value(float v){return std::isnan(v)?0:std::isinf(v)?std::copysign(std::numeric_limits<float>::max(),v):v;}
void codes_valid(const std::vector<double>& x){for(double v:x)if(std::isinf(v))throw Error("categorical covariates must hold finite codes or NaN; got an infinite value");}
}
TargetEncoding target_encode(const std::vector<double>& target,const std::vector<double>& past,const std::vector<double>& future,double smooth){
 if(target.size()!=past.size()||!std::isfinite(smooth)||smooth<=0)throw Error("invalid target encoding inputs");
 codes_valid(past);codes_valid(future);
 auto cats=past;cats.erase(std::remove_if(cats.begin(),cats.end(),[](double x){return !std::isfinite(x);}),cats.end());std::sort(cats.begin(),cats.end());cats.erase(std::unique(cats.begin(),cats.end()),cats.end());
 size_t nan=cats.size();std::vector<double> values(target.size()),sums(nan+1),counts(nan+1);double count=0;
 auto slot=[&](double code){if(std::isnan(code))return nan;auto it=std::lower_bound(cats.begin(),cats.end(),code);return it!=cats.end()&&*it==code?size_t(it-cats.begin()):nan+1;};
 for(size_t i=0;i<target.size();++i)if(std::isfinite(target[i])){values[i]=target[i];++count;}
 double mean=count?numpy_sum(values)/count:0;
 for(size_t i=0;i<past.size();++i){size_t s=slot(past[i]);sums[s]+=values[i];counts[s]+=std::isfinite(target[i])?1:0;}
 std::vector<double> lookup(nan+2,mean);for(size_t i=0;i<=nan;++i)lookup[i]=(smooth*mean+sums[i])/(smooth+counts[i]);
 TargetEncoding e;for(double v:past)e.past.push_back(float(lookup[slot(v)]));for(double v:future)e.future.push_back(float(lookup[slot(v)]));return e;
}
bool supports_known_covariates(const Config& c){
 if(!c.covariate_enabled)return false;
 std::vector<bool> ops;for(auto k:c.block_schedule)for(bool t:layer_axes(k))ops.push_back(t);
 for(size_t i=0;i<ops.size();++i)if(ops[i]){bool time=false,var=false;for(size_t j=i+1;j<ops.size();++j){time|=ops[j];var|=!ops[j];}if(time&&var)return true;}
 return false;
}
AssembledRows assemble_rows(const Config& c,const std::vector<float>& context,const std::vector<int8_t>& user,size_t B,size_t V,size_t T,size_t H,const std::vector<float>& past,size_t P,const std::vector<float>& known,size_t K,size_t cap,const std::vector<double>& cp,size_t CP,const std::vector<double>& ck,size_t CK){
 if(!B||!V||!T||!H||!c.patch_size||!c.context_length)throw Error("invalid row dimensions");
 size_t TH=add(T,H),BV=mul(B,V);
 if(context.size()!=mul(BV,T)||(!user.empty()&&user.size()!=context.size())||past.size()!=mul(mul(B,P),T)||known.size()!=mul(mul(B,K),TH)||cp.size()!=mul(mul(B,CP),T)||ck.size()!=mul(mul(B,CK),TH))throw Error("row input shape mismatch");
 for(auto v:user)if(v!=0&&v!=1)throw Error("observation mask must be boolean");
 if((P||K||CP||CK)&&!c.covariate_enabled)throw Error("this checkpoint was not trained with covariates");
 if((K||CK)&&!supports_known_covariates(c))throw Error("this checkpoint cannot condition on known-future covariates");
 codes_valid(cp);codes_valid(ck);
 cap=cap?cap:c.context_length;if(cap>c.context_length)throw Error("max_context exceeds stored position range");
 size_t kept=std::min(T,cap),ps=c.patch_size,pad=(ps-kept%ps)%ps,len=add(kept,pad),hor=mul(add(H,ps-1)/ps,ps);
 size_t all=add(add(V,add(P,CP)),add(K,CK)),R=mul(B,all);
 AssembledRows a;a.rows_count=R;a.length=len;a.horizon=hor;a.rows.assign(mul(R,len),0);a.mask.assign(a.rows.size(),0);a.groups.resize(R);
 if(P||CP||K||CK)a.roles.resize(R);
 if(K||CK){a.future.assign(mul(R,hor),0);a.future_mask.assign(a.future.size(),0);}
 std::vector<float> filled(context.size());std::vector<int8_t> obs(context.size());
 for(size_t r=0;r<BV;++r){float prev=0;for(size_t t=0;t<T;++t){size_t i=r*T+t;obs[i]=std::isfinite(context[i])&&(user.empty()||user[i]);if(obs[i])prev=context[i];filled[i]=prev;}}
 auto encode=[&](const std::vector<double>& codes,size_t C,size_t width){std::vector<float> out(codes.size());
  for(size_t b=0;b<B;++b)for(size_t j=0;j<C;++j){size_t off=(b*C+j)*width;
   if(V!=1){for(size_t t=0;t<width;++t)out[off+t]=float(codes[off+t]);continue;}
   std::vector<double> target(T);bool started=false;
   for(size_t t=0;t<T;++t){started|=obs[b*T+t];target[t]=started&&t>=T-kept?double(filled[b*T+t]):NAN;}
   std::vector<double> p(codes.begin()+off,codes.begin()+off+T),f(codes.begin()+off+T,codes.begin()+off+width);
   auto e=target_encode(target,p,f);
   for(size_t t=0;t<T;++t)out[off+t]=std::isnan(target[t])&&std::isnan(p[t])?NAN:e.past[t];
   std::copy(e.future.begin(),e.future.end(),out.begin()+off+T);
  }return out;
 };
 auto ep=encode(cp,CP,T),ek=encode(ck,CK,TH);
 for(size_t b=0;b<B;++b)for(size_t j=0;j<all;++j){size_t r=b*all+j;int role=j<V?0:j<V+P+CP?1:2;a.groups[r]=b;if(!a.roles.empty())a.roles[r]=role;
  const std::vector<float>* src;size_t off;
  if(j<V){src=&filled;off=(b*V+j)*T;}
  else if(j<V+P){src=&past;off=(b*P+j-V)*T;}
  else if(j<V+P+CP){src=&ep;off=(b*CP+j-V-P)*T;}
  else if(j<V+P+CP+K){src=&known;off=(b*K+j-V-P-CP)*TH;}
  else {src=&ek;off=(b*CK+j-V-P-CP-K)*TH;}
  for(size_t t=0;t<kept;++t){size_t from=off+T-kept+t,to=r*len+pad+t;float v=(*src)[from];a.rows[to]=role?finite_value(v):v;a.mask[to]=role?!std::isnan(v):obs[(b*V+j)*T+T-kept+t];}
  if(role==2)for(size_t t=0;t<H;++t){float v=(*src)[off+T+t];a.future[r*hor+t]=finite_value(v);a.future_mask[r*hor+t]=!std::isnan(v);}
 }
 return a;
}
}
