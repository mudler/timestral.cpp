#include "predict_block.hpp"
#include <limits>
#include <algorithm>
#include <cmath>
namespace ts {
BlockPrediction PredictBlock::context_only(const std::vector<float>& inputs,const std::vector<int8_t>& observed,size_t R,size_t T,size_t h,const TimeMask& mask,const std::vector<int64_t>& positions,const std::vector<int8_t>& vm) {
 const auto& c=model_.config();size_t ps=c.patch_size,D=c.d_model,L=T/ps;
 if(!R||!T||T%ps||!h||h>c.block_size||L>std::numeric_limits<size_t>::max()-h)throw Error("invalid context block dimensions");
 size_t S=L+h;
 if(R>std::numeric_limits<size_t>::max()/S/D||positions.size()!=R*S||mask.allowed.empty()||mask.enc_len!=L)throw Error("context block requires matching two-stream masks and positions");
 auto scaled=scale_rows(c,inputs,observed,R,T);auto embed=encoder_.embed(scaled.scaled,observed,R,T);
 embed.output=context_patch_projection(model_,embed.input,R,L);
 std::vector<float> x(R*S*D);auto query=model_.tensor("patch.forecast_query");const float* q=static_cast<const float*>(query->data);
 for(size_t r=0;r<R;++r){std::copy_n(embed.output.begin()+r*L*D,L*D,x.begin()+r*S*D);for(size_t j=L;j<S;++j)std::copy_n(q,D,x.begin()+(r*S+j)*D);}
 auto norm=encoder_.run(x,R,S,mask,positions,{},vm).at("out_norm");std::vector<float> dec(R*h*D);
 for(size_t r=0;r<R;++r)std::copy_n(norm.begin()+(r*S+L)*D,h*D,dec.begin()+r*h*D);
 BlockPrediction result;result.normalized=monotone_quantiles(c,head_.project(dec,R,h));
 for(size_t r=0;r<R;++r){result.loc.push_back(scaled.loc[r*T+T-1]);result.scale.push_back(scaled.scale[r*T+T-1]);}
 result.output=denormalize_quantiles(result.normalized,result.loc,result.scale,c.num_quantiles(),c.use_asinh);return result;
}
}

#include "masks.hpp"
#include "covariates.hpp"
namespace ts {
BlockPrediction PredictBlock::predict(const std::vector<float>& inputs,const std::vector<int8_t>& observed,size_t R,size_t T,size_t h,const std::vector<int64_t>& groups,const std::vector<int64_t>& roles,const std::vector<float>& future,const std::vector<int8_t>& fm){
 const auto& c=model_.config();size_t ps=c.patch_size,D=c.d_model;
 auto mul=[](size_t a,size_t b){if(b&&a>SIZE_MAX/b)throw Error("prediction dimensions overflow");return a*b;};
 if(!R||!T||!h||!ps||T%ps||h>c.block_size||inputs.size()!=mul(R,T)||observed.size()!=inputs.size()||groups.size()!=R||(!roles.empty()&&roles.size()!=R))throw Error("invalid prediction dimensions");
 if(T>size_t(c.context_length)+ps-1)throw Error("context exceeds stored position range");
 for(float v:inputs)if(!std::isfinite(v))throw Error("prediction inputs must be prepared finite values");
 for(float v:future)if(!std::isfinite(v))throw Error("known future must be prepared finite values");
 for(auto v:roles)if(v<0||v>2)throw Error("invalid row role");
 bool cov=c.covariate_enabled&&!roles.empty()&&std::any_of(roles.begin(),roles.end(),[](int64_t v){return v!=0;});
 if(!cov){auto a=build_masks(c,observed,R,T,h,groups);return context_only(inputs,observed,R,T,h,a.time,a.positions,a.variate);}
 if(std::find(roles.begin(),roles.end(),2)!=roles.end()&&!supports_known_covariates(c))throw Error("this checkpoint cannot condition on known-future covariates");
 size_t H=mul(h,ps);if(T>SIZE_MAX-H)throw Error("prediction dimensions overflow");size_t W=T+H,N=W/ps,L=T/ps,S=mul(N,2);
 if(future.empty()!=fm.empty()||(!future.empty()&&(future.size()!=mul(R,H)||fm.size()!=future.size())))throw Error("known_future_mask is required alongside matching known_future");
 std::vector<float> full(mul(R,W),0);std::vector<int8_t> obs(full.size(),0),known(R);
 for(size_t r=0;r<R;++r){std::copy_n(inputs.begin()+r*T,T,full.begin()+r*W);std::copy_n(observed.begin()+r*T,T,obs.begin()+r*W);known[r]=roles[r]==2;
  if(!future.empty()){std::copy_n(future.begin()+r*H,H,full.begin()+r*W+T);std::copy_n(fm.begin()+r*H,H,obs.begin()+r*W+T);}}
 auto a=build_masks(c,obs,R,W,h,groups,roles);
 auto scaled=scale_rows(c,full,obs,R,W);overwrite_known_rows(c,full,obs,known,W,scaled);
 auto embed=encoder_.embed(scaled.scaled,obs,R,W);
 // Keep the reviewed context projection rounding for the full covariate window.
 embed.output=context_patch_projection(model_,embed.input,R,N);
 std::vector<float> x(mul(mul(R,S),D));const float* query=static_cast<const float*>(model_.tensor("patch.forecast_query")->data);
 for(size_t r=0;r<R;++r){std::copy_n(embed.output.begin()+r*N*D,N*D,x.begin()+r*S*D);for(size_t p=N;p<S;++p)std::copy_n(query,D,x.begin()+(r*S+p)*D);}
 auto norm=encoder_.run(x,R,S,a.time,a.positions,{},a.variate).at("out_norm");std::vector<float> dec(mul(mul(R,N),D));
 for(size_t r=0;r<R;++r)std::copy_n(norm.begin()+(r*S+N)*D,N*D,dec.begin()+r*N*D);
 auto quantiles=monotone_quantiles(c,head_.project(dec,R,N));size_t Q=c.num_quantiles();BlockPrediction out;out.normalized.resize(mul(mul(R,H),Q));
 for(size_t r=0;r<R;++r){std::copy_n(quantiles.begin()+(r*N+L)*ps*Q,H*Q,out.normalized.begin()+r*H*Q);
  // Known-row replacement precedes anchor selection; frames are step-expanded.
  out.loc.push_back(scaled.loc[r*W+(L-1)*ps]);out.scale.push_back(scaled.scale[r*W+(L-1)*ps]);}
 out.output=denormalize_quantiles(out.normalized,out.loc,out.scale,Q,c.use_asinh);return out;
}
}
