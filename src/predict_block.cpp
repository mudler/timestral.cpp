#include "predict_block.hpp"
#include <limits>
#include <algorithm>
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
