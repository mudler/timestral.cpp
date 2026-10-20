#ifdef TS_STANDALONE_MKL
#include "standalone_mkl.hpp"
#endif
#include "head.hpp"
#include "weight_policy.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
namespace ts {
namespace {
// Copyright Naoki Shibata and contributors 2010 - 2025.
// Distributed under the Boost Software License, Version 1.0.
// See third_party/notices/SLEEF.txt for the complete licence.
// Adapted from SLEEF 906ca7512ee483296780a81a21b9ca715d40dfe1 xexpf.
// Source scope and modifications: third_party/notices/README.md.
// SLEEF F32 exp range reduction and polynomial, as used by ATen AVX2.
// Verified against the pinned binary on 100,000 inputs and 417,792 SiLU
// activations. Retain libm handling outside the normal exponent range.
static float activation_exp(float x) {
    if(!std::isfinite(x) || x < -87 || x > 88)return std::exp(x);
    float q=std::nearbyint(x*1.4426950408889634f);
    float r=std::fma(q,-.693145751953125f,x);
    r=std::fma(q,-1.428606765330187045e-6f,r);
    float u=.000198527617612853646f;
    for(float c:{.00139304355252534151f,.00833336077630519867f,
                 .0416664853692054749f,.166666671633720398f,.5f})u=std::fma(u,r,c);
    return std::ldexp(1.f+std::fma(r*r,u,r),int(q));
}
void head_silu_f32(ggml_tensor* dst,int ith,int nth,void*){auto x=static_cast<const float*>(dst->src[0]->data);auto y=static_cast<float*>(dst->data);for(int64_t i=ith;i<ggml_nelements(dst);i+=nth)y[i]=x[i]/(1.f+activation_exp(-x[i]));}
ggml_tensor* head_silu(ggml_context* ctx,ggml_tensor* x){ggml_tensor* args[]={x};return ggml_custom_4d(ctx,GGML_TYPE_F32,x->ne[0],x->ne[1],x->ne[2],1,args,1,head_silu_f32,1,nullptr);}
void head_linear_f32(ggml_tensor* dst,int ith,int nth,void*) {
 const auto w=static_cast<const float*>(dst->src[0]->data),x=static_cast<const float*>(dst->src[1]->data);
 auto y=static_cast<float*>(dst->data);const float* b=dst->src[2]?static_cast<const float*>(dst->src[2]->data):nullptr;const size_t K=dst->src[0]->ne[0],N=dst->ne[0];
 for(size_t i=ith;i<size_t(ggml_nelements(dst));i+=nth){float sum=0;for(size_t j=0;j<K;++j)sum=std::fma(w[(i%N)*K+j],x[(i/N)*K+j],sum);y[i]=b?sum+b[i%N]:sum;}
}
void head_linear_f64_impl(ggml_tensor* dst,int ith,int nth,void*) {
 const auto w=static_cast<const float*>(dst->src[0]->data),x=static_cast<const float*>(dst->src[1]->data);
 auto y=static_cast<float*>(dst->data);const float* bias=dst->src[2]?static_cast<const float*>(dst->src[2]->data):nullptr;
 const size_t K=dst->src[0]->ne[0],N=dst->ne[0];
 for(size_t i=ith;i<size_t(ggml_nelements(dst));i+=nth){double sum=0;for(size_t k=0;k<K;++k)sum+=double(w[(i%N)*K+k])*double(x[(i/N)*K+k]);float rounded=float(sum);y[i]=bias?rounded+bias[i%N]:rounded;}
}
ggml_tensor* precise_head_linear(ggml_context* ctx,const Model& m,const char* name,ggml_tensor* x){
 auto w=m.tensor(std::string(name)+".weight");if(w->type!=GGML_TYPE_F32)return reduced_projection(ctx,w,x,nullptr);
 ggml_tensor* args[]={w,x};return ggml_custom_4d(ctx,GGML_TYPE_F32,w->ne[1],x->ne[1],x->ne[2],1,args,2,head_linear_f64_impl,1,nullptr);
}
ggml_tensor* head_linear(ggml_context* ctx,const Model& m,const char* name,ggml_tensor* x,bool bias=false){
 auto reduced=m.tensor(std::string(name)+".weight");
 if(reduced->type!=GGML_TYPE_F32)return reduced_projection(ctx,reduced,x,bias?m.tensor(std::string(name)+".bias"):nullptr);
 // Small reference SGEMM dots accumulate sequential F32 FMA, not double.
 if(x->ne[0]>=192)return affine(ctx,m,name,x,bias);
 auto w=m.tensor(std::string(name)+".weight");ggml_tensor* args[]={w,x,bias?m.tensor(std::string(name)+".bias"):nullptr};
 return ggml_custom_4d(ctx,GGML_TYPE_F32,w->ne[1],x->ne[1],x->ne[2],1,args,bias?3:2,head_linear_f32,1,nullptr);
}
}

// Internal test seam for the small-input projection dispatch, not public ABI.
ggml_tensor* detail_head_projection(ggml_context* ctx,const Model& m,const char* name,ggml_tensor* x,bool bias) {
 return head_linear(ctx,m,name,x,bias);
}

std::vector<float> monotone_quantiles(const Config& c, const std::vector<float>& raw) {
 const size_t Q=c.num_quantiles(),m=c.median_index();
 if(!Q||raw.empty()||raw.size()%Q)throw Error("invalid raw quantile layout");
 for(float x:raw)if(std::isnan(x))throw Error("NaN raw quantile");
 std::vector<float> widths(Q),out(raw.size());
 for(size_t q=0;q<Q;++q) {
  double w=q<m?c.quantile_knots[q+1]-c.quantile_knots[q]:q>m?c.quantile_knots[q]-c.quantile_knots[q-1]:0;
  widths[q]=float(c.quantile_width_factor)*float(w);
 }
 const float limit=std::numeric_limits<float>::max();
 auto clamp=[&](float x){if(std::isnan(x))throw Error("NaN monotone quantile");return std::max(-limit,std::min(limit,x));};
 for(size_t i=0;i<raw.size();i+=Q){
  auto inc=[&](size_t q){float x=raw[i+q];return widths[q]*(x>20?x:std::log1p(activation_exp(x)));};
  // ATen CPU cumsum uses a double accumulator and casts each prefix.
  float median=raw[i+m];double sum=0;out[i+m]=clamp(median);
  for(size_t q=m+1;q<Q;++q){sum+=inc(q);out[i+q]=clamp(median+float(sum));}
  sum=0;for(size_t q=m;q-->0;){sum+=inc(q);out[i+q]=clamp(median-float(sum));}
 }
 return out;
}
std::vector<float> Head::project_impl(const std::vector<float>& decoder,size_t rows,size_t patches,bool legacy) {
 size_t D=model_.config().d_model;
 if(!rows||!patches||rows>std::numeric_limits<size_t>::max()/patches/D||decoder.size()!=rows*patches*D)throw Error("invalid head input dimensions");
 auto ctx=runner_.begin();auto x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,patches,rows);ggml_set_input(x);
 auto hidden=legacy?head_linear(ctx,model_,"head.hidden",x):precise_head_linear(ctx,model_,"head.hidden",x);
 auto activated=head_silu(ctx,hidden);
 auto out=legacy?head_linear(ctx,model_,"head.out",activated):precise_head_linear(ctx,model_,"head.out",activated);
 auto residual=legacy?head_linear(ctx,model_,"head.res",x):precise_head_linear(ctx,model_,"head.res",x);
 auto y=ggml_add(ctx,out,residual);runner_.allocate({y});runner_.upload(x,decoder);runner_.compute();return runner_.read(y);
 }
std::vector<float> Head::project(const std::vector<float>& x,size_t rows,size_t patches){return project_impl(x,rows,patches,false);}
std::vector<float> detail::head_project_legacy(Head& h,const std::vector<float>& x,size_t rows,size_t patches){return h.project_impl(x,rows,patches,true);}
void detail::head_linear_f64(ggml_tensor* dst,int ith,int nth,void* data){head_linear_f64_impl(dst,ith,nth,data);}
}
namespace ts {
std::vector<float> context_patch_projection(const Model& m,const std::vector<float>& features,size_t R,size_t L) {
 GraphRunner runner;auto ctx=runner.begin();auto x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,2*m.config().patch_size,L,R);ggml_set_input(x);
 auto patch=[&](const char* name,ggml_tensor* input){
#ifdef TS_STANDALONE_MKL
 auto w=m.tensor(std::string(name)+".weight");if(w->type==GGML_TYPE_F32)return standalone_patch(ctx,w,input,m.tensor(std::string(name)+".bias"));
#endif
 return head_linear(ctx,m,name,input,true);
 };
 auto h=patch("patch.hidden",x);
 auto y=ggml_add(ctx,patch("patch.out",head_silu(ctx,h)),patch("patch.res",x));
 runner.allocate({y});runner.upload(x,features);runner.compute();return runner.read(y);
}
}
