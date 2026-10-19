#include "layers.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
namespace ts {
// F32 graph linear with a widened accumulator, rounded once with bias.
// The default SIMD F32 dot accumulation can exceed the measured tiny floor.
// Inputs and weights remain F32; no reference values enter this kernel.
static void linear_f32(ggml_tensor* dst,int ith,int nth,void*) {
    auto weight=dst->src[0], input=dst->src[1], bias=dst->src[2];
    const auto w=static_cast<const float*>(weight->data);
    const auto x=static_cast<const float*>(input->data);
    const auto b=bias ? static_cast<const float*>(bias->data) : nullptr;
    auto y=static_cast<float*>(dst->data);
    const int64_t k=weight->ne[0], n=weight->ne[1];
    for(int64_t i=ith;i<ggml_nelements(dst);i+=nth) {
        double sum=b ? b[i%n] : 0;
        for(int64_t j=0;j<k;++j)sum+=double(w[(i%n)*k+j])*x[(i/n)*k+j];
        y[i]=float(sum);
    }
}
ResidualOutput residual_block(ggml_context* ctx, const Model& m, ggml_tensor* x) {
    const auto& c=m.config();
    if (!x || x->type!=GGML_TYPE_F32 || x->ne[0]!=2*c.patch_size || x->ne[3]!=1 || !ggml_is_contiguous(x)) throw Error("invalid patch feature layout");
    auto linear=[&](const char* w,const char* b,ggml_tensor* input) {
        auto weight=m.tensor(w), bias=m.tensor(b);
        if(weight->ne[0]!=input->ne[0] || bias->ne[0]!=weight->ne[1] || ggml_nelements(bias)!=bias->ne[0]) throw Error("invalid residual weight/bias dimensions");
        ggml_tensor* args[]={weight,input,bias};
        return ggml_custom_4d(ctx,GGML_TYPE_F32,weight->ne[1],input->ne[1],input->ne[2],1,args,3,linear_f32,1,nullptr);
    };
    auto hidden=linear("patch.hidden.weight","patch.hidden.bias",x);
    auto out=linear("patch.out.weight","patch.out.bias",ggml_silu(ctx,hidden));
    auto residual=linear("patch.res.weight","patch.res.bias",x);
    return {hidden,ggml_add(ctx,out,residual)};
}
ggml_tensor* apply_rope(ggml_context* ctx, ggml_tensor* x, ggml_tensor* cos, ggml_tensor* sin) {
    if(!x || !cos || !sin || x->type!=GGML_TYPE_F32 || cos->type!=GGML_TYPE_F32 || sin->type!=GGML_TYPE_F32 || x->ne[0]%2) throw Error("invalid RoPE layout");
    for(int i=0;i<4;++i) if(cos->ne[i]!=sin->ne[i] || cos->ne[i]!=(i==1?1:x->ne[i])) throw Error("invalid RoPE table layout");
    auto half=x->ne[0]/2;
    auto a=ggml_view_4d(ctx,x,half,x->ne[1],x->ne[2],x->ne[3],x->nb[1],x->nb[2],x->nb[3],0);
    auto b=ggml_view_4d(ctx,x,half,x->ne[1],x->ne[2],x->ne[3],x->nb[1],x->nb[2],x->nb[3],half*sizeof(float));
    auto rotated=ggml_concat(ctx,ggml_neg(ctx,b),a,0);
    return ggml_add(ctx,ggml_mul(ctx,x,cos),ggml_mul(ctx,rotated,sin));
}
}

namespace ts {
ggml_tensor* affine(ggml_context* ctx,const Model& m,const std::string& base,ggml_tensor* x,bool bias) {
    auto w=m.tensor(base+".weight"); auto b=bias?m.tensor(base+".bias"):nullptr;
    if(x->type!=GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[3]!=1 || w->ne[0]!=x->ne[0] || (b && ggml_nelements(b)!=w->ne[1]))throw Error("invalid affine layout");
    ggml_tensor* args[]={w,x,b};
    return ggml_custom_4d(ctx,GGML_TYPE_F32,w->ne[1],x->ne[1],x->ne[2],1,args,b?3:2,linear_f32,1,nullptr);
}
// Widen the reduction/inverse, round normalized activations before learned scale.
// This is an independent evaluation, not a copy of PyTorch SIMD reduction order.
static void rms_f32(ggml_tensor* dst,int ith,int nth,void* userdata) {
    const auto x=(const float*)dst->src[0]->data,w=(const float*)dst->src[1]->data;
    auto y=(float*)dst->data;float eps;auto bits=reinterpret_cast<uintptr_t>(userdata);std::memcpy(&eps,&bits,sizeof(eps));
    auto D=dst->ne[0];
    for(int64_t row=ith;row<ggml_nelements(dst)/D;row+=nth){double sum=0;
        for(int64_t d=0;d<D;++d){sum+=double(x[row*D+d])*x[row*D+d];}
        double inv=1.0/std::sqrt(sum/D+double(eps));
        for(int64_t d=0;d<D;++d)y[row*D+d]=float(double(x[row*D+d])*inv)*w[d];
    }
}
ggml_tensor* rms_norm(ggml_context* ctx,ggml_tensor* x,ggml_tensor* scale,float eps) {
    if(!x || !scale || !ggml_is_contiguous(x) || x->type!=GGML_TYPE_F32 || scale->ne[0]!=x->ne[0] || ggml_nelements(scale)!=x->ne[0] || !(eps>0))throw Error("invalid RMSNorm layout");
    uintptr_t bits=0;std::memcpy(&bits,&eps,sizeof(eps));auto data=reinterpret_cast<void*>(bits);ggml_tensor* args[]={x,scale};
    return ggml_custom_4d(ctx,GGML_TYPE_F32,x->ne[0],x->ne[1],x->ne[2],1,args,2,rms_f32,1,data);
}
// Independent F32 non-flash attention. Explicit rounded logits/probabilities;
// widened dot accumulators match the affine parity policy. No weight conversion.
static void sdpa_f32(ggml_tensor* dst,int ith,int nth,void*) {
    auto q=dst->src[0],k=dst->src[1],v=dst->src[2],mask=dst->src[3];
    auto Q=(const float*)q->data,K=(const float*)k->data,V=(const float*)v->data,M=(const float*)mask->data;
    auto Y=(float*)dst->data;
    const int64_t H=q->ne[1],hd=q->ne[0],S=q->ne[2],D=H*hd;
    std::vector<float> logits(S),p(S);
    for(int64_t job=ith;job<q->ne[3]*S*H;job+=nth) {
        int64_t h=job%H,t=job/H%S,r=job/(H*S);
        float mx=-INFINITY;
        for(int64_t j=0;j<S;++j){double dot=0;for(int64_t d=0;d<hd;++d)dot+=double(Q[(r*S+t)*D+h*hd+d])*K[(r*S+j)*D+h*hd+d];
            logits[j]=float(dot)*float(1/std::sqrt(double(hd)))+M[(r*S+t)*S+j];mx=std::max(mx,logits[j]);}
        double sum=0;for(int64_t j=0;j<S;++j){p[j]=std::exp(logits[j]-mx);sum+=p[j];}
        for(int64_t j=0;j<S;++j)p[j]=float(p[j]/sum);
        for(int64_t d=0;d<hd;++d){double y=0;for(int64_t j=0;j<S;++j)y+=double(p[j])*V[(r*S+j)*D+h*hd+d];Y[(r*S+t)*D+h*hd+d]=float(y);}
    }
}
ggml_tensor* attention_time(ggml_context* ctx,ggml_tensor* q,ggml_tensor* k,ggml_tensor* v,ggml_tensor* mask) {
    if(!q || !k || !v || !mask || !ggml_are_same_shape(q,k) || !ggml_are_same_shape(q,v) || mask->ne[0]!=q->ne[2] || mask->ne[1]!=q->ne[2] || mask->ne[2]!=q->ne[3])throw Error("invalid SDPA layout");
    ggml_tensor* args[]={q,k,v,mask};
    return ggml_custom_4d(ctx,GGML_TYPE_F32,q->ne[0]*q->ne[1],q->ne[2],q->ne[3],1,args,4,sdpa_f32,1,nullptr);
}
ggml_tensor* swiglu(ggml_context* ctx,const Model& m,const std::string& base,ggml_tensor* x) {
    auto z=affine(ctx,m,base+"ffn_w12",x,m.config().mlp_bias);auto F=z->ne[0]/2;
    auto half=[&](size_t offset){return ggml_view_3d(ctx,z,F,z->ne[1],z->ne[2],z->nb[1],z->nb[2],offset);};
    auto hidden=ggml_mul(ctx,half(0),ggml_silu(ctx,half(F*sizeof(float))));
    return affine(ctx,m,base+"ffn_w3",ggml_cont(ctx,hidden),m.config().mlp_bias);
}
}
