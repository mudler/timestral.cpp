#include "layers.hpp"
namespace ts {
// F32 graph linear with a widened accumulator, rounded once with bias.
// The default SIMD F32 dot accumulation can exceed the measured tiny floor.
// Inputs and weights remain F32; no reference values enter this kernel.
static void linear_f32(ggml_tensor* dst,int ith,int nth,void*) {
    auto weight=dst->src[0], input=dst->src[1], bias=dst->src[2];
    const auto w=static_cast<const float*>(weight->data);
    const auto x=static_cast<const float*>(input->data);
    const auto b=static_cast<const float*>(bias->data);
    auto y=static_cast<float*>(dst->data);
    const int64_t k=weight->ne[0], n=weight->ne[1];
    for(int64_t i=ith;i<ggml_nelements(dst);i+=nth) {
        double sum=b[i%n];
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
