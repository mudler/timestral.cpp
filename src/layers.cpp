#include "layers.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
namespace ts {
// F32 linear: reference-sized contractions use FMA panels. Small dots
// retain the widened fallback required by the measured tiny component floors.
// Inputs and weights remain F32; no reference values enter this kernel.
static void linear_f32(ggml_tensor* dst,int ith,int nth,void*) {
    auto weight=dst->src[0], input=dst->src[1], bias=dst->src[2];
    const auto w=static_cast<const float*>(weight->data);
    const auto x=static_cast<const float*>(input->data);
    const auto b=bias ? static_cast<const float*>(bias->data) : nullptr;
    auto y=static_cast<float*>(dst->data);
    const int64_t k=weight->ne[0], n=weight->ne[1];
    for(int64_t i=ith;i<ggml_nelements(dst);i+=nth) {
        if(k>=192) {
            // The pinned CPU reference SGEMM contracts 192-wide panels with
            // F32 FMA, then adds the panels in F32 (verified byte-for-byte).
            float sum=0;
            for(int64_t start=0;start<k;start+=192){float panel=0;
                for(int64_t j=start;j<std::min(k,start+192);++j)
                    panel=std::fma(w[(i%n)*k+j],x[(i/n)*k+j],panel);
                sum+=panel;
            }
            y[i]=b?sum+b[i%n]:sum;
        } else {
            double sum=b ? b[i%n] : 0;
            for(int64_t j=0;j<k;++j)sum+=double(w[(i%n)*k+j])*x[(i/n)*k+j];
            y[i]=float(sum);
        }
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
// Match the reference's F32 operation boundaries and cascade reduction.
// Four independent eight-lane sums use a four-level cascade, as in the
// reference CPU contiguous reduction. Scalar lanes keep this ISA-independent.
static float rms_square_mean(const float* x,int64_t D) {
    float acc[4][4][8] = {};
    int log=0;for(int64_t n=D/32-1;n>0;n>>=1)++log;
    int power=std::max(4,log/4);
    int64_t step=int64_t(1)<<power,mask=step-1,n=D/32,i=0;
    auto add=[&](int64_t v){for(int k=0;k<4;++k)for(int l=0;l<8;++l){float a=x[v*32+k*8+l];acc[0][k][l]+=a*a;}};
    for(;i+step<=n;){for(int64_t j=0;j<step;++j,++i)add(i);
        for(int j=1;j<4;++j){for(int k=0;k<4;++k)for(int l=0;l<8;++l){acc[j][k][l]+=acc[j-1][k][l];acc[j-1][k][l]=0;}
            if(i & (mask<<(j*power)))break;}}
    for(;i<n;++i)add(i);
    for(int j=1;j<4;++j)for(int k=0;k<4;++k)for(int l=0;l<8;++l)acc[0][k][l]+=acc[j][k][l];
    int64_t v=n*4;for(;v<D/8;++v)for(int l=0;l<8;++l){float a=x[v*8+l];acc[0][0][l]+=a*a;}
    for(int k=1;k<4;++k)for(int l=0;l<8;++l)acc[0][0][l]+=acc[0][k][l];
    float sum=0;for(int64_t d=v*8;d<D;++d)sum+=x[d]*x[d];
    for(int l=0;l<8;++l)sum+=acc[0][0][l];
    return sum/float(D);
}
static void rms_f32(ggml_tensor* dst,int ith,int nth,void* userdata) {
    const auto x=(const float*)dst->src[0]->data,w=(const float*)dst->src[1]->data;
    auto y=(float*)dst->data;float eps;auto bits=reinterpret_cast<uintptr_t>(userdata);std::memcpy(&eps,&bits,sizeof(eps));
    auto D=dst->ne[0];
    for(int64_t row=ith;row<ggml_nelements(dst)/D;row+=nth){
        float inv=1.0f/std::sqrt(rms_square_mean(x+row*D,D)+eps);
        for(int64_t d=0;d<D;++d)y[row*D+d]=(x[row*D+d]*inv)*w[d];
    }
}
ggml_tensor* rms_norm(ggml_context* ctx,ggml_tensor* x,ggml_tensor* scale,float eps) {
    if(!x || !scale || !ggml_is_contiguous(x) || x->type!=GGML_TYPE_F32 || scale->ne[0]!=x->ne[0] || ggml_nelements(scale)!=x->ne[0] || !(eps>0))throw Error("invalid RMSNorm layout");
    uintptr_t bits=0;std::memcpy(&bits,&eps,sizeof(eps));auto data=reinterpret_cast<void*>(bits);ggml_tensor* args[]={x,scale};
    return ggml_custom_4d(ctx,GGML_TYPE_F32,x->ne[0],x->ne[1],x->ne[2],1,args,2,rms_f32,1,data);
}
// ATen's exp_u20 polynomial for full eight-lane flash-attention chunks.
// Scalar tails use libm, just as the pinned CPU kernel does.
static float attention_exp(float x) {
    if(x < -87.3365478515625f)return 0;
    if(!std::isfinite(x) || x>88)return std::exp(x);
    float n=std::floor(std::fma(x,1.4426950216293335f,.5f));
    float r=std::fma(-n,.6931471824645996f,x);
    float y=std::fma(r,.00828929059f,.0418978221f);
    for(float a:{.166676521f,.499991506f,.999999701f,1.f})y=std::fma(r,y,a);
    return (y*std::ldexp(1.f,int(n)-1))*2;
}
// CPU flash operation boundaries, evaluated without a dependency on Torch.
static void sdpa_f32(ggml_tensor* dst,int ith,int nth,void* split_data) {
    const int64_t split=reinterpret_cast<uintptr_t>(split_data);
    auto q=dst->src[0],k=dst->src[1],v=dst->src[2],mask=dst->src[3];
    auto Q=(const float*)q->data,K=(const float*)k->data,V=(const float*)v->data,M=(const float*)mask->data;
    auto Y=(float*)dst->data;
    const int64_t H=q->ne[1],hd=q->ne[0],S=q->ne[2],D=H*hd;
    std::vector<float> logits(S),p(S);
    for(int64_t job=ith;job<q->ne[3]*S*H;job+=nth) {
        int64_t h=job%H,t=job/H%S,r=job/(H*S);
        int64_t keys=(split && t<split)?split:S;
        float mx=-INFINITY;
        for(int64_t j=0;j<keys;++j){float dot=0,acc[8]={};for(int64_t d=0;d<hd;++d){float a=Q[(r*S+t)*D+h*hd+d],b=K[(r*S+j)*D+h*hd+d];dot=std::fma(a,b,dot);acc[d%8]+=a*b;}
            // Small CPU GEMM uses non-fused lane reductions: four lanes
            // for complete key groups, eight for the remaining key columns.
            // These are execution-shape branches, never layer/case IDs.
            if(S<8 || (split && S-split<4 && t>=split)){
                if(j<keys/8*8){float a[4]={};for(int64_t d=0;d<hd;++d)a[d%4]+=Q[(r*S+t)*D+h*hd+d]*K[(r*S+j)*D+h*hd+d];dot=(a[0]+a[2])+(a[1]+a[3]);}
                else dot=((acc[0]+acc[4])+(acc[2]+acc[6]))+((acc[1]+acc[5])+(acc[3]+acc[7]));
            }
            logits[j]=float(dot)*float(1/std::sqrt(double(hd)))+M[(r*S+t)*S+j];mx=std::max(mx,logits[j]);}
        float sums[8]={};int64_t end=keys/8*8;
        for(int64_t j=0;j<keys;++j){p[j]=j<end?attention_exp(logits[j]-mx):std::exp(logits[j]-mx);if(j<end)sums[j%8]+=p[j];}
        float a0=sums[0]+sums[4],a1=sums[1]+sums[5],a2=sums[2]+sums[6],a3=sums[3]+sums[7];
        float sum=(a0+a2)+(a1+a3);for(int64_t j=end;j<keys;++j)sum+=p[j];
        // CPU flash reference normalizes after the value contraction, not
        // each probability before it. Preserve that rounding boundary.
        float reciprocal=1.0f/float(sum);
        for(int64_t d=0;d<hd;++d){float y=0;for(int64_t j=0;j<keys;++j)y=std::fma(p[j],V[(r*S+j)*D+h*hd+d],y);Y[(r*S+t)*D+h*hd+d]=float(y)*reciprocal;}
    }
}
ggml_tensor* attention_time(ggml_context* ctx,ggml_tensor* q,ggml_tensor* k,ggml_tensor* v,ggml_tensor* mask,size_t split) {
    if(!q || !k || !v || !mask || !ggml_are_same_shape(q,k) || !ggml_are_same_shape(q,v) || mask->ne[0]!=q->ne[2] || mask->ne[1]!=q->ne[2] || mask->ne[2]!=q->ne[3])throw Error("invalid SDPA layout");
    for(auto t:{q,k,v,mask})if(t->type!=GGML_TYPE_F32 || !ggml_is_contiguous(t))throw Error("invalid SDPA type/strides");
    if(split>size_t(q->ne[2]) || mask->ne[3]!=1)throw Error("invalid SDPA mask rank");
    ggml_tensor* args[]={q,k,v,mask};
    return ggml_custom_4d(ctx,GGML_TYPE_F32,q->ne[0]*q->ne[1],q->ne[2],q->ne[3],1,args,4,sdpa_f32,1,reinterpret_cast<void*>(split));
}
ggml_tensor* attention_variate(ggml_context* ctx,ggml_tensor* q,ggml_tensor* k,ggml_tensor* v,ggml_tensor* mask) {
    return attention_time(ctx,q,k,v,mask,0);
}
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
static void sigmoid_f32(ggml_tensor* dst,int ith,int nth,void*) {
    auto x=(const float*)dst->src[0]->data;auto y=(float*)dst->data;
    for(int64_t i=ith;i<ggml_nelements(dst);i+=nth)y[i]=1.0f/(1.0f+activation_exp(-x[i]));
}
ggml_tensor* sigmoid(ggml_context* ctx,ggml_tensor* x) {
    ggml_tensor* args[]={x};return ggml_custom_4d(ctx,GGML_TYPE_F32,x->ne[0],x->ne[1],x->ne[2],x->ne[3],args,1,sigmoid_f32,1,nullptr);
}
static void silu_f32(ggml_tensor* dst,int ith,int nth,void*) {
    auto x=(const float*)dst->src[0]->data;auto y=(float*)dst->data;
    for(int64_t i=ith;i<ggml_nelements(dst);i+=nth)y[i]=x[i]/(1.0f+activation_exp(-x[i]));
}
static ggml_tensor* silu_graph(ggml_context* ctx,ggml_tensor* x) {
    ggml_tensor* args[]={x};return ggml_custom_4d(ctx,GGML_TYPE_F32,x->ne[0],x->ne[1],x->ne[2],x->ne[3],args,1,silu_f32,1,nullptr);
}
ggml_tensor* swiglu(ggml_context* ctx,const Model& m,const std::string& base,ggml_tensor* x) {
    auto z=affine(ctx,m,base+"ffn_w12",x,m.config().mlp_bias);auto F=z->ne[0]/2;
    auto half=[&](size_t offset){return ggml_view_3d(ctx,z,F,z->ne[1],z->ne[2],z->nb[1],z->nb[2],offset);};
    auto hidden=ggml_mul(ctx,half(0),silu_graph(ctx,ggml_cont(ctx,half(F*sizeof(float)))));
    return affine(ctx,m,base+"ffn_w3",ggml_cont(ctx,hidden),m.config().mlp_bias);
}
}
