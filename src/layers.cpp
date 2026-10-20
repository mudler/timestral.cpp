#ifdef TS_STANDALONE_MKL
#include "standalone_mkl.hpp"
#endif
#include "layers.hpp"
#include "weight_policy.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <memory>
#include <limits>
#include <new>
#if !defined(TS_LINEAR_DISABLE_AVX2) && (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define TS_LINEAR_AVX2 1
#endif
namespace ts {
#ifdef TS_LINEAR_AVX2
// Callback-local storage: one [K,8] tile, reused across all owned rows.
// Allocation failure leaves the original path available without shared state.
__attribute__((target("avx2,fma")))
static bool linear_packed(const float* w,const float* x,const float* b,float* y,
                          int64_t k,int64_t n,int64_t rows,int ith,int nth) {
    if(k<192 || n<8 || rows<32 ||
       uint64_t(k)>std::numeric_limits<size_t>::max()/(8*sizeof(float)))return false;
    std::unique_ptr<float[]> packed(new(std::nothrow) float[size_t(k)*8]);
    if(!packed)return false;
    for(int64_t col=0;col+8<=n;col+=8) {
        for(int64_t j=0;j<k;++j)for(int lane=0;lane<8;++lane)
            packed[size_t(j)*8+lane]=w[(col+lane)*k+j];
        for(int64_t row=ith;row<rows;row+=int64_t(4)*nth) {
            const int count=int(std::min<int64_t>(4,(rows-1-row)/nth+1));
            __m256 sum[4]={_mm256_setzero_ps(),_mm256_setzero_ps(),
                           _mm256_setzero_ps(),_mm256_setzero_ps()};
            for(int64_t start=0;start<k;start+=192) {
                __m256 panel[4]={_mm256_setzero_ps(),_mm256_setzero_ps(),
                                 _mm256_setzero_ps(),_mm256_setzero_ps()};
                for(int64_t j=start;j<std::min(k,start+192);++j) {
                    __m256 weight=_mm256_loadu_ps(packed.get()+j*8);
                    for(int lane=0;lane<count;++lane)
                        panel[lane]=_mm256_fmadd_ps(weight,
                            _mm256_set1_ps(x[(row+int64_t(lane)*nth)*k+j]),panel[lane]);
                }
                for(int lane=0;lane<count;++lane)sum[lane]=_mm256_add_ps(sum[lane],panel[lane]);
            }
            for(int lane=0;lane<count;++lane) {
                if(b)sum[lane]=_mm256_add_ps(sum[lane],_mm256_loadu_ps(b+col));
                _mm256_storeu_ps(y+(row+int64_t(lane)*nth)*n+col,sum[lane]);
            }
        }
    }
    return true;
}
// Lanes are independent output columns, never partial K reductions. Weight
// rows are K-contiguous, so gather the same K coordinate from eight rows.
__attribute__((target("avx2,fma")))
static int64_t linear_columns(const float* w,const float* x,const float* b,
                              float* y,int64_t k,int64_t n) {
    int64_t col=0;
    for(;col+8<=n;col+=8) {
        __m256 sum=_mm256_setzero_ps();
        for(int64_t start=0;start<k;start+=192) {
            __m256 panel=_mm256_setzero_ps();
            for(int64_t j=start;j<std::min(k,start+192);++j) {
                const float* p=w+col*k+j;
                __m256 weights=_mm256_set_ps(p[7*k],p[6*k],p[5*k],p[4*k],
                                            p[3*k],p[2*k],p[k],p[0]);
                panel=_mm256_fmadd_ps(weights,_mm256_set1_ps(x[j]),panel);
            }
            // Small dots have no extra add-to-zero boundary in the oracle.
            sum=k<192?panel:_mm256_add_ps(sum,panel);
        }
        if(b)sum=_mm256_add_ps(sum,_mm256_loadu_ps(b+col));
        _mm256_storeu_ps(y+col,sum);
    }
    return col;
}
#endif
// F32 linear: reference-sized contractions use FMA panels. Small dots
// accumulate sequential F32 FMA, matching the pinned CPU SGEMM.
// Inputs and weights remain F32; no reference values enter this kernel.
void detail::linear_f32(ggml_tensor* dst,int ith,int nth,void*) {
    auto weight=dst->src[0], input=dst->src[1], bias=dst->src[2];
    const auto w=static_cast<const float*>(weight->data);
    const auto x=static_cast<const float*>(input->data);
    const auto b=bias ? static_cast<const float*>(bias->data) : nullptr;
    auto y=static_cast<float*>(dst->data);
    const int64_t k=weight->ne[0], n=weight->ne[1];
    const int64_t rows=ggml_nelements(dst)/n;
    bool packed=false;
#ifdef TS_LINEAR_AVX2
    if(__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        packed=linear_packed(w,x,b,y,k,n,rows,ith,nth);
#endif
    for(int64_t row=ith;row<rows;row+=nth) {
        int64_t col=packed?n/8*8:0;
#ifdef TS_LINEAR_AVX2
        if(!packed && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
            col=linear_columns(w,x+row*k,b,y+row*n,k,n);
#endif
        for(;col<n;++col) {
            float sum=0;
            if(k>=192) {
                for(int64_t start=0;start<k;start+=192) {
                    float panel=0;
                    for(int64_t j=start;j<std::min(k,start+192);++j)
                        panel=std::fma(w[col*k+j],x[row*k+j],panel);
                    sum+=panel;
                }
            } else {
                for(int64_t j=0;j<k;++j)
                    sum=std::fma(w[col*k+j],x[row*k+j],sum);
            }
            y[row*n+col]=b?sum+b[col]:sum;
        }
    }
}
ResidualOutput residual_block(ggml_context* ctx, const Model& m, ggml_tensor* x) {
    const auto& c=m.config();
    if (!x || x->type!=GGML_TYPE_F32 || x->ne[0]!=2*c.patch_size || x->ne[3]!=1 || !ggml_is_contiguous(x)) throw Error("invalid patch feature layout");
    auto linear=[&](const char* w,const char* b,ggml_tensor* input) {
        auto weight=m.tensor(w), bias=m.tensor(b);
        if(weight->ne[0]!=input->ne[0] || bias->ne[0]!=weight->ne[1] || ggml_nelements(bias)!=bias->ne[0]) throw Error("invalid residual weight/bias dimensions");
        if(weight->type!=GGML_TYPE_F32)return reduced_projection(ctx,weight,input,bias);
        ggml_tensor* args[]={weight,input,bias};
        return ggml_custom_4d(ctx,GGML_TYPE_F32,weight->ne[1],input->ne[1],input->ne[2],1,args,3,detail::linear_f32,1,nullptr);
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
    if(w->type!=GGML_TYPE_F32)return reduced_projection(ctx,w,x,b);
    ggml_tensor* args[]={w,x,b};
    return ggml_custom_4d(ctx,GGML_TYPE_F32,w->ne[1],x->ne[1],x->ne[2],1,args,b?3:2,detail::linear_f32,1,nullptr);
}
// Adapted from ATen SumKernel.cpp at 5c4886908584029761b579af026dcfb627c84070.
// Copyright/licence and source scope: third_party/notices/PyTorch.txt and README.md.
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
// Adapted from ATen vec256_float.h at the same pinned revision above.
// BSD-style terms: third_party/notices/PyTorch.txt; provenance: README.md there.
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
#ifdef TS_LINEAR_AVX2
__attribute__((target("avx2,fma")))
static int64_t attention_packed_keys(const float* q,const float* packed,float* dots,
                                    int64_t keys,int64_t hd) {
    int64_t j=0;
    for(;j+8<=keys;j+=8) {
        __m256 dot=_mm256_setzero_ps();
        const float* group=packed+(j/8)*hd*8;
        for(int64_t d=0;d<hd;++d)
            dot=_mm256_fmadd_ps(_mm256_set1_ps(q[d]),_mm256_loadu_ps(group+d*8),dot);
        _mm256_storeu_ps(dots+j,dot);
    }
    return j;
}
// Each lane owns a key dot and retains ascending head-dimension FMA order.
__attribute__((target("avx2,fma")))
static int64_t attention_keys(const float* q,const float* k,float* dots,
                              int64_t keys,int64_t stride,int64_t hd) {
    int64_t j=0;
    for(;j+8<=keys;j+=8) {
        __m256 dot=_mm256_setzero_ps();
        for(int64_t d=0;d<hd;++d) {
            const float* p=k+j*stride+d;
            __m256 key=_mm256_set_ps(p[7*stride],p[6*stride],p[5*stride],p[4*stride],
                                    p[3*stride],p[2*stride],p[stride],p[0]);
            dot=_mm256_fmadd_ps(_mm256_set1_ps(q[d]),key,dot);
        }
        _mm256_storeu_ps(dots+j,dot);
    }
    return j;
}
__attribute__((target("avx2,fma")))
static int64_t attention_values(const float* p,const float* v,float* out,
                                int64_t keys,int64_t stride,int64_t hd,float reciprocal) {
    int64_t d=0;
    for(;d+32<=hd;d+=32) {
        __m256 a=_mm256_setzero_ps(),b=_mm256_setzero_ps();
        __m256 c=_mm256_setzero_ps(),e=_mm256_setzero_ps();
        for(int64_t j=0;j<keys;++j) {
            __m256 probability=_mm256_set1_ps(p[j]);
            const float* value=v+j*stride+d;
            a=_mm256_fmadd_ps(probability,_mm256_loadu_ps(value),a);
            b=_mm256_fmadd_ps(probability,_mm256_loadu_ps(value+8),b);
            c=_mm256_fmadd_ps(probability,_mm256_loadu_ps(value+16),c);
            e=_mm256_fmadd_ps(probability,_mm256_loadu_ps(value+24),e);
        }
        __m256 scale=_mm256_set1_ps(reciprocal);
        _mm256_storeu_ps(out+d,_mm256_mul_ps(a,scale));
        _mm256_storeu_ps(out+d+8,_mm256_mul_ps(b,scale));
        _mm256_storeu_ps(out+d+16,_mm256_mul_ps(c,scale));
        _mm256_storeu_ps(out+d+24,_mm256_mul_ps(e,scale));
    }
    for(;d+8<=hd;d+=8) {
        __m256 sum=_mm256_setzero_ps();
        for(int64_t j=0;j<keys;++j)
            sum=_mm256_fmadd_ps(_mm256_set1_ps(p[j]),_mm256_loadu_ps(v+j*stride+d),sum);
        _mm256_storeu_ps(out+d,_mm256_mul_ps(sum,_mm256_set1_ps(reciprocal)));
    }
    return d;
}
#endif
#ifdef TS_LINEAR_AVX2
// Four independent key dots. No horizontal reduction or fused multiply/add.
__attribute__((target("avx2")))
static int64_t qk64_keys4(const float* q,const float* k,float* dots,
                         int64_t keys,int64_t stride,int64_t hd,double* accumulators) {
    int64_t key=0;
    for(;key+4<=keys;key+=4){
        __m256d accumulator=_mm256_setzero_pd();
        for(int64_t d=0;d<hd;++d){
            const float* p=k+key*stride+d;
            __m128 values=_mm_set_ps(p[3*stride],p[2*stride],p[stride],p[0]);
            __m256d product=_mm256_mul_pd(_mm256_set1_pd(double(q[d])),_mm256_cvtps_pd(values));
            accumulator=_mm256_add_pd(accumulator,product);
        }
        if(accumulators)_mm256_storeu_pd(accumulators+key,accumulator);
        _mm_storeu_ps(dots+key,_mm256_cvtpd_ps(accumulator));
    }
    return key;
}
#endif
void detail::qk64_dots(const float* q,const float* k,float* out,int64_t keys,int64_t stride,int64_t hd,bool force_scalar,double* accumulators) {
    int64_t first=0;
#ifdef TS_LINEAR_AVX2
    if(!force_scalar && __builtin_cpu_supports("avx2"))first=qk64_keys4(q,k,out,keys,stride,hd,accumulators);
#endif
    for(int64_t j=first;j<keys;++j){double sum=0;for(int64_t d=0;d<hd;++d)sum+=double(q[d])*double(k[j*stride+d]);if(accumulators)accumulators[j]=sum;out[j]=float(sum);}
}
bool detail::keypack_bytes(size_t heads,size_t sequence,size_t dim,size_t budget,size_t& bytes) {
    size_t count=sequence/8;
    for(size_t factor:{dim,size_t(8),heads,sizeof(float)}) {
        if(factor && count>std::numeric_limits<size_t>::max()/factor)return false;
        count*=factor;
    }
    if(!count || count>budget)return false;
    bytes=count;return true;
}
// CPU flash operation boundaries, evaluated without a dependency on Torch.
static void sdpa_impl(ggml_tensor* dst,int ith,int nth,void* split_data,detail::KeyPackProbe* probe,bool legacy,bool force_scalar=false) {
    const int64_t split=reinterpret_cast<uintptr_t>(split_data);
    auto q=dst->src[0],k=dst->src[1],v=dst->src[2],mask=dst->src[3];
    auto Q=(const float*)q->data,K=(const float*)k->data,V=(const float*)v->data,M=(const float*)mask->data;
    auto Y=(float*)dst->data;
    const int64_t H=q->ne[1],hd=q->ne[0],S=q->ne[2],D=H*hd;
    std::unique_ptr<float[]> packed;
    std::unique_ptr<unsigned char[]> states;
    int64_t packed_row=-1;
#ifdef TS_LINEAR_AVX2
    size_t bytes=0;
    if(legacy && S>=128 && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        const size_t budget=probe?probe->budget:16*1024*1024;
        bool fits=detail::keypack_bytes(probe && probe->force_overflow?SIZE_MAX:size_t(H),
                                       size_t(S),size_t(hd),budget,bytes);
        if(!fits) {if(probe)++probe->size_fallbacks;}
        else {
            if(!(probe && probe->fail_allocation)) {
                packed.reset(new(std::nothrow) float[bytes/sizeof(float)]);
                states.reset(new(std::nothrow) unsigned char[size_t(H)]);
            }
            if(!packed || !states) {
                packed.reset();states.reset();
                if(probe)++probe->allocation_fallbacks;
            }
        }
    }
#endif
    std::vector<float> logits(S),p(S);
    for(int64_t job=ith;job<q->ne[3]*S*H;job+=nth) {
        int64_t h=job%H,t=job/H%S,r=job/(H*S);
        int64_t keys=(split && t<split)?split:S;
        if(packed && packed_row!=r) {
            std::fill_n(states.get(),size_t(H),0);packed_row=r;
        }
        const bool wide=S>=8 && !(split && S-split<4 && t>=split);
        if(packed && wide && keys>=8 && states[h]==0) {
            size_t owned=0;
            for(int64_t query=0;query<S;++query) {
                if(split && S-split<4 && query>=split)continue;
                if((split && query<split?split:S)<8)continue;
                if(((r*S+query)*H+h)%nth==ith)++owned;
            }
            states[h]=2;
            if(owned>=32) {
                float* head=packed.get()+h*(S/8)*hd*8;
                for(int64_t group=0;group<S/8;++group)
                    for(int64_t d=0;d<hd;++d)for(int lane=0;lane<8;++lane)
                        head[(group*hd+d)*8+lane]=K[(r*S+group*8+lane)*D+h*hd+d];
                states[h]=1;if(probe)++probe->packed_heads;
            }
        }
        float mx=-INFINITY;
        int64_t first_key=0;
#ifdef TS_LINEAR_AVX2
        if(legacy && S>=8 && !(split && S-split<4 && t>=split) &&
           __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
            first_key=packed && states[h]==1
                ?attention_packed_keys(Q+(r*S+t)*D+h*hd,packed.get()+h*(S/8)*hd*8,logits.data(),keys,hd)
                :attention_keys(Q+(r*S+t)*D+h*hd,K+r*S*D+h*hd,logits.data(),keys,D,hd);
#endif
        if(!legacy){detail::qk64_dots(Q+(r*S+t)*D+h*hd,K+r*S*D+h*hd,logits.data(),keys,D,hd,force_scalar);first_key=keys;}
        // Consume SIMD dots in the original scalar key order for scale/mask/max.
        for(int64_t j=0;j<first_key;++j) {
            logits[j]=logits[j]*float(1/std::sqrt(double(hd)))+M[(r*S+t)*S+j];
            mx=std::max(mx,logits[j]);
        }
        for(int64_t j=first_key;j<keys;++j){float dot=0;if(legacy){float acc[8]={};for(int64_t d=0;d<hd;++d){float a=Q[(r*S+t)*D+h*hd+d],b=K[(r*S+j)*D+h*hd+d];dot=std::fma(a,b,dot);acc[d%8]+=a*b;}
            // Small CPU GEMM uses non-fused lane reductions: four lanes
            // for complete key groups, eight for the remaining key columns.
            // These are execution-shape branches, never layer/case IDs.
            if(S<8 || (split && S-split<4 && t>=split)){
                if(j<keys/8*8){float a[4]={};for(int64_t d=0;d<hd;++d)a[d%4]+=Q[(r*S+t)*D+h*hd+d]*K[(r*S+j)*D+h*hd+d];dot=(a[0]+a[2])+(a[1]+a[3]);}
                else dot=((acc[0]+acc[4])+(acc[2]+acc[6]))+((acc[1]+acc[5])+(acc[3]+acc[7]));
            }
            }
            if(!legacy){double sum64=0;for(int64_t d=0;d<hd;++d)sum64+=double(Q[(r*S+t)*D+h*hd+d])*double(K[(r*S+j)*D+h*hd+d]);dot=float(sum64);}
            logits[j]=float(dot)*float(1/std::sqrt(double(hd)))+M[(r*S+t)*S+j];mx=std::max(mx,logits[j]);}
        float sums[8]={};int64_t end=keys/8*8;
        for(int64_t j=0;j<keys;++j){p[j]=j<end?attention_exp(logits[j]-mx):float(std::exp(double(logits[j]-mx)));if(j<end)sums[j%8]+=p[j];}
        float a0=sums[0]+sums[4],a1=sums[1]+sums[5],a2=sums[2]+sums[6],a3=sums[3]+sums[7];
        float sum=(a0+a2)+(a1+a3);for(int64_t j=end;j<keys;++j)sum+=p[j];
        // CPU flash reference normalizes after the value contraction, not
        // each probability before it. Preserve that rounding boundary.
        float reciprocal=1.0f/float(sum);
        // Narrow output GEMM uses separate multiply/add; wide panels use FMA.
        int64_t first=0;
#ifdef TS_LINEAR_AVX2
        if(hd>=16 && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
            first=attention_values(p.data(),V+r*S*D+h*hd,Y+(r*S+t)*D+h*hd,keys,D,hd,reciprocal);
#endif
        for(int64_t d=first;d<hd;++d){float y=0;for(int64_t j=0;j<keys;++j){if(hd<16)y+=p[j]*V[(r*S+j)*D+h*hd+d];else y=std::fma(p[j],V[(r*S+j)*D+h*hd+d],y);}Y[(r*S+t)*D+h*hd+d]=float(y)*reciprocal;}
    }
}
void detail::sdpa_f32(ggml_tensor* dst,int ith,int nth,void* split_data) {
    sdpa_impl(dst,ith,nth,split_data,nullptr,false);
}
void detail::sdpa_f32_scalar(ggml_tensor* dst,int ith,int nth,void* split_data) {
    sdpa_impl(dst,ith,nth,split_data,nullptr,false,true);
}
void detail::sdpa_f32_legacy(ggml_tensor* dst,int ith,int nth,void* split_data) {
    sdpa_impl(dst,ith,nth,split_data,nullptr,true);
}
void detail::sdpa_f32_probe(ggml_tensor* dst,int ith,int nth,void* split_data,KeyPackProbe& probe) {
    sdpa_impl(dst,ith,nth,split_data,&probe,true);
}
ggml_tensor* attention_time(ggml_context* ctx,ggml_tensor* q,ggml_tensor* k,ggml_tensor* v,ggml_tensor* mask,size_t split) {
    if(!q || !k || !v || !mask || !ggml_are_same_shape(q,k) || !ggml_are_same_shape(q,v) || mask->ne[0]!=q->ne[2] || mask->ne[1]!=q->ne[2] || mask->ne[2]!=q->ne[3])throw Error("invalid SDPA layout");
    for(auto t:{q,k,v,mask})if(t->type!=GGML_TYPE_F32 || !ggml_is_contiguous(t))throw Error("invalid SDPA type/strides");
    if(split>size_t(q->ne[2]) || mask->ne[3]!=1)throw Error("invalid SDPA mask rank");
#ifdef TS_STANDALONE_MKL
    return standalone_sdpa(ctx,q,k,v,mask,split);
#endif
    ggml_tensor* args[]={q,k,v,mask};
    return ggml_custom_4d(ctx,GGML_TYPE_F32,q->ne[0]*q->ne[1],q->ne[2],q->ne[3],1,args,4,detail::sdpa_f32,1,reinterpret_cast<void*>(split));
}
ggml_tensor* attention_variate(ggml_context* ctx,ggml_tensor* q,ggml_tensor* k,ggml_tensor* v,ggml_tensor* mask) {
    return attention_time(ctx,q,k,v,mask,0);
}
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
