#include "layers.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>
// Frozen pre-experiment scalar callback from 2adffc0. Its exp polynomial
// retains the provenance and terms in third_party/notices/PyTorch.txt.
namespace oracle {
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
        for(int64_t j=0;j<keys;++j){p[j]=j<end?attention_exp(logits[j]-mx):float(std::exp(double(logits[j]-mx)));if(j<end)sums[j%8]+=p[j];}
        float a0=sums[0]+sums[4],a1=sums[1]+sums[5],a2=sums[2]+sums[6],a3=sums[3]+sums[7];
        float sum=(a0+a2)+(a1+a3);for(int64_t j=end;j<keys;++j)sum+=p[j];
        // CPU flash reference normalizes after the value contraction, not
        // each probability before it. Preserve that rounding boundary.
        float reciprocal=1.0f/float(sum);
        // Narrow output GEMM uses separate multiply/add; wide panels use FMA.
        for(int64_t d=0;d<hd;++d){float y=0;for(int64_t j=0;j<keys;++j){if(hd<16)y+=p[j]*V[(r*S+j)*D+h*hd+d];else y=std::fma(p[j],V[(r*S+j)*D+h*hd+d],y);}Y[(r*S+t)*D+h*hd+d]=float(y)*reciprocal;}
    }
}
}
int main() {
    std::mt19937 rng(821);size_t cases=0,values=0;
    for(int hd:{1,7,8,15,16,17,23,32,63,64,65})
    for(int seq:{1,3,7,8,9,17,33})
    for(int split:{0,1,seq/2,seq})
    for(int pattern=0;pattern<4;++pattern) {
        const int heads=2,rows=2,D=hd*heads;
        std::vector<float> q(rows*seq*D),k(q.size()),v(q.size()),mask(rows*seq*seq);
        auto fill=[&](std::vector<float>& x,bool value) {
            for(size_t i=0;i<x.size();++i) {
                if(pattern==0)x[i]=float(int(rng()%2049)-1024)/1024;
                if(pattern==1)x[i]=std::ldexp(float(int(rng()%3)-1),value?int(rng()%81)-40:-3);
                if(pattern==2)x[i]=i%4==0?0.f:i%4==1?-0.f:i%4==2?1.e-38f:-1.e-38f;
                if(pattern==3)x[i]=float(int(i%3)-1)*.125f;
            }
        };
        fill(q,false);fill(k,false);fill(v,true);
        for(int r=0;r<rows;++r)for(int t=0;t<seq;++t)for(int j=0;j<seq;++j)
            mask[(r*seq+t)*seq+j]=pattern==3?-1.e30f:(j>t?-INFINITY:0.f);
        std::vector<float> expected(q.size()),out(q.size()+2,-12345);
        ggml_tensor qt{},kt{},vt{},mt{},dst{};
        qt.ne[0]=hd;qt.ne[1]=heads;qt.ne[2]=seq;qt.ne[3]=rows;
        qt.data=q.data();kt.data=k.data();vt.data=v.data();mt.data=mask.data();
        dst.src[0]=&qt;dst.src[1]=&kt;dst.src[2]=&vt;dst.src[3]=&mt;
        dst.data=expected.data();void* data=reinterpret_cast<void*>(uintptr_t(split));
        oracle::sdpa_f32(&dst,0,1,data);
        for(float f:expected)if(!std::isfinite(f)){std::cerr<<"nonfinite oracle\n";return 1;}
        dst.data=out.data()+1;
        for(int threads:{1,6}) {
            std::fill(out.begin()+1,out.end()-1,-12345);
            for(int ith=0;ith<threads;++ith)ts::detail::sdpa_f32(&dst,ith,threads,data);
            if(std::memcmp(expected.data(),out.data()+1,expected.size()*sizeof(float)) ||
               out.front()!=-12345 || out.back()!=-12345) {
                std::cerr<<"FAIL hd="<<hd<<" seq="<<seq<<" split="<<split
                         <<" pattern="<<pattern<<" threads="<<threads<<'\n';return 1;
            }
            ++cases;values+=expected.size();
        }
    }
    std::cout<<"PASS "<<cases<<" exact attention comparisons, "<<values<<" output values\n";
}
