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
static void keypack_cases(bool expect_packing) {
    size_t bytes=0;
    if(!ts::detail::keypack_bytes(12,1056,64,16*1024*1024,bytes) || bytes!=3244032 ||
       ts::detail::keypack_bytes(SIZE_MAX,128,64,SIZE_MAX,bytes) ||
       ts::detail::keypack_bytes(12,1056,64,1,bytes))throw std::runtime_error("size guards");
    size_t cases=0,packed_heads=0,allocation_fallbacks=0,size_fallbacks=0;
    for(int seq:{127,128,129,132})for(int split:{0,seq-3,seq-4})
    for(int hd:{7,33,65})for(int threads:{1,4,6})
    for(int pattern=0;pattern<3;++pattern) {
        const int H=2,R=2,D=H*hd;
        std::vector<float> q(R*seq*D),k(q.size()),v(q.size()),mask(R*seq*seq);
        for(size_t i=0;i<q.size();++i) {
            q[i]=float(int(i%17)-8)/32;k[i]=float(int((i*7)%23)-11)/32;
            v[i]=i%4==0?-0.f:float(int(i%31)-15)/16;
            if(pattern==1) {
                // Alternating wide key exponents force cancellation while
                // bounded queries keep logits and oracle outputs finite.
                q[i]=std::ldexp(float(int(i%3)-1),-12);
                k[i]=std::ldexp(i%2?1.f:-1.f,int(i%41)-20);
                v[i]=std::ldexp(float(int(i%3)-1),int(i%81)-40);
            }
            if(pattern==2) {
                q[i]=i%2?-0.f:0.f;
                k[i]=i%4==0?-0.f:i%4==1?0.f:i%4==2?1.e-38f:-1.e-38f;
                v[i]=k[i];
            }
        }
        for(int r=0;r<R;++r)for(int t=0;t<seq;++t)for(int j=0;j<seq;++j)
            mask[(r*seq+t)*seq+j]=j>t?-INFINITY:0.f;
        const auto sq=q,sk=k,sv=v,sm=mask;
        std::vector<float> expected(q.size()),out(q.size()+2,-12345);
        ggml_tensor qt{},kt{},vt{},mt{},dst{};
        qt.ne[0]=hd;qt.ne[1]=H;qt.ne[2]=seq;qt.ne[3]=R;
        qt.data=q.data();kt.data=k.data();vt.data=v.data();mt.data=mask.data();
        dst.src[0]=&qt;dst.src[1]=&kt;dst.src[2]=&vt;dst.src[3]=&mt;
        void* data=reinterpret_cast<void*>(uintptr_t(split));dst.data=expected.data();
        oracle::sdpa_f32(&dst,0,1,data);
        for(float f:expected)if(!std::isfinite(f))throw std::runtime_error("nonfinite focused oracle");
        dst.data=out.data()+1;
        for(int mode=0;mode<4;++mode) {
            ts::detail::KeyPackProbe probe;
            probe.fail_allocation=mode==1;probe.force_overflow=mode==3;
            if(mode==2)probe.budget=1;
            std::fill(out.begin()+1,out.end()-1,-12345);
            for(int ith=0;ith<threads;++ith)ts::detail::sdpa_f32_probe(&dst,ith,threads,data,probe);
            if(std::memcmp(expected.data(),out.data()+1,expected.size()*sizeof(float)) ||
               out.front()!=-12345 || out.back()!=-12345 ||
               std::memcmp(q.data(),sq.data(),q.size()*4) || std::memcmp(k.data(),sk.data(),k.size()*4) ||
               std::memcmp(v.data(),sv.data(),v.size()*4) || std::memcmp(mask.data(),sm.data(),mask.size()*4))
                throw std::runtime_error("keypack exact/immutability");
            // H=2,nth=4 assigns alternating queries: 125/126 eligible queries
            // own 31/32 each; 129 owns 32/33, explicitly crossing the threshold.
            size_t expected_packs=0;
            if(expect_packing && seq>=128 && mode==0) {
                for(int r=0;r<R;++r)for(int h=0;h<H;++h)for(int ith=0;ith<threads;++ith) {
                    int owned=0;
                    for(int t=0;t<seq;++t) {
                        if(split && seq-split<4 && t>=split)continue;
                        if(((r*seq+t)*H+h)%threads==ith)++owned;
                    }
                    if(owned>=32)++expected_packs;
                }
            }
            if(probe.packed_heads!=expected_packs)
                throw std::runtime_error("dispatch/owned threshold/reset count");
            const size_t expected_alloc=expect_packing && seq>=128 && mode==1?size_t(threads):0;
            const size_t expected_size=expect_packing && seq>=128 && mode>=2?size_t(threads):0;
            if(probe.allocation_fallbacks!=expected_alloc || probe.size_fallbacks!=expected_size)
                throw std::runtime_error("dispatch/fallback coverage count");
            packed_heads+=probe.packed_heads;allocation_fallbacks+=probe.allocation_fallbacks;
            size_fallbacks+=probe.size_fallbacks;++cases;
        }
    }
    std::cout<<"PASS "<<cases<<" focused keypack comparisons packed_heads="<<packed_heads
             <<" allocation_fallbacks="<<allocation_fallbacks<<" size_fallbacks="<<size_fallbacks<<'\n';
}
int main(int argc,char** argv) {
    bool expect_packing=false;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    expect_packing=__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
    // The manually linked scalar-fallback executable must declare its build.
    if(argc==2 && std::string(argv[1])=="--expect-scalar")expect_packing=false;
    else if(argc!=1)throw std::runtime_error("usage: test_attention [--expect-scalar]");
    std::cout<<"EXPECT keypack="<<(expect_packing?"enabled":"disabled")<<'\n';
    keypack_cases(expect_packing);
    std::mt19937 rng(821);size_t cases=0,values=0;
    for(int hd:{1,7,8,15,16,17,23,31,32,33,63,64,65})
    for(int seq:{1,3,7,8,9,15,16,17,23,24,25,31,32,33,65})
    for(int split:{0,1,seq/2,std::max(0,seq-3),std::max(0,seq-4),seq})
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
