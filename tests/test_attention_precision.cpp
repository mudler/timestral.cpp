#include "layers.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <thread>
#include <limits>
#include <vector>
// Independent full scalar SDPA oracle: frozen non-QK boundaries, explicit new QK policy.
namespace precision_oracle {
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
        for(int64_t j=0;j<keys;++j){double accumulator=0;for(int64_t d=0;d<hd;++d)accumulator+=double(Q[(r*S+t)*D+h*hd+d])*double(K[(r*S+j)*D+h*hd+d]);float dot=float(accumulator);
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
static void direct_dots(){size_t tests=0;
 for(int keys:{3,4,5,7,8,9})for(int hd:{3,7,33,64,65})for(int pattern=0;pattern<4;++pattern){
  int stride=hd*3+1;std::vector<float> q(hd),k(keys*stride,12345),expected(keys),out(keys+2,-12345);std::vector<double> acc_expected(keys),acc_out(keys+2,-12345);
  for(int d=0;d<hd;++d)q[d]=pattern==0?std::numeric_limits<float>::denorm_min():1.f;
  for(int key=0;key<keys;++key)for(int d=0;d<hd;++d){float value=0;
   if(pattern==0)value=d%2?1.f:-.5f; // nonzero subnormal*normal products
   if(pattern==1)value=d%3==0?std::ldexp(1.f,54):d%3==1?1.f:-std::ldexp(1.f,54);
   if(pattern==2){q[d]=std::numeric_limits<float>::denorm_min();value=d%2?std::numeric_limits<float>::denorm_min():-std::numeric_limits<float>::denorm_min();}
   if(pattern==3){q[d]=d%2?-0.f:0.f;value=key%2?-1.f:1.f;}
   k[key*stride+d]=value*(key%2?-1.f:1.f);
  }
  for(int key=0;key<keys;++key){double sum=0;for(int d=0;d<hd;++d)sum+=double(q[d])*double(k[key*stride+d]);acc_expected[key]=sum;expected[key]=float(sum);}
  const auto sq=q,sk=k;
  for(bool scalar:{false,true}){ts::detail::qk64_dots(q.data(),k.data(),out.data()+1,keys,stride,hd,scalar,acc_out.data()+1);if(std::memcmp(acc_out.data()+1,acc_expected.data(),keys*8)||acc_out.front()!=-12345||acc_out.back()!=-12345||std::memcmp(out.data()+1,expected.data(),keys*4)||out.front()!=-12345||out.back()!=-12345||std::memcmp(q.data(),sq.data(),q.size()*4)||std::memcmp(k.data(),sk.data(),k.size()*4))throw std::runtime_error("direct dot policy/tails/immutability");++tests;}
  if(pattern==1&&hd==3&&expected[0]!=0.f)throw std::runtime_error("sequential-double cancellation order");
 }
 std::cout<<"PASS direct QK cases="<<tests<<" including nonzero subnormal products and order-sensitive cancellation\n";
}
int main(){direct_dots();size_t cases=0,values=0;
 for(int hd:{1,7,8,15,16,17,33,64,65})for(int seq:{1,3,4,5,7,8,9,17,33})
 for(int split:{0,seq/2,std::max(0,seq-3),seq})for(int pattern=0;pattern<4;++pattern){
  int H=pattern==3?12:pattern%3+1,R=2,D=H*hd;std::vector<float> q(R*seq*D),k(q.size()),v(q.size()),mask(R*seq*seq);
  for(size_t i=0;i<q.size();++i){q[i]=float(int(i%17)-8)/32;k[i]=float(int((i*7)%23)-11)/32;v[i]=float(int(i%31)-15)/16;
   if(pattern==1){q[i]=std::ldexp(float(int(i%3)-1),-12);k[i]=std::ldexp(i%2?1.f:-1.f,int(i%41)-20);v[i]=std::ldexp(float(int(i%3)-1),int(i%81)-40);}
   if(pattern==2){q[i]=std::numeric_limits<float>::denorm_min();k[i]=i%2?-1.f:.5f;v[i]=i%2?1.e-38f:-1.e-38f;}}
  for(int r=0;r<R;++r)for(int t=0;t<seq;++t)for(int j=0;j<seq;++j)mask[(r*seq+t)*seq+j]=pattern==3?-1.e30f:(j>t?-INFINITY:0.f);
  auto sq=q,sk=k,sv=v,sm=mask;std::vector<float> expected(q.size()),out(q.size()+2,-12345);
  ggml_tensor qt{},kt{},vt{},mt{},dst{};qt.ne[0]=hd;qt.ne[1]=H;qt.ne[2]=seq;qt.ne[3]=R;qt.data=q.data();kt.data=k.data();vt.data=v.data();mt.data=mask.data();dst.src[0]=&qt;dst.src[1]=&kt;dst.src[2]=&vt;dst.src[3]=&mt;dst.data=expected.data();void* data=reinterpret_cast<void*>(uintptr_t(split));precision_oracle::sdpa_f32(&dst,0,1,data);
  for(float x:expected)if(!std::isfinite(x))throw std::runtime_error("nonfinite oracle");
  dst.data=out.data()+1;
  for(bool scalar:{false,true})for(int nth:{1,2,3,6}){std::fill(out.begin()+1,out.end()-1,-12345);std::vector<std::thread> threads;for(int ith=0;ith<nth;++ith)threads.emplace_back([&,ith]{if(scalar)ts::detail::sdpa_f32_scalar(&dst,ith,nth,data);else ts::detail::sdpa_f32(&dst,ith,nth,data);});for(auto& thread:threads)thread.join();
   if(std::memcmp(expected.data(),out.data()+1,expected.size()*4)||out.front()!=-12345||out.back()!=-12345||std::memcmp(q.data(),sq.data(),q.size()*4)||std::memcmp(k.data(),sk.data(),k.size()*4)||std::memcmp(v.data(),sv.data(),v.size()*4)||std::memcmp(mask.data(),sm.data(),mask.size()*4))throw std::runtime_error("precision oracle/sentinel/immutability");++cases;values+=expected.size();}
 }
 std::cout<<"PASS binary64 QK full SDPA cases="<<cases<<" values="<<values<<" concurrent nth=1,2,3,6\n";
}
