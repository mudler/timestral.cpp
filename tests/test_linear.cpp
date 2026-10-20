#include "layers.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

// Original scalar callback, retained independently of the optimized kernel.
static std::vector<float> oracle(const std::vector<float>& w,const std::vector<float>& x,
                                 const float* b,int k,int n,int rows) {
    std::vector<float> y(n*rows);
    for(int i=0;i<n*rows;++i) {
        float sum=0;
        if(k>=192) {
            for(int start=0;start<k;start+=192) {
                float panel=0;
                for(int j=start;j<std::min(k,start+192);++j)
                    panel=std::fma(w[(i%n)*k+j],x[(i/n)*k+j],panel);
                sum+=panel;
            }
        } else {
            for(int j=0;j<k;++j)sum=std::fma(w[(i%n)*k+j],x[(i/n)*k+j],sum);
        }
        y[i]=b?sum+b[i%n]:sum;
    }
    return y;
}
int main() {
    std::mt19937 rng(739);
    size_t cases=0,values=0;
    for(int k:{1,7,32,64,191,192,193,383,384,385,768,2048})
    for(int n:{1,7,8,9,17,64,768})
    for(int rows:{1,3,17,31,32,33,34,35,36,39})
    for(int pattern=0;pattern<3;++pattern) {
        std::vector<float> w(k*n),x(k*rows),b(n),y(n*rows+2,-12345);
        auto fill=[&](std::vector<float>& v) {
            for(size_t i=0;i<v.size();++i) {
                if(pattern==0)v[i]=float(int(rng()%2049)-1024)/1024.f;
                // Cancellation, wide exponents, subnormals and signed zeros.
                if(pattern==1)v[i]=std::ldexp(float(int(rng()%3)-1),int(rng()%81)-40);
                if(pattern==2)v[i]=(i%4==0?0.f:i%4==1?-0.f:i%4==2?1.e-38f:-1.e-38f);
            }
        };
        fill(w);fill(x);fill(b);
        const auto saved_w=w,saved_x=x,saved_b=b;
        for(bool bias:{false,true}) {
            auto expected=oracle(w,x,bias?b.data():nullptr,k,n,rows);
            ggml_tensor wt{},xt{},bt{},dst{};
            wt.ne[0]=k;wt.ne[1]=n;wt.data=w.data();xt.data=x.data();bt.data=b.data();
            dst.ne[0]=n;dst.ne[1]=rows;dst.ne[2]=dst.ne[3]=1;
            dst.data=y.data()+1;dst.src[0]=&wt;dst.src[1]=&xt;dst.src[2]=bias?&bt:nullptr;
            for(int threads:{1,6}) {
                std::fill(y.begin()+1,y.end()-1,-12345);
                for(int ith=0;ith<threads;++ith)ts::detail::linear_f32(&dst,ith,threads,nullptr);
                if(std::memcmp(expected.data(),y.data()+1,expected.size()*sizeof(float)) ||
                   y.front()!=-12345 || y.back()!=-12345 ||
                   std::memcmp(w.data(),saved_w.data(),w.size()*sizeof(float)) ||
                   std::memcmp(x.data(),saved_x.data(),x.size()*sizeof(float)) ||
                   std::memcmp(b.data(),saved_b.data(),b.size()*sizeof(float))) {
                    std::cerr<<"exact parity failed K="<<k<<" N="<<n<<" rows="<<rows
                             <<" pattern="<<pattern<<" bias="<<bias<<" threads="<<threads<<'\n';return 1;
                }
                ++cases;values+=expected.size();
            }
        }
    }
    std::cout<<"PASS "<<cases<<" exact oracle comparisons, "<<values<<" output values\n";
}
