#include "head.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>
using namespace ts;
static void check(bool ok,const char* why){if(!ok)throw Error(why);}
static float exp_reference(float x){if(!std::isfinite(x)||x<-87||x>88)return std::exp(x);float q=std::nearbyint(x*1.4426950408889634f),r=std::fma(q,-.693145751953125f,x);r=std::fma(q,-1.428606765330187045e-6f,r);float u=.000198527617612853646f;for(float c:{.00139304355252534151f,.00833336077630519867f,.0416664853692054749f,.166666671633720398f,.5f})u=std::fma(u,r,c);return std::ldexp(1.f+std::fma(r*r,u,r),int(q));}
static std::vector<float> contraction(const std::vector<float>& x,const ggml_tensor* w){size_t K=w->ne[0],N=w->ne[1],rows=x.size()/K;auto weights=(const float*)w->data;std::vector<float> y(rows*N);for(size_t r=0;r<rows;++r)for(size_t n=0;n<N;++n){double sum=0;for(size_t k=0;k<K;++k)sum+=double(weights[n*K+k])*double(x[r*K+k]);y[r*N+n]=float(sum);}return y;}
static void full_route(){
 auto model=Model::load(std::string(TS_SOURCE_DIR)+"/tests/fixtures/tiny/timestral-tiny-f32.gguf");size_t D=model.config().d_model;Head head(model);size_t cases=0;
 for(size_t rows:{1,2})for(size_t patches:{1,3})for(int pattern=0;pattern<3;++pattern){std::vector<float>x(rows*patches*D);for(size_t i=0;i<x.size();++i)x[i]=pattern==0?float(int(i%23)-11)/13.f:pattern==1?std::ldexp(float(int(i%3)-1),int(i%31)-15):std::nextafter(0.f,1.f);auto saved=x;auto h=contraction(x,model.tensor("head.hidden.weight"));for(auto& v:h)v=v/(1.f+exp_reference(-v));auto expected=contraction(h,model.tensor("head.out.weight")),residual=contraction(x,model.tensor("head.res.weight"));for(size_t i=0;i<expected.size();++i)expected[i]+=residual[i];for(int threads:{1,2,3,6}){set_cpu_threads(threads);auto actual=head.project(x,rows,patches);check(actual.size()==expected.size()&&!std::memcmp(actual.data(),expected.data(),actual.size()*4),"production Head::project binary64 oracle");check(!std::memcmp(x.data(),saved.data(),x.size()*4),"decoder mutated");++cases;}}
 set_cpu_threads(1);std::cout<<"PASS production Head::project oracle cases="<<cases<<'\n';
}
#ifdef TS_HEAD64_CALLBACK
static void direct(){size_t count=0;for(int K:{1,7,8,31,32,191,192,193,768,1024})for(int N:{1,5})for(int pattern=0;pattern<4;++pattern){const int rows=3;std::vector<float>w(K*N),x(K*rows),bias(N),expected(N*rows),out(N*rows+2,-12345);for(size_t i=0;i<w.size();++i)w[i]=float(int(i%17)-8)/19;for(size_t i=0;i<x.size();++i)x[i]=float(int(i%23)-11)/29;for(int i=0;i<N;++i)bias[i]=float(i)/7;
 if(pattern==1){std::fill(x.begin(),x.end(),1.f);for(size_t i=0;i<w.size();++i)w[i]=i%3==0?std::ldexp(1.f,54):i%3==1?1.f:-std::ldexp(1.f,54);}
 if(pattern==2){std::fill(x.begin(),x.end(),std::nextafter(0.f,1.f));for(size_t i=0;i<w.size();++i)w[i]=i%2?-.5f:1.f;}
 if(pattern==3){for(size_t i=0;i<x.size();++i)x[i]=i%2?-0.f:0.f;for(size_t i=0;i<w.size();++i)w[i]=i%2?std::nextafter(0.f,1.f):-std::nextafter(0.f,1.f);}
 auto sw=w,sx=x,sb=bias;for(bool use_bias:{false,true}){for(int r=0;r<rows;++r)for(int n=0;n<N;++n){double sum=0;for(int k=0;k<K;++k)sum+=double(w[n*K+k])*double(x[r*K+k]);float rounded=float(sum);expected[r*N+n]=use_bias?rounded+bias[n]:rounded;}
 ggml_tensor wt{},xt{},bt{},dst{};wt.ne[0]=K;wt.ne[1]=N;wt.data=w.data();xt.data=x.data();bt.data=bias.data();dst.ne[0]=N;dst.ne[1]=rows;dst.ne[2]=dst.ne[3]=1;dst.src[0]=&wt;dst.src[1]=&xt;dst.src[2]=use_bias?&bt:nullptr;dst.data=out.data()+1;
 for(int nth:{1,2,3,6}){std::fill(out.begin()+1,out.end()-1,-12345);std::vector<std::thread> threads;for(int ith=0;ith<nth;++ith)threads.emplace_back([&,ith]{detail::head_linear_f64(&dst,ith,nth,nullptr);});for(auto& t:threads)t.join();check(!std::memcmp(out.data()+1,expected.data(),expected.size()*4)&&out.front()==-12345&&out.back()==-12345,"direct head contraction/guards");check(!std::memcmp(w.data(),sw.data(),w.size()*4)&&!std::memcmp(x.data(),sx.data(),x.size()*4)&&!std::memcmp(bias.data(),sb.data(),bias.size()*4),"direct inputs mutated");++count;}}
 }std::cout<<"PASS direct binary64 head cases="<<count<<" concurrent nth=1,2,3,6\n";}
#endif
int main(){try{full_route();
#ifdef TS_HEAD64_CALLBACK
 direct();
#endif
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
