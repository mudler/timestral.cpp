#include "timestral_capi.h"
#include "capi_internal.hpp"
#include "fixture.hpp"
#include <climits>
#include <cstddef>
#include <iostream>
#include <memory>
#include <thread>
#ifdef TS_CAPI_SHARED_PATH
#include <dlfcn.h>
#endif
extern "C" {size_t capi_c_size(void);size_t capi_c_offset(int);int capi_c_smoke(void);}
using Handle=std::unique_ptr<timestral_ctx,decltype(&timestral_capi_free)>;
size_t checks=0,transports=0;
void check(bool b,const char* msg){++checks;require(b,msg);}
Handle load(const std::string& p,int n=1){Handle h(timestral_capi_load(p.c_str(),n),timestral_capi_free);check(bool(h),timestral_capi_last_error(nullptr));return h;}
void bad(timestral_ctx* h,timestral_forecast_request r){int32_t q=99;check(!timestral_capi_forecast(h,&r,&q),"expected forecast rejection");check(q==0,"out_q reset");check(*timestral_capi_last_error(h),"missing error");}
void units(const std::string& path) {
    check(capi_c_smoke()==1 && capi_c_size()==sizeof(timestral_forecast_request),"C99 size/version");
#define O(x) offsetof(timestral_forecast_request,x)
    size_t offsets[]={O(context),O(observation_mask),O(B),O(V),O(T),O(prediction_length),O(past_cov),O(c_past),O(known_cov),O(c_known),O(cat_past),O(cc_past),O(cat_known),O(cc_known),O(symmetric_averaging),O(clip_nonnegative),O(max_context)};
    for(int i=0;i<17;++i)check(offsets[i]==capi_c_offset(i),"C99 field offset");
    check(!timestral_capi_load("nonexistent-model",1),"bad file");std::string e=timestral_capi_last_error(nullptr);
    timestral_capi_abi_version();timestral_capi_free_buffer(nullptr);check(e==timestral_capi_last_error(nullptr),"nonclearing calls");
    std::thread other([]{require(!*timestral_capi_last_error(nullptr),"TLS starts empty");timestral_capi_num_quantiles(nullptr);require(*timestral_capi_last_error(nullptr),"TLS own error");});other.join();check(e==timestral_capi_last_error(nullptr),"TLS isolated");
    check(!timestral_capi_load(nullptr,1) && !timestral_capi_load(path.c_str(),0),"load errors");
    auto h=load(path,2);check(!*timestral_capi_last_error(nullptr),"load clears TLS");
    check(std::isnan(timestral_capi_quantile_level(h.get(),-1)),"invalid index");check(*timestral_capi_last_error(h.get()),"query error");check(timestral_capi_num_quantiles(h.get())>0 && !*timestral_capi_last_error(h.get()),"query clears");
    float x=1;timestral_forecast_request r{};r.context=&x;r.B=r.V=r.T=r.prediction_length=1;r.symmetric_averaging=r.clip_nonnegative=-1;
    auto t=r;t.B=0;bad(h.get(),t);t=r;t.c_past=-1;bad(h.get(),t);t=r;t.c_past=1;bad(h.get(),t);t=r;t.past_cov=&x;bad(h.get(),t);
    t=r;t.c_known=1;bad(h.get(),t);t=r;t.known_cov=&x;bad(h.get(),t);
    double code=1;t=r;t.cc_past=1;bad(h.get(),t);t=r;t.cat_past=&code;bad(h.get(),t);
    t=r;t.cc_known=1;bad(h.get(),t);t=r;t.cat_known=&code;bad(h.get(),t);
    t=r;t.context=nullptr;bad(h.get(),t);
    t=r;t.symmetric_averaging=2;bad(h.get(),t);t=r;t.clip_nonnegative=-2;bad(h.get(),t);t=r;t.max_context=-1;bad(h.get(),t);
    t=r;t.B=t.V=t.T=INT_MAX;bad(h.get(),t);
    t=r;t.B=t.V=t.prediction_length=INT_MAX;bad(h.get(),t);
    uint8_t mask=2;t=r;t.observation_mask=&mask;bad(h.get(),t);
    int32_t q=9;check(!timestral_capi_forecast(nullptr,&r,&q) && q==0 && *timestral_capi_last_error(nullptr),"null handle");
    check(!timestral_capi_forecast(h.get(),nullptr,&q),"null request");check(!timestral_capi_forecast(h.get(),&r,nullptr),"null out_q");
    auto config=ts::Model::load(path).config();config.block_schedule={ts::LayerKind::Time};
    bool unsupported=false;try{ts::capi_detail::validate_covariates(config,true,true,0);}catch(const ts::Error& e){unsupported=std::string(e.what()).find("known-future")!=std::string::npos;}check(unsupported,"unsupported known covariates private validation");
    ts::Forecast f{1,1,1,{.5},{NAN}};bool rejected=false;try{auto p=ts::capi_detail::copy_output(f,1,1,1,1);timestral_capi_free_buffer(p);}catch(const ts::Error&){rejected=true;}check(rejected,"nonfinite output seam");
    ts::set_cpu_threads(3);float* out=timestral_capi_forecast(h.get(),&r,&q);check(out && q>0 && ts::cpu_threads()==3,"forecast/restored threads");std::vector<float> saved(out,out+q);
    auto h2=load(path,1);float* next=timestral_capi_forecast(h2.get(),&r,&q);check(next && ts::cpu_threads()==3 && !std::memcmp(out,saved.data(),saved.size()*4),"independent handles/output ownership");timestral_capi_free_buffer(next);timestral_capi_free_buffer(out);
    float known_pair[2]={1,1};t=r;t.known_cov=known_pair;t.c_known=1;t.max_context=INT_MAX;bad(h.get(),t);check(ts::cpu_threads()==3,"threads restored on validation error");
    double infinity=INFINITY;t=r;t.cat_past=&infinity;t.cc_past=1;bad(h.get(),t);
    check(ts::cpu_threads()==3,"threads restored on Forecaster exception");
    out=timestral_capi_forecast(h.get(),&r,&q);check(out && !*timestral_capi_last_error(h.get()),"forecast clears previous error");timestral_capi_free_buffer(out);ts::set_cpu_threads(1);
#ifdef TS_CAPI_SHARED_PATH
    void* lib=dlopen(TS_CAPI_SHARED_PATH,RTLD_NOW);check(lib,"dlopen");
    for(auto n:{"abi_version","load","free","last_error","num_quantiles","quantile_level","forecast","free_buffer"})check(dlsym(lib,(std::string("timestral_capi_")+n).c_str()),"dlsym");dlclose(lib);
#endif
}
void transport(const std::string& model,const std::string& base,const std::string& name,int H) {
    auto m=ts::Model::load(model);auto h=load(model);Fixture f(base+"/baseline-"+name+"-f32.gguf");auto shape=f.tensor("input.context");
    auto x=f.f("input.context");timestral_forecast_request r{};r.context=x.data();r.B=shape->ne[2];r.V=shape->ne[1];r.T=shape->ne[0];r.prediction_length=H;r.symmetric_averaging=r.clip_nonnegative=-1;
    ts::ForecastRequest n;n.batch=r.B;n.targets=r.V;n.length=r.T;n.horizon=H;n.context=x;
    std::vector<float> past,known;std::vector<double> cp,ck;
    if(ggml_get_tensor(f.b.ctx,"input.past")){past=f.f("input.past");r.past_cov=past.data();r.c_past=f.tensor("input.past")->ne[1];n.past=past;n.past_count=r.c_past;}
    if(ggml_get_tensor(f.b.ctx,"input.known")){known=f.f("input.known");r.known_cov=known.data();r.c_known=f.tensor("input.known")->ne[1];n.known=known;n.known_count=r.c_known;}
    if(ggml_get_tensor(f.b.ctx,"input.cat_past")){cp=f.get<double>("input.cat_past",GGML_TYPE_F64);r.cat_past=cp.data();r.cc_past=f.tensor("input.cat_past")->ne[1];n.categorical_past=cp;n.categorical_past_count=r.cc_past;}
    if(ggml_get_tensor(f.b.ctx,"input.cat_known")){ck=f.get<double>("input.cat_known",GGML_TYPE_F64);r.cat_known=ck.data();r.cc_known=f.tensor("input.cat_known")->ne[1];n.categorical_known=ck;n.categorical_known_count=r.cc_known;}
    ts::Forecaster fc(m);ts::set_cpu_threads(1);auto expected=fc.forecast(n);int32_t Q=0;auto out=timestral_capi_forecast(h.get(),&r,&Q);
    check(out,timestral_capi_last_error(h.get()));check(Q==int(expected.quantile_levels.size()),"Q mismatch");check(!std::memcmp(out,expected.quantiles.data(),expected.quantiles.size()*4),"transport bytes");
    for(int i=0;i<Q;++i)check(timestral_capi_quantile_level(h.get(),i)==expected.quantile_levels[i],"levels");timestral_capi_free_buffer(out);++transports;
    // Explicit boolean mask and flag overrides use the same copied input.
    std::vector<uint8_t> observed(x.size(),1);observed[0]=0;
    r.observation_mask=observed.data();n.observation_mask.assign(observed.begin(),observed.end());
    r.symmetric_averaging=r.clip_nonnegative=0;n.symmetric_averaging=false;n.clip_nonnegative=false;
    auto explicit_expected=fc.forecast(n);out=timestral_capi_forecast(h.get(),&r,&Q);
    check(out,timestral_capi_last_error(h.get()));check(!std::memcmp(out,explicit_expected.quantiles.data(),explicit_expected.quantiles.size()*4),"explicit mask/flags bytes");timestral_capi_free_buffer(out);++transports;
    std::cout<<"TRANSPORT "<<name<<" bytes="<<expected.quantiles.size()*4<<" PASS\n";
}
int main(int argc,char** argv){try {
    bool real=argc>1 && std::string(argv[1])=="--real";
    if(real){const char* p=getenv("TIMESTRAL_TEST_GGUF"),*s=getenv("TIMESTRAL_TEST_REAL_BASELINES");if(!p||!s)return 77;transport(p,std::string(s)+"/real","uni_short",96);transport(p,std::string(s)+"/real","known_cov",64);}
    else {std::string model=TS_SOURCE_DIR "/tests/fixtures/tiny/timestral-tiny-f32.gguf";units(model);const char* s=getenv("TIMESTRAL_TEST_TINY");if(!s)return 77;transport(model,std::string(s)+"/tiny","uni_short",24);transport(model,std::string(s)+"/tiny","known_cov",16);transport(model,std::string(s)+"/tiny","cat_uni",10);}
    std::cout<<"PASS capi checks="<<checks<<" transports="<<transports<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
