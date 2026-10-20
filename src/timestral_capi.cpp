#include "timestral_capi.h"
#include "capi_internal.hpp"
#include <climits>
#include <memory>
#include <string>
#include <utility>
namespace {
struct ErrorText {
    std::string text;
    bool fallback=false;
    void clear() noexcept {text.clear();fallback=false;}
    void set(const char* message) noexcept {
        try {text=message?message:"unknown C ABI error";fallback=false;}
        catch(...) {fallback=true;}
    }
    const char* get() const noexcept {return fallback?"C ABI error (diagnostic allocation failed)":text.c_str();}
};
thread_local ErrorText tls_error;
struct ThreadGuard {
    int old;
    explicit ThreadGuard(int n):old(ts::cpu_threads()){ts::set_cpu_threads(n);}
    ~ThreadGuard(){try{ts::set_cpu_threads(old);}catch(...){}}
};
void require(bool ok,const char* msg){if(!ok)throw ts::Error(msg);}
void pointer_count(const void* p,int32_t n){require(n>=0,"negative covariate count");require((n!=0)==(p!=nullptr),"covariate pointer/count mismatch");}
}
struct timestral_ctx {
    ts::Model model;
    ts::Forecaster forecaster;
    int threads;
    ErrorText error;
    timestral_ctx(ts::Model&& m,int n):model(std::move(m)),forecaster(model),threads(n){}
};
namespace {
ErrorText& error(timestral_ctx* c) noexcept {return c?c->error:tls_error;}
void caught(timestral_ctx* c) noexcept {
    try {throw;}catch(const std::exception& e){error(c).set(e.what());}
    catch(...){error(c).set("unknown C ABI exception");}
}
}
extern "C" {
int timestral_capi_abi_version(void){return 1;}
timestral_ctx* timestral_capi_load(const char* path,int threads) {
    try {
        require(path && *path,"model path is required");require(threads>=1 && threads<=1024,"invalid CPU thread count");
        auto p=std::make_unique<timestral_ctx>(ts::Model::load(path),threads);
        tls_error.clear();return p.release();
    }catch(...){caught(nullptr);return nullptr;}
}
void timestral_capi_free(timestral_ctx* c){try{delete c;}catch(...){caught(nullptr);}}
const char* timestral_capi_last_error(timestral_ctx* c){return error(c).get();}
int timestral_capi_num_quantiles(timestral_ctx* c) {
    try {require(c,"null handle");auto n=c->model.config().num_quantiles();require(n<=INT_MAX,"quantile count overflow");c->error.clear();return int(n);}
    catch(...){caught(c);return 0;}
}
double timestral_capi_quantile_level(timestral_ctx* c,int i) {
    try {require(c,"null handle");const auto& q=c->model.config().quantile_knots;require(i>=0 && size_t(i)<q.size(),"quantile index out of range");double v=q[size_t(i)];c->error.clear();return v;}
    catch(...){caught(c);return std::numeric_limits<double>::quiet_NaN();}
}
float* timestral_capi_forecast(timestral_ctx* c,const timestral_forecast_request* a,int32_t* out_q) {
    if(out_q)*out_q=0;
    try {
        require(c,"null handle");require(a,"null request");require(out_q,"out_q is required");require(a->context,"context is required");
        require(a->B>0 && a->V>0 && a->T>0 && a->prediction_length>0,"invalid forecast dimensions");
        require(a->max_context>=0,"negative max_context");
        require(a->symmetric_averaging>=-1 && a->symmetric_averaging<=1 && a->clip_nonnegative>=-1 && a->clip_nonnegative<=1,"invalid forecast flag");
        pointer_count(a->past_cov,a->c_past);pointer_count(a->known_cov,a->c_known);
        pointer_count(a->cat_past,a->cc_past);pointer_count(a->cat_known,a->cc_known);
        size_t B=a->B,V=a->V,T=a->T,H=a->prediction_length;
        require(T<=SIZE_MAX-H,"forecast length overflow");size_t TH=T+H;
        using namespace ts::capi_detail;
        // All counts/bytes are checked before the first caller-buffer read.
        size_t nx=count<float>(B,V,T),np=count<float>(B,a->c_past,T),nk=count<float>(B,a->c_known,TH);
        size_t ncp=count<double>(B,a->cc_past,T),nck=count<double>(B,a->cc_known,TH);
        count<int8_t>(B,V,T);size_t Q=c->model.config().num_quantiles();require(Q<=INT32_MAX,"quantile count overflow");
        size_t no=mul(count<float>(B,V,H),Q);require(no<=std::vector<float>().max_size(),"output exceeds vector limit");mul(no,sizeof(float));
        validate_covariates(c->model.config(),a->c_past || a->c_known || a->cc_past || a->cc_known,
                            a->c_known || a->cc_known,size_t(a->max_context));
        ts::ForecastRequest r;r.batch=B;r.targets=V;r.length=T;r.horizon=H;r.max_context=a->max_context;
        r.past_count=a->c_past;r.known_count=a->c_known;r.categorical_past_count=a->cc_past;r.categorical_known_count=a->cc_known;
        if(a->observation_mask){for(size_t i=0;i<nx;++i)require(a->observation_mask[i]<=1,"observation mask must be boolean");r.observation_mask.assign(a->observation_mask,a->observation_mask+nx);}
        r.context.assign(a->context,a->context+nx);
        if(np)r.past.assign(a->past_cov,a->past_cov+np);if(nk)r.known.assign(a->known_cov,a->known_cov+nk);
        if(ncp)r.categorical_past.assign(a->cat_past,a->cat_past+ncp);if(nck)r.categorical_known.assign(a->cat_known,a->cat_known+nck);
        if(a->symmetric_averaging!=-1)r.symmetric_averaging=a->symmetric_averaging!=0;
        if(a->clip_nonnegative!=-1)r.clip_nonnegative=a->clip_nonnegative!=0;
        ThreadGuard guard(c->threads);auto f=c->forecaster.forecast(r);
        float* result=copy_output(f,B,V,H,Q);*out_q=int32_t(Q);c->error.clear();return result;
    }catch(...){caught(c);return nullptr;}
}
void timestral_capi_free_buffer(void* p){std::free(p);}
}
