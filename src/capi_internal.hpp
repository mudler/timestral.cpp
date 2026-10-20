#pragma once
#include "forecaster.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
namespace ts::capi_detail {
inline size_t mul(size_t a,size_t b) {
    if(b && a>std::numeric_limits<size_t>::max()/b)throw Error("C ABI dimensions overflow");
    return a*b;
}
template<class T> inline size_t count(size_t a,size_t b,size_t c) {
    size_t n=mul(mul(a,b),c);
    if(n>std::vector<T>().max_size())throw Error("C ABI buffer exceeds vector limit");
    mul(n,sizeof(T));return n;
}
inline void validate_covariates(const Config& c,bool any,bool known,size_t max_context) {
    if(any && !c.covariate_enabled)throw Error("this checkpoint was not trained with covariates");
    if(known && !supports_known_covariates(c))throw Error("this checkpoint cannot condition on known-future covariates");
    if(max_context>size_t(c.context_length))throw Error("max_context exceeds stored position range");
}
// Private finite-result seam: no extra C ABI symbol or model mutation.
inline float* copy_output(const Forecast& f,size_t B,size_t V,size_t H,size_t Q) {
    const size_t n=mul(count<float>(B,V,H),Q),bytes=mul(n,sizeof(float));
    if(f.batch!=B || f.targets!=V || f.horizon!=H || f.quantile_levels.size()!=Q || f.quantiles.size()!=n)
        throw Error("C ABI forecast output shape mismatch");
    for(float x:f.quantiles)if(!std::isfinite(x))throw Error("C ABI nonfinite forecast output");
    auto p=static_cast<float*>(std::malloc(bytes));
    if(!p)throw std::bad_alloc();
    std::memcpy(p,f.quantiles.data(),bytes);return p;
}
}
