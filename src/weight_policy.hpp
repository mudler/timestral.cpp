#pragma once
#include "ggml.h"
#include "common.hpp"
#include <regex>
#include <string>
namespace ts {
// Exact B.4 storage allowlist. Shape/schema validation remains the loader's job.
inline bool reduced_weight_name(const std::string& name) {
    static const std::regex block(R"(^blk\.[0-9]+\.(attn\.[0-9]+\.(qkv|gate|out)|ffn_w12|ffn_w3)\.weight$)");
    return name=="patch.out.weight" || name=="head.hidden.weight" ||
           name=="head.out.weight" || name=="head.res.weight" || std::regex_match(name,block);
}
inline bool reduced_weight_type(ggml_type type) {
    return type==GGML_TYPE_F16 || type==GGML_TYPE_Q8_0 || type==GGML_TYPE_Q6_K ||
           type==GGML_TYPE_Q5_K || type==GGML_TYPE_Q4_K;
}
inline bool weight_type_allowed(const std::string& name,ggml_type type,int64_t ne0) {
    if(ne0<=0)return false;
    if(type==GGML_TYPE_F32)return true;
    return reduced_weight_type(type) &&
           reduced_weight_name(name) && ne0%ggml_blck_size(type)==0;
}
inline ggml_tensor* reduced_projection(ggml_context* ctx,ggml_tensor* w,ggml_tensor* x,ggml_tensor* b) {
    if(!reduced_weight_type(w->type) || !weight_type_allowed(w->name,w->type,w->ne[0]) ||
       x->type!=GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[3]!=1 ||
       w->ne[0]!=x->ne[0] || (b && (b->type!=GGML_TYPE_F32 || ggml_nelements(b)!=w->ne[1])))
        throw Error("invalid reduced projection");
    auto y=ggml_mul_mat(ctx,w,x);
    return b?ggml_add(ctx,y,b):y;
}
}
