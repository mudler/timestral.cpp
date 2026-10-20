#pragma once
#include "model_loader.hpp"
namespace ts {
struct ResidualOutput { ggml_tensor* hidden; ggml_tensor* output; };
ResidualOutput residual_block(ggml_context* ctx, const Model& model, ggml_tensor* x);
// x: [head_dim, heads, sequence, rows]; tables: [head_dim, 1, sequence, rows].
ggml_tensor* apply_rope(ggml_context* ctx, ggml_tensor* x, ggml_tensor* cos, ggml_tensor* sin);
}

namespace ts {
ggml_tensor* affine(ggml_context*,const Model&,const std::string&,ggml_tensor*,bool);
ggml_tensor* rms_norm(ggml_context*,ggml_tensor*,ggml_tensor*,float);
ggml_tensor* attention_time(ggml_context*,ggml_tensor*,ggml_tensor*,ggml_tensor*,ggml_tensor*,size_t split=0);
// Same SDPA kernel, with sequence=rows, batch=patches; no RoPE or causal default.
ggml_tensor* attention_variate(ggml_context*,ggml_tensor*,ggml_tensor*,ggml_tensor*,ggml_tensor*);
ggml_tensor* swiglu(ggml_context*,const Model&,const std::string&,ggml_tensor*);
}

namespace ts {
ggml_tensor* sigmoid(ggml_context*,ggml_tensor*);
}

namespace ts::detail {
// Internal custom-op callback; exposed for exact scalar-oracle tests.
void linear_f32(ggml_tensor*,int,int,void*);
void sdpa_f32(ggml_tensor*,int,int,void*);
}

namespace ts::detail {
// Internal deterministic fallback/coverage seam, not part of the public ABI.
struct KeyPackProbe {
    size_t budget=16*1024*1024;
    bool fail_allocation=false;
    bool force_overflow=false;
    size_t packed_heads=0,allocation_fallbacks=0,size_fallbacks=0;
};
bool keypack_bytes(size_t heads,size_t sequence,size_t dim,size_t budget,size_t& bytes);
void sdpa_f32_probe(ggml_tensor*,int,int,void*,KeyPackProbe&);
}
