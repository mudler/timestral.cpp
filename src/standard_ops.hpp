#pragma once
// Backend-neutral forecast graph built from standard ggml operations.
//
// The default build path uses ggml_custom_4d callbacks that reproduce the
// reference rounding bit for bit on CPU. The standard-op path uses only
// built-in ggml operations, so a GPU backend can schedule every node. It is
// held to the NORM gate (docs/PLAN.md, D.1), not to bit-exact gates.
#include "ggml.h"
namespace ts {
// True when new graphs use standard ggml operations. The default is false,
// or true when the library is built with TIMESTRAL_STANDARD_OPS=ON. The
// environment variable TIMESTRAL_STANDARD_OPS=0|1 overrides the default for
// the process. The flag is read when a graph is built.
bool standard_ops();
void set_standard_ops(bool enabled);

// y = w x (+ b) with ggml_mul_mat. w: [K, M], x: [K, N, R].
ggml_tensor* standard_linear(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, ggml_tensor* bias);
// ggml_rms_norm(x, eps) * scale.
ggml_tensor* standard_rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* scale, float eps);
// Scaled dot-product attention. q, k, v: [head_dim, heads, seq, batch];
// mask: additive F32 [seq, seq, batch] (0 or -inf). Returns [head_dim*heads, seq, batch].
ggml_tensor* standard_attention(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* mask);
}
