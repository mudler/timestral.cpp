#pragma once
#include "model_loader.hpp"
namespace ts {
struct ResidualOutput { ggml_tensor* hidden; ggml_tensor* output; };
ResidualOutput residual_block(ggml_context* ctx, const Model& model, ggml_tensor* x);
// x: [head_dim, heads, sequence, rows]; tables: [head_dim, 1, sequence, rows].
ggml_tensor* apply_rope(ggml_context* ctx, ggml_tensor* x, ggml_tensor* cos, ggml_tensor* sin);
}
