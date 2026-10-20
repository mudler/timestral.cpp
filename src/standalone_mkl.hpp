#pragma once
#include "ggml.h"
#include <string>
namespace ts {
void standalone_mkl_init();
std::string standalone_mkl_identity();
ggml_tensor* standalone_patch(ggml_context*,ggml_tensor* weight,ggml_tensor* input,ggml_tensor* bias);
ggml_tensor* standalone_sdpa(ggml_context*,ggml_tensor*,ggml_tensor*,ggml_tensor*,ggml_tensor*,size_t split);
}
