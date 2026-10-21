#pragma once
// Native C++17 API. The flat C ABI is in timestral_capi.h.
// Model owns the mapped GGUF; it must outlive Forecaster. A Forecaster is not
// reentrant; independent instances may be used on independent calling threads.
#include "model_loader.hpp"
#include "forecaster.hpp"
