#pragma once
// Native C++17 API. Flat C ABI and CLI integration follow in M8.
// Model owns the mapped GGUF; it must outlive Forecaster. A Forecaster is not
// reentrant; independent instances may be used on independent calling threads.
#include "model_loader.hpp"
#include "forecaster.hpp"
