#include "common.hpp"

#ifndef TIMESTRAL_VERSION
#define TIMESTRAL_VERSION "unknown"
#endif

namespace ts {

const char* version() { return TIMESTRAL_VERSION; }

} // namespace ts
