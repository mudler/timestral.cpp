#include "common.hpp"
#ifdef TS_STANDALONE_MKL
#include "standalone_mkl.hpp"
#endif

#ifndef TIMESTRAL_VERSION
#define TIMESTRAL_VERSION "unknown"
#endif

namespace ts {

const char* version() { return TIMESTRAL_VERSION; }

} // namespace ts

namespace ts {const char* backend_identity(){
#ifdef TS_STANDALONE_MKL
static const std::string identity=standalone_mkl_identity();return identity.c_str();
#else
return "native-cpu";
#endif
}}
