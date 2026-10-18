#pragma once
#include <cstdio>
#include <stdexcept>
#include <string>

#define TS_LOG(...) do { std::fprintf(stderr, "[timestral] " __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)

namespace ts {

// Error type thrown by the library for bad input and bad model files.
struct Error : std::runtime_error {
    explicit Error(const std::string& what) : std::runtime_error(what) {}
};

// Library version string, set at configure time.
const char* version();

} // namespace ts
