#pragma once
#include "encoder.hpp"
namespace ts {
struct Masks {
 TimeMask time;
 std::vector<int8_t> obs,key_valid,variate,active;
 std::vector<int64_t> first_group,inverse,positions,anchor;
};
// observed covers the context in the context-only path, the full window otherwise.
Masks build_masks(const Config&,const std::vector<int8_t>& observed,size_t rows,
 size_t length,size_t horizon,const std::vector<int64_t>& groups,
 const std::vector<int64_t>& roles = {});
}
