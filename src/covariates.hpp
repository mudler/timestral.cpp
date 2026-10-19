#pragma once
#include "config.hpp"
#include <cstddef>
namespace ts {
struct TargetEncoding {std::vector<float> past,future;};
TargetEncoding target_encode(const std::vector<double>& target,const std::vector<double>& past,
 const std::vector<double>& future={},double smooth=1.0);
bool supports_known_covariates(const Config&);
// Canonical input shapes [B,V,T], [B,C,T], [B,C,T+H]. Numeric rows precede
// categorical rows of the same role. Empty optional vectors require count zero.
struct AssembledRows {
 size_t rows_count=0,length=0,horizon=0;
 std::vector<float> rows,future;
 std::vector<int8_t> mask,future_mask;
 std::vector<int64_t> roles,groups;
};
AssembledRows assemble_rows(const Config&,const std::vector<float>& context,
 const std::vector<int8_t>& user_mask,size_t batch,size_t targets,size_t length,size_t prediction_length,
 const std::vector<float>& past,size_t past_count,const std::vector<float>& known,size_t known_count,
 size_t max_context=0,const std::vector<double>& cat_past={},size_t cat_past_count=0,
 const std::vector<double>& cat_known={},size_t cat_known_count=0);
}
