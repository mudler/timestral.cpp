#pragma once
#include "encoder.hpp"
#include "head.hpp"
#include "scaler.hpp"
namespace ts {
struct BlockPrediction { std::vector<float> output, normalized, loc, scale; };
// Context-only M4 path. Prepared inputs [rows,length]; caller supplies M5 masks
// and positions for [context patches; horizon patches]. No covariate semantics.
class PredictBlock {
public:
 explicit PredictBlock(const Model& m) : model_(m), encoder_(m), head_(m) {}
 BlockPrediction context_only(const std::vector<float>& inputs,
  const std::vector<int8_t>& observed, size_t rows, size_t length, size_t horizon,
  const TimeMask&, const std::vector<int64_t>& positions,
  const std::vector<int8_t>& variate_mask = {});
private:
 const Model& model_;
 Encoder encoder_;
 Head head_;
};
}
