#pragma once
#include "encoder.hpp"
#include "head.hpp"
#include "scaler.hpp"
namespace ts {
struct BlockPrediction { std::vector<float> output, normalized, loc, scale; };
// Prepared packed inputs [rows,length]; native prediction builds all masks.
// The explicit context_only replay API remains available for component tests.
class PredictBlock {
public:
 explicit PredictBlock(const Model& m) : model_(m), encoder_(m), head_(m) {}
 BlockPrediction predict(const std::vector<float>& inputs,
  const std::vector<int8_t>& observed,size_t rows,size_t length,size_t horizon,
  const std::vector<int64_t>& groups,const std::vector<int64_t>& roles={},
  const std::vector<float>& known_future={},const std::vector<int8_t>& known_future_mask={});
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
