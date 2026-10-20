#pragma once
#include "covariates.hpp"
#include "predict_block.hpp"
#include <optional>
#include <functional>
namespace ts {
// Canonical packed input shapes are (B,V,T), (B,C,T), (B,C,T+H).
// Categorical codes stay double until target encoding; NaN denotes missing.
struct ForecastRequest {
 size_t batch=0, targets=0, length=0, horizon=0;
 std::vector<float> context, past, known;
 std::vector<int8_t> observation_mask;
 std::vector<double> categorical_past, categorical_known;
 size_t past_count=0, known_count=0, categorical_past_count=0, categorical_known_count=0;
 size_t max_context=0, per_shot_horizon_patches=0;
 std::optional<bool> symmetric_averaging, clip_nonnegative;
 std::optional<double> quantile_real_cap_k;
};
struct Forecast {
 size_t batch=0, targets=0, horizon=0;
 std::vector<double> quantile_levels;
 std::vector<float> quantiles; // (B,V,H,Q)
};
struct DecodeBlock {
 size_t offset=0, patches=0, length=0;
 std::vector<float> context, future, predicted, refined;
 std::vector<int8_t> mask, future_mask;
};
using DecodeProgress = std::function<void(size_t offset, size_t patches, size_t context_length)>;
struct ForecastAnchors {
 size_t batch=0,targets=0;
 std::vector<float> loc,scale; // owned [B,V], first positive native prediction
};
struct ForecastTrace {
 std::vector<DecodeBlock> positive, negative;
 // Optional observer called immediately before each heavy prediction; no tensor ownership.
 DecodeProgress before_block;
 bool capture_initial_anchors=false;
 bool capture_blocks=true;
 std::optional<ForecastAnchors> initial_anchors;
};
class Forecaster {
public:
 explicit Forecaster(const Model& model): model_(model), predictor_(model) {}
 Forecast forecast(const ForecastRequest&, ForecastTrace* trace=nullptr);
 // Packed native decode entry point; also used to audit free-running contexts.
 std::vector<float> decode(const AssembledRows&,size_t horizon,size_t max_context=0,
                          size_t per_shot=0,std::vector<DecodeBlock>* trace=nullptr,
                          const DecodeProgress& progress={});
private:
 std::vector<float> decode_impl(const AssembledRows&,size_t,size_t,size_t,
     std::vector<DecodeBlock>*,const DecodeProgress&,ForecastAnchors*);
 const Model& model_;
 PredictBlock predictor_;
 std::vector<float> raw(const AssembledRows&,const ForecastRequest&,std::vector<DecodeBlock>*,const DecodeProgress&,ForecastAnchors* = nullptr);
};
}
