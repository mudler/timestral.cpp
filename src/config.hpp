#pragma once
// ts::Config: every KV of the GGUF schema (docs/PLAN.md, section B.1), read by the loader.
// Nothing in this struct has a default: a missing key is a load error.
#include <cstdint>
#include <string>
#include <vector>

namespace ts {

enum class LayerKind { Time, Variate, TimeVariate, VariateTime };
enum class Gate { None, SdpaSigmoid };

// Ordered attention axes of a layer kind; true = time axis (reference: LAYER_KIND_AXES).
inline std::vector<bool> layer_axes(LayerKind k) {
    switch (k) {
        case LayerKind::Time:        return {true};
        case LayerKind::Variate:     return {false};
        case LayerKind::TimeVariate: return {true, false};
        case LayerKind::VariateTime: return {false, true};
    }
    return {};
}

struct Config {
    // general.* and traceability
    std::string name, license, license_name, license_link, source_url;
    std::string hf_revision, code_sha, weights_sha256;
    uint32_t converter_version = 0;

    // architecture
    uint32_t patch_size = 0, d_model = 0, num_heads = 0, mlp_hidden_dim = 0, n_layers = 0;
    std::vector<std::string> block_schedule_names;
    std::vector<LayerKind> block_schedule;
    std::vector<double> quantile_knots;
    uint32_t block_size = 0, context_length = 0;

    // rope
    double rope_theta = 0, rope_partial_factor = 0;
    uint32_t rope_dim = 0;
    int32_t rope_table_pos_min = 0;

    // attention and mlp
    float norm_eps = 0;
    bool attn_bias = false, mlp_bias = false;
    Gate time_gate = Gate::None, variate_gate = Gate::None;
    bool split_context_horizon = false, qkv_head_major = false;

    // scaler and covariates
    double minimum_scale = 0;
    bool use_asinh = false, stabilize_with_global = false;
    bool covariate_enabled = false, attend_missing_patches = false;
    double quantile_width_factor = 0;

    // forecaster defaults (overridable at run time)
    uint32_t per_shot_horizon_patches = 0;
    double quantile_real_cap_k = 0, extreme_value_threshold = 0;
    bool symmetric_averaging = false, clip_nonnegative = false;
    double target_encoding_smooth = 0;

    // derived
    uint32_t head_dim() const { return d_model / num_heads; }
    uint32_t num_quantiles() const { return (uint32_t)quantile_knots.size(); }
    uint32_t median_index() const {
        for (size_t i = 0; i < quantile_knots.size(); ++i) if (quantile_knots[i] == 0.5) return (uint32_t)i;
        return 0;
    }
    // Positions stored in rope.cos / rope.sin: table_pos_min .. -table_pos_min. All in int64:
    // the loader bounds table_pos_min, so this cannot overflow for a config it accepted.
    int64_t rope_table_len() const { return 2 * (-(int64_t)rope_table_pos_min) + 1; }
};

} // namespace ts
