#include "model_loader.hpp"
#include "weight_policy.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>

namespace ts {

namespace {

constexpr int64_t kMaxRopePatches = int64_t(1) << 20;

[[noreturn]] void fail(const std::string& m) { throw Error(m); }

struct KV {
    const gguf_context* g;

    int64_t id(const char* key, gguf_type want) const {
        int64_t i = gguf_find_key(g, key);
        if (i < 0) fail(std::string("missing KV: ") + key);
        if (gguf_get_kv_type(g, i) != want)
            fail(std::string("KV ") + key + " has type " + gguf_type_name(gguf_get_kv_type(g, i)) +
                 ", expected " + gguf_type_name(want));
        return i;
    }
    std::string str(const char* k) const { return gguf_get_val_str(g, id(k, GGUF_TYPE_STRING)); }
    uint32_t u32(const char* k) const { return gguf_get_val_u32(g, id(k, GGUF_TYPE_UINT32)); }
    int32_t i32(const char* k) const { return gguf_get_val_i32(g, id(k, GGUF_TYPE_INT32)); }
    float f32(const char* k) const { return gguf_get_val_f32(g, id(k, GGUF_TYPE_FLOAT32)); }
    double f64(const char* k) const { return gguf_get_val_f64(g, id(k, GGUF_TYPE_FLOAT64)); }
    bool boolean(const char* k) const { return gguf_get_val_bool(g, id(k, GGUF_TYPE_BOOL)); }
    std::vector<std::string> str_arr(const char* k) const {
        int64_t i = id(k, GGUF_TYPE_ARRAY);
        if (gguf_get_arr_type(g, i) != GGUF_TYPE_STRING) fail(std::string("KV ") + k + " is not an array of strings");
        std::vector<std::string> v;
        for (size_t j = 0; j < gguf_get_arr_n(g, i); ++j) v.emplace_back(gguf_get_arr_str(g, i, j));
        return v;
    }
    std::vector<double> f64_arr(const char* k) const {
        int64_t i = id(k, GGUF_TYPE_ARRAY);
        if (gguf_get_arr_type(g, i) != GGUF_TYPE_FLOAT64) fail(std::string("KV ") + k + " is not an array of float64");
        size_t n = gguf_get_arr_n(g, i);
        const double* p = static_cast<const double*>(gguf_get_arr_data(g, i));
        return std::vector<double>(p, p + n);
    }
};

LayerKind parse_kind(const std::string& s) {
    if (s == "time") return LayerKind::Time;
    if (s == "variate") return LayerKind::Variate;
    if (s == "time_variate") return LayerKind::TimeVariate;
    if (s == "variate_time") return LayerKind::VariateTime;
    fail("unknown layer kind '" + s + "'");
}

Gate parse_gate(const std::string& s) {
    if (s == "none") return Gate::None;
    if (s == "sdpa_sigmoid") return Gate::SdpaSigmoid;
    fail("unknown attention gate '" + s + "'");
}

TensorSpec spec1(const std::string& n, int64_t a) {
    TensorSpec s; s.name = n; s.ne[0] = a; s.n_dims = 1; return s;
}
TensorSpec spec2(const std::string& n, int64_t a, int64_t b) {
    TensorSpec s; s.name = n; s.ne[0] = a; s.ne[1] = b; s.n_dims = 2; return s;
}

} // namespace

Config read_config(const gguf_context* g) {
    KV kv{g};
    Config c;
    {   // architecture first: refuse anything that is not a timestral file
        std::string arch = kv.str("general.architecture");
        if (arch != "timestral") fail("general.architecture is '" + arch + "', expected 'timestral'");
    }
    c.name = kv.str("general.name");
    c.license = kv.str("general.license");
    c.license_name = kv.str("general.license.name");
    c.license_link = kv.str("general.license.link");
    c.source_url = kv.str("general.source.url");
    c.hf_revision = kv.str("timestral.source.hf_revision");
    c.code_sha = kv.str("timestral.source.code_sha");
    c.weights_sha256 = kv.str("timestral.source.weights_sha256");
    c.converter_version = kv.u32("timestral.converter_version");
    if (c.converter_version != 1)
        fail("converter_version " + std::to_string(c.converter_version) + " is not supported (expected 1)");

    c.patch_size = kv.u32("timestral.patch_size");
    c.d_model = kv.u32("timestral.embedding_length");
    c.num_heads = kv.u32("timestral.attention.head_count");
    c.mlp_hidden_dim = kv.u32("timestral.feed_forward_length");
    c.n_layers = kv.u32("timestral.block_count");
    c.block_schedule_names = kv.str_arr("timestral.block_schedule");
    for (auto& s : c.block_schedule_names) c.block_schedule.push_back(parse_kind(s));
    c.quantile_knots = kv.f64_arr("timestral.quantile_knots");
    c.block_size = kv.u32("timestral.block_size");
    c.context_length = kv.u32("timestral.context_length");
    c.rope_theta = kv.f64("timestral.rope.freq_base");
    c.rope_partial_factor = kv.f64("timestral.rope.partial_factor");
    c.rope_dim = kv.u32("timestral.rope.dimension_count");
    c.rope_table_pos_min = kv.i32("timestral.rope.table_pos_min");
    c.norm_eps = kv.f32("timestral.attention.layer_norm_rms_epsilon");
    c.attn_bias = kv.boolean("timestral.attention.bias");
    c.mlp_bias = kv.boolean("timestral.mlp.bias");
    c.time_gate = parse_gate(kv.str("timestral.attention.time_gate"));
    c.variate_gate = parse_gate(kv.str("timestral.attention.variate_gate"));
    c.split_context_horizon = kv.boolean("timestral.attention.split_context_horizon");
    c.qkv_head_major = kv.boolean("timestral.attention.qkv_head_major");
    c.minimum_scale = kv.f64("timestral.scaler.minimum_scale");
    c.use_asinh = kv.boolean("timestral.scaler.use_asinh");
    c.stabilize_with_global = kv.boolean("timestral.scaler.stabilize_with_global");
    c.covariate_enabled = kv.boolean("timestral.covariate_enabled");
    c.attend_missing_patches = kv.boolean("timestral.attend_missing_patches");
    c.quantile_width_factor = kv.f64("timestral.quantile_width_factor");
    c.per_shot_horizon_patches = kv.u32("timestral.forecaster.per_shot_horizon_patches");
    c.quantile_real_cap_k = kv.f64("timestral.forecaster.quantile_real_cap_k");
    c.extreme_value_threshold = kv.f64("timestral.forecaster.extreme_value_threshold");
    c.symmetric_averaging = kv.boolean("timestral.forecaster.symmetric_averaging");
    c.clip_nonnegative = kv.boolean("timestral.forecaster.clip_nonnegative");
    c.target_encoding_smooth = kv.f64("timestral.target_encoding.smooth");

    if (!std::isfinite(c.target_encoding_smooth) || c.target_encoding_smooth <= 0)
        fail("target_encoding.smooth must be finite and positive");

    // The checks of TimestralConfig.__post_init__, plus the schema's own rules.
    if (!c.qkv_head_major) fail("qkv_head_major is false: the file has torch row order, which this loader does not read");
    if (c.stabilize_with_global) fail("stabilize_with_global=true is not supported");
    if (c.patch_size == 0 || c.d_model == 0 || c.num_heads == 0 || c.mlp_hidden_dim == 0 ||
        c.block_size == 0 || c.context_length == 0)
        fail("a dimension is zero");
    if (c.d_model % c.num_heads) fail("d_model is not divisible by num_heads");
    if (c.n_layers != c.block_schedule.size()) fail("block_count does not match the length of block_schedule");
    if (c.n_layers == 0) fail("block_schedule is empty");
    if (c.rope_dim == 0 || c.rope_dim % 2) fail("rope dimension_count must be positive and even");
    if (c.rope_partial_factor < 0.0 || c.rope_partial_factor > 1.0) fail("rope partial_factor out of range");
    if (c.rope_dim != (uint32_t)(int64_t)((double)c.head_dim() * c.rope_partial_factor))
        fail("rope.dimension_count does not equal int(head_dim * partial_factor)");
    if (c.rope_dim > c.head_dim()) fail("rope dimension_count exceeds head_dim");
    if (c.quantile_knots.empty()) fail("quantile_knots is empty");
    if (std::find(c.quantile_knots.begin(), c.quantile_knots.end(), 0.5) == c.quantile_knots.end())
        fail("quantile_knots must contain 0.5");
    for (size_t i = 1; i < c.quantile_knots.size(); ++i)
        if (!(c.quantile_knots[i - 1] < c.quantile_knots[i])) fail("quantile_knots must be increasing");
    if (c.context_length % c.patch_size) fail("context_length is not a multiple of patch_size");
    // Arithmetic in int64. The table holds 2 * patches + 1 positions, so bound the patch count:
    // 2^20 patches (about 33.5 million steps at patch_size 32) is far above the released models
    // (1056) and keeps the table length and its byte size far from any overflow.
    const int64_t patches = (int64_t)(c.context_length / c.patch_size) + (int64_t)c.block_size;
    if (patches > kMaxRopePatches)
        fail("context_length / patch_size + block_size = " + std::to_string(patches) +
             " exceeds the supported maximum of " + std::to_string(kMaxRopePatches) + " rope table positions per side");
    const int64_t want_min = -patches;
    if ((int64_t)c.rope_table_pos_min != want_min)
        fail("rope.table_pos_min " + std::to_string(c.rope_table_pos_min) + " does not equal " + std::to_string(want_min));
    if (!(c.norm_eps > 0.f)) fail("norm eps must be positive");
    return c;
}

size_t expected_tensor_count(const Config& c) {
    return expected_schema(c).size();
}

std::vector<TensorSpec> expected_schema(const Config& c) {
    const int64_t D = c.d_model, F = c.mlp_hidden_dim, ps = c.patch_size, Q = c.num_quantiles();
    const int64_t hd = c.head_dim(), P = c.rope_table_len();
    std::vector<TensorSpec> v;
    v.push_back(spec2("patch.hidden.weight", 2 * ps, 4 * D));
    v.push_back(spec1("patch.hidden.bias", 4 * D));
    v.push_back(spec2("patch.out.weight", 4 * D, D));
    v.push_back(spec1("patch.out.bias", D));
    v.push_back(spec2("patch.res.weight", 2 * ps, D));
    v.push_back(spec1("patch.res.bias", D));
    v.push_back(spec1("patch.forecast_query", D));
    for (uint32_t i = 0; i < c.n_layers; ++i) {
        const std::string b = "blk." + std::to_string(i) + ".";
        auto axes = layer_axes(c.block_schedule[i]);
        for (size_t j = 0; j < axes.size(); ++j) {
            const std::string a = b + "attn." + std::to_string(j) + ".";
            v.push_back(spec1(b + "attn_norm." + std::to_string(j) + ".weight", D));
            v.push_back(spec2(a + "qkv.weight", D, 3 * D));
            if (c.attn_bias) v.push_back(spec1(a + "qkv.bias", 3 * D));
            Gate gate = axes[j] ? c.time_gate : c.variate_gate;
            if (gate == Gate::SdpaSigmoid) {
                v.push_back(spec2(a + "gate.weight", D, D));
                if (c.attn_bias) v.push_back(spec1(a + "gate.bias", D));
            }
            v.push_back(spec2(a + "out.weight", D, D));
            if (c.attn_bias) v.push_back(spec1(a + "out.bias", D));
        }
        v.push_back(spec1(b + "ffn_norm.weight", D));
        v.push_back(spec2(b + "ffn_w12.weight", D, 2 * F));
        if (c.mlp_bias) v.push_back(spec1(b + "ffn_w12.bias", 2 * F));
        v.push_back(spec2(b + "ffn_w3.weight", F, D));
        if (c.mlp_bias) v.push_back(spec1(b + "ffn_w3.bias", D));
    }
    v.push_back(spec1("out_norm.weight", D));
    v.push_back(spec2("head.hidden.weight", D, 4 * D));
    v.push_back(spec2("head.out.weight", 4 * D, ps * Q));
    v.push_back(spec2("head.res.weight", D, ps * Q));
    v.push_back(spec1("rope.inv_freq", hd / 2));
    v.push_back(spec2("rope.cos", hd, P));
    v.push_back(spec2("rope.sin", hd, P));
    return v;
}

Model::Model(Model&& o) noexcept { *this = std::move(o); }

Model& Model::operator=(Model&& o) noexcept {
    if (this != &o) {
        reset();
        cfg_ = std::move(o.cfg_);
        gguf_ = o.gguf_; o.gguf_ = nullptr;
        ctx_ = o.ctx_; o.ctx_ = nullptr;
        file_ = std::move(o.file_);
        buf_ = o.buf_; o.buf_ = nullptr;
        tensors_ = std::move(o.tensors_);
        names_ = std::move(o.names_);
    }
    return *this;
}

void Model::reset() {
    if (buf_) { ggml_backend_buffer_free(buf_); buf_ = nullptr; }
    if (ctx_) { ggml_free(ctx_); ctx_ = nullptr; }
    if (gguf_) { gguf_free(gguf_); gguf_ = nullptr; }
    tensors_.clear();
    names_.clear();
}

Model::~Model() { reset(); }

ggml_tensor* Model::tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) fail("tensor not found: " + name);
    return it->second;
}

int64_t Model::rope_index(int64_t position) const {
    const int64_t len = tensor("rope.cos")->ne[1];
    const int64_t idx = position - (int64_t)cfg_.rope_table_pos_min;
    if (idx < 0 || idx >= len)
        fail("position " + std::to_string(position) + " is outside the rope table [" +
             std::to_string(cfg_.rope_table_pos_min) + ", " + std::to_string(cfg_.rope_table_pos_min + len - 1) + "]");
    return idx;
}

bool Model::in_mapping(const ggml_tensor* t) const {
    const unsigned char* p = static_cast<const unsigned char*>(t->data);
    return p && file_.data() && p >= file_.data() && p + ggml_nbytes(t) <= file_.data() + file_.size();
}

Model Model::load(const std::string& path) {
    Model m;
    try {
        gguf_init_params ip{ /*no_alloc=*/true, /*ctx=*/&m.ctx_ };
        m.gguf_ = gguf_init_from_file(path.c_str(), ip);
        if (!m.gguf_) fail("cannot read GGUF file: " + path);
        m.cfg_ = read_config(m.gguf_);
        const Config& c = m.cfg_;

        // Schema check: the file must hold exactly the tensors the config implies.
        std::vector<TensorSpec> schema = expected_schema(c);
        std::set<std::string> want;
        for (auto& s : schema) want.insert(s.name);
        const int64_t n_t = gguf_get_n_tensors(m.gguf_);
        std::set<std::string> have;
        for (int64_t i = 0; i < n_t; ++i) {
            std::string n = gguf_get_tensor_name(m.gguf_, i);
            if (!have.insert(n).second) fail("duplicate tensor in file: " + n);
            if (!want.count(n)) fail("unexpected tensor in file (not in the schema of this config): " + n);
        }
        for (auto& s : schema) {
            if (!have.count(s.name)) fail("missing tensor: " + s.name);
            ggml_tensor* t = ggml_get_tensor(m.ctx_, s.name.c_str());
            if (!t) fail("internal: tensor not in ggml context: " + s.name);
            if (!weight_type_allowed(s.name, t->type, t->ne[0]))
                fail("tensor " + s.name + " has type " + ggml_type_name(t->type) + ", forbidden by the storage policy");
            for (int d = 0; d < GGML_MAX_DIMS; ++d)
                if (t->ne[d] != s.ne[d])
                    fail("tensor " + s.name + " has shape [" + std::to_string(t->ne[0]) + ", " + std::to_string(t->ne[1]) +
                         ", " + std::to_string(t->ne[2]) + ", " + std::to_string(t->ne[3]) + "], expected [" +
                         std::to_string(s.ne[0]) + ", " + std::to_string(s.ne[1]) + ", " + std::to_string(s.ne[2]) +
                         ", " + std::to_string(s.ne[3]) + "]");
        }

        // Map the file and point every tensor into it (zero copy). Bounds are checked here
        // against the real file size, not only by gguf.
        m.file_ = MappedFile(path);
        m.buf_ = backend_buffer_from_mapping(m.file_);
        const size_t data_off = gguf_get_data_offset(m.gguf_);
        for (int64_t i = 0; i < n_t; ++i) {
            const char* name = gguf_get_tensor_name(m.gguf_, i);
            ggml_tensor* t = ggml_get_tensor(m.ctx_, name);
            size_t off = data_off + gguf_get_tensor_offset(m.gguf_, i);
            size_t nb = ggml_nbytes(t);
            if (off > m.file_.size() || nb > m.file_.size() - off)
                fail(std::string("tensor data out of file bounds: ") + name);
            if (off % gguf_get_alignment(m.gguf_)) fail(std::string("tensor data not aligned: ") + name);
            if (ggml_backend_tensor_alloc(m.buf_, t, const_cast<unsigned char*>(m.file_.data()) + off) != GGML_STATUS_SUCCESS)
                fail(std::string("cannot place tensor: ") + name);
            m.tensors_[name] = t;
            m.names_.push_back(name);
        }

        // rope.inv_freq: recompute with the reference formula and warn on a difference (P9).
        const ggml_tensor* inv = m.tensor("rope.inv_freq");
        const float* stored = static_cast<const float*>(inv->data);
        int diff = 0;
        for (uint32_t i = 0; i < c.head_dim() / 2; ++i) {
            float want_v = 0.f;
            if (i < c.rope_dim / 2) {
                float idx = (float)(2 * i);
                want_v = 1.0f / std::pow((float)c.rope_theta, idx / (float)c.rope_dim);
            }
            if (std::memcmp(&want_v, &stored[i], 4) != 0) ++diff;
        }
        if (diff) TS_LOG("warning: rope.inv_freq differs from the recomputed table in %d entries; the stored table is used", diff);
    } catch (...) {
        m.reset();
        throw;
    }
    return m;
}

} // namespace ts
