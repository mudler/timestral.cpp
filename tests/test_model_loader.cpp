// Tests of the GGUF loader (docs/PLAN.md, M0b).
//
//   test_model_loader          the two tiny fixtures (always runs)
//   test_model_loader --real   the 209M GGUF from TIMESTRAL_TEST_GGUF (77 when unset)
//
// For each file: load, print the config and tensor names, check the tensor count against the
// formula of B.2, compare the loader's config with the config.json that was converted (gate
// G0.2, EXACT), and check that the weights are used in place from the file mapping. For the
// tiny files the test also rewrites the GGUF with one defect at a time and expects the loader
// to refuse it: every KV removed in turn, wrongly typed KVs, inconsistent KVs, and missing,
// extra, wrongly typed and wrongly shaped tensors.
#include "json_min.hpp"
#include "model_loader.hpp"
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>

using namespace ts;

static int g_fail = 0, g_checks = 0;
#define EXPECT(cond, what) do { ++g_checks; if (!(cond)) { ++g_fail; \
    std::fprintf(stderr, "TEST FAILED line %d: %s\n", __LINE__, what); } } while (0)

static std::string tmp_path(const std::string& name) { return std::string(TS_TEST_TMP_DIR) + "/" + name; }

static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

static bool bits_equal(double a, double b) { return std::memcmp(&a, &b, 8) == 0; }

// ---------------------------------------------------------------------------
// G0.2: the loader's Config equals config.json, EXACT
// ---------------------------------------------------------------------------

static void check_g02(const Config& c, const std::string& json_path, const std::string& label) {
    std::string text = read_file(json_path);
    EXPECT(!text.empty(), "config.json copy is readable");
    if (text.empty()) return;
    std::map<std::string, jsonmin::Value> j;
    try { j = jsonmin::Parser(text).object(); }
    catch (const std::exception& e) { EXPECT(false, e.what()); return; }

    const char* keys[] = {"patch_size", "d_model", "num_heads", "mlp_hidden_dim", "block_schedule", "quantile_knots",
        "block_size", "context_length", "rope_theta", "rope_partial_factor", "norm_eps", "attn_bias", "mlp_bias",
        "time_attn_gate", "variate_attn_gate", "minimum_scale", "use_asinh", "stabilize_with_global",
        "covariate_enabled", "split_context_horizon_attn", "attend_missing_patches", "quantile_width_factor"};
    for (const char* k : keys) EXPECT(j.count(k) == 1, (std::string("config.json has key ") + k).c_str());
    EXPECT(j.size() == sizeof(keys) / sizeof(keys[0]), "config.json has no keys the test does not know");
    if (g_fail) return;

    int bad = 0;
    auto eq = [&](bool ok, const char* what) { ++g_checks; if (!ok) { ++g_fail; ++bad; std::fprintf(stderr, "[G0.2 %s] MISMATCH %s\n", label.c_str(), what); } };
    auto I = [&](const char* k) { return (long long)j[k].integer(); };
    auto D = [&](const char* k) { return j[k].num(); };
    auto Bo = [&](const char* k) { return j[k].b; };
    auto G = [&](const char* k) { return j[k].text == "sdpa_sigmoid" ? Gate::SdpaSigmoid : Gate::None; };
    eq(I("patch_size") == c.patch_size, "patch_size");
    eq(I("d_model") == c.d_model, "d_model");
    eq(I("num_heads") == c.num_heads, "num_heads");
    eq(I("mlp_hidden_dim") == c.mlp_hidden_dim, "mlp_hidden_dim");
    eq(I("block_size") == c.block_size, "block_size");
    eq(I("context_length") == c.context_length, "context_length");
    eq(bits_equal(D("rope_theta"), c.rope_theta), "rope_theta");
    eq(bits_equal(D("rope_partial_factor"), c.rope_partial_factor), "rope_partial_factor");
    float eps = (float)D("norm_eps");
    eq(std::memcmp(&eps, &c.norm_eps, 4) == 0, "norm_eps (as float32)");
    eq(Bo("attn_bias") == c.attn_bias, "attn_bias");
    eq(Bo("mlp_bias") == c.mlp_bias, "mlp_bias");
    eq(G("time_attn_gate") == c.time_gate, "time_attn_gate");
    eq(G("variate_attn_gate") == c.variate_gate, "variate_attn_gate");
    eq(j["time_attn_gate"].text == "sdpa_sigmoid" || j["time_attn_gate"].text == "none", "time gate value is known");
    eq(bits_equal(D("minimum_scale"), c.minimum_scale), "minimum_scale");
    eq(Bo("use_asinh") == c.use_asinh, "use_asinh");
    eq(Bo("stabilize_with_global") == c.stabilize_with_global, "stabilize_with_global");
    eq(Bo("covariate_enabled") == c.covariate_enabled, "covariate_enabled");
    eq(Bo("split_context_horizon_attn") == c.split_context_horizon, "split_context_horizon_attn");
    eq(Bo("attend_missing_patches") == c.attend_missing_patches, "attend_missing_patches");
    eq(bits_equal(D("quantile_width_factor"), c.quantile_width_factor), "quantile_width_factor");
    const auto& sched = j["block_schedule"].arr;
    eq(sched.size() == c.block_schedule_names.size() && sched.size() == c.n_layers, "block_schedule length");
    for (size_t i = 0; i < sched.size() && i < c.block_schedule_names.size(); ++i)
        eq(sched[i].text == c.block_schedule_names[i], "block_schedule entry");
    const auto& kn = j["quantile_knots"].arr;
    eq(kn.size() == c.quantile_knots.size(), "quantile_knots length");
    for (size_t i = 0; i < kn.size() && i < c.quantile_knots.size(); ++i)
        eq(bits_equal(kn[i].num(), c.quantile_knots[i]), "quantile_knots entry");
    // derived values the loader checked from KV must equal the same derivation from the JSON
    eq((long long)c.rope_dim == (long long)((double)(I("d_model") / I("num_heads")) * D("rope_partial_factor")), "rope_dim derived");
    eq(c.rope_table_pos_min == -(int)(I("context_length") / I("patch_size") + I("block_size")), "table_pos_min derived");
    std::printf("G0.2 [%s]: %zu fields compared EXACT -> %s\n", label.c_str(), sizeof(keys) / sizeof(keys[0]) + 2, bad ? "FAIL" : "OK");
}

// ---------------------------------------------------------------------------
// Positive checks on one file
// ---------------------------------------------------------------------------

static void print_config(const Config& c) {
    std::printf("config: name=%s patch_size=%u d_model=%u heads=%u head_dim=%u mlp=%u layers=%u quantiles=%u median=%u\n",
                c.name.c_str(), c.patch_size, c.d_model, c.num_heads, c.head_dim(), c.mlp_hidden_dim, c.n_layers,
                c.num_quantiles(), c.median_index());
    std::printf("  block_size=%u context_length=%u rope: theta=%g partial=%g dim=%u pos_min=%d table_len=%lld\n",
                c.block_size, c.context_length, c.rope_theta, c.rope_partial_factor, c.rope_dim,
                c.rope_table_pos_min, (long long)c.rope_table_len());
    std::printf("  attn_bias=%d mlp_bias=%d norm_eps=%g width_factor=%g min_scale=%g schedule:", c.attn_bias, c.mlp_bias,
                (double)c.norm_eps, c.quantile_width_factor, c.minimum_scale);
    for (auto& s : c.block_schedule_names) std::printf(" %s", s.c_str());
    std::printf("\n  hf_revision=%s code_sha=%s weights_sha256=%s\n", c.hf_revision.c_str(), c.code_sha.c_str(),
                c.weights_sha256.c_str());
}

// The formula of B.2, taken literally: 11 + sum over layers of (3 + 4 * n_attn) + 3, plus
// 3 * n_attn per layer with attn_bias and 2 per layer with mlp_bias. It assumes both gates on.
static size_t plan_count_formula(const Config& c) {
    size_t n = 11 + 3;
    for (auto k : c.block_schedule) {
        size_t na = layer_axes(k).size();
        n += 3 + 4 * na;
        if (c.attn_bias) n += 3 * na;
        if (c.mlp_bias) n += 2;
    }
    return n;
}

static void check_file(const std::string& path, const std::string& label, bool compare_bytes, bool print_names) {
    std::printf("== %s: %s\n", label.c_str(), path.c_str());
    Model m = Model::load(path);
    const Config& c = m.config();
    print_config(c);
    std::printf("tensors: %zu\n", m.tensor_names().size());
    if (print_names) for (auto& n : m.tensor_names()) std::printf("  %s\n", n.c_str());

    EXPECT(c.time_gate == Gate::SdpaSigmoid && c.variate_gate == Gate::SdpaSigmoid, "both gates on (the plan's formula assumes it)");
    EXPECT(m.tensor_names().size() == plan_count_formula(c), "tensor count equals the B.2 formula");
    EXPECT(m.tensor_names().size() == expected_tensor_count(c), "tensor count equals the schema built from the config");
    for (auto& n : m.tensor_names()) {
        ggml_tensor* t = m.tensor(n);
        EXPECT(t->type == GGML_TYPE_F32, "all tensors are F32");
        EXPECT(m.in_mapping(t), "weights are used in place from the file mapping");
    }
    {   // run-time position lookup is checked against the table's own length
        const int64_t len = m.tensor("rope.cos")->ne[1];
        const int64_t lo = c.rope_table_pos_min, hi = lo + len - 1;
        EXPECT(len == c.rope_table_len() && hi == -lo, "rope table is symmetric and as long as the config says");
        EXPECT(m.rope_index(lo) == 0 && m.rope_index(hi) == len - 1 && m.rope_index(0) == -lo, "rope_index at the ends and at 0");
        bool a = false, b = false;
        try { m.rope_index(lo - 1); } catch (const Error&) { a = true; }
        try { m.rope_index(hi + 1); } catch (const Error&) { b = true; }
        EXPECT(a && b, "rope_index refuses positions outside the table");
    }
    std::string json = path.substr(0, path.size() - 5) + ".config.json";   // strip ".gguf"
    check_g02(c, json, label);

    if (compare_bytes) {   // independent read through gguf with copied data
        ggml_context* ctx = nullptr;
        gguf_init_params ip{ false, &ctx };
        gguf_context* g = gguf_init_from_file(path.c_str(), ip);
        EXPECT(g != nullptr, "independent gguf read");
        if (g) {
            for (auto& n : m.tensor_names()) {
                ggml_tensor* a = m.tensor(n);
                ggml_tensor* b = ggml_get_tensor(ctx, n.c_str());
                EXPECT(b && ggml_nbytes(a) == ggml_nbytes(b) && std::memcmp(a->data, b->data, ggml_nbytes(a)) == 0,
                       "loader tensor bytes equal an independent read");
            }
            gguf_free(g); ggml_free(ctx);
        }
    }
}

// ---------------------------------------------------------------------------
// Negative tests: rewrite the GGUF with one defect, expect a refusal
// ---------------------------------------------------------------------------

struct Mut {
    gguf_context* g;                       // new context (KV copied from the source)
    std::vector<ggml_tensor*>* tensors;    // tensors to write, in order; edit freely
    ggml_context* ctx;                     // scratch context for new tensors
};

static ggml_tensor* clone_with_ne(ggml_context* ctx, const ggml_tensor* src, ggml_type type,
                                  int64_t n0, int64_t n1, int64_t n2, int64_t n3) {
    ggml_tensor* t = ggml_new_tensor_4d(ctx, type, n0, n1, n2, n3);
    ggml_set_name(t, src->name);
    return t;
}

static std::string make_variant(const std::string& src, const std::string& name, const std::function<void(Mut&)>& fn) {
    ggml_context* ctx = nullptr;
    gguf_init_params ip{ false, &ctx };
    gguf_context* g = gguf_init_from_file(src.c_str(), ip);
    if (!g) { std::fprintf(stderr, "cannot read %s\n", src.c_str()); std::exit(1); }
    gguf_context* g2 = gguf_init_empty();
    gguf_set_kv(g2, g);
    std::vector<ggml_tensor*> ts;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) ts.push_back(ggml_get_tensor(ctx, gguf_get_tensor_name(g, i)));
    ggml_init_params sp{ 8u << 20, nullptr, false };
    ggml_context* scratch = ggml_init(sp);
    Mut mu{g2, &ts, scratch};
    fn(mu);
    for (ggml_tensor* t : ts) gguf_add_tensor(g2, t);
    std::string out = tmp_path(name);
    if (!gguf_write_to_file(g2, out.c_str(), false)) { std::fprintf(stderr, "write failed %s\n", out.c_str()); std::exit(1); }
    gguf_free(g2); ggml_free(scratch); gguf_free(g); ggml_free(ctx);
    return out;
}

static ggml_tensor*& find(Mut& m, const char* name) {
    for (auto& t : *m.tensors) if (std::strcmp(t->name, name) == 0) return t;
    std::fprintf(stderr, "test bug: no tensor %s\n", name); std::exit(1);
}

// The variant must load when `expect_ok`, otherwise it must be refused with a message that
// contains `needle` (so the test fails for the reason it claims).
static void variant(const std::string& src, const std::string& label, const std::function<void(Mut&)>& fn,
                    bool expect_ok, const char* needle = "") {
    std::string p = make_variant(src, "variant.gguf", fn);
    try {
        Model m = Model::load(p);
        EXPECT(expect_ok, ("defect accepted: " + label).c_str());
        if (!expect_ok) std::fprintf(stderr, "  [%s] was loaded but must be refused\n", label.c_str());
    } catch (const Error& e) {
        std::string msg = e.what();
        bool ok = !expect_ok && msg.find(needle) != std::string::npos;
        EXPECT(ok, ("wrong outcome: " + label + ": " + msg).c_str());
        if (ok) std::printf("  refused [%s]: %s\n", label.c_str(), msg.c_str());
    }
}

static void negative_tests(const std::string& src, const std::string& tag) {
    std::printf("== negative tests on %s\n", tag.c_str());
    variant(src, "rewrite without change", [](Mut&) {}, true);

    // every KV removed in turn
    {
        ggml_context* ctx = nullptr;
        gguf_init_params ip{ true, &ctx };
        gguf_context* g = gguf_init_from_file(src.c_str(), ip);
        std::vector<std::string> keys;
        for (int64_t i = 0; i < gguf_get_n_kv(g); ++i) keys.push_back(gguf_get_key(g, i));
        gguf_free(g); if (ctx) ggml_free(ctx);
        EXPECT(keys.size() >= 40, "the file has the KV of B.1");
        int refused = 0;
        for (auto& k : keys) {
            std::string p = make_variant(src, "variant.gguf", [&](Mut& m) { gguf_remove_key(m.g, k.c_str()); });
            try { Model m = Model::load(p); EXPECT(false, ("KV removed but file loads: " + k).c_str()); }
            catch (const Error& e) {
                bool ok = std::string(e.what()).find("missing KV: " + k) != std::string::npos;
                EXPECT(ok, ("removed KV gives another error: " + k + ": " + e.what()).c_str());
                refused += ok;
            }
        }
        std::printf("  %d of %zu removed KV refused with 'missing KV'\n", refused, keys.size());
    }

    // KV of the wrong type
    variant(src, "patch_size as float32", [](Mut& m) { gguf_remove_key(m.g, "timestral.patch_size"); gguf_set_val_f32(m.g, "timestral.patch_size", 8.f); }, false, "has type");
    variant(src, "embedding_length as string", [](Mut& m) { gguf_remove_key(m.g, "timestral.embedding_length"); gguf_set_val_str(m.g, "timestral.embedding_length", "32"); }, false, "has type");
    variant(src, "bias flag as u32", [](Mut& m) { gguf_remove_key(m.g, "timestral.attention.bias"); gguf_set_val_u32(m.g, "timestral.attention.bias", 1); }, false, "has type");
    variant(src, "knots as float32", [](Mut& m) { gguf_remove_key(m.g, "timestral.quantile_knots"); float v[3] = {0.1f, 0.5f, 0.9f}; gguf_set_arr_data(m.g, "timestral.quantile_knots", GGUF_TYPE_FLOAT32, v, 3); }, false, "not an array of float64");
    variant(src, "schedule as u32 array", [](Mut& m) { gguf_remove_key(m.g, "timestral.block_schedule"); uint32_t v[4] = {0, 1, 2, 3}; gguf_set_arr_data(m.g, "timestral.block_schedule", GGUF_TYPE_UINT32, v, 4); }, false, "not an array of strings");
    variant(src, "rope_theta as float32", [](Mut& m) { gguf_remove_key(m.g, "timestral.rope.freq_base"); gguf_set_val_f32(m.g, "timestral.rope.freq_base", 10000.f); }, false, "has type");

    variant(src, "nondefault smoothing", [](Mut& m) { gguf_set_val_f64(m.g, "timestral.target_encoding.smooth", 3.); }, true);
    for(double smooth:{0.,-1.,double(NAN),double(INFINITY)})
        variant(src, "invalid smoothing", [smooth](Mut& m) { gguf_set_val_f64(m.g, "timestral.target_encoding.smooth", smooth); }, false, "target_encoding.smooth");

    // inconsistent KV
    variant(src, "architecture", [](Mut& m) { gguf_set_val_str(m.g, "general.architecture", "llama"); }, false, "expected 'timestral'");
    variant(src, "qkv not head-major", [](Mut& m) { gguf_set_val_bool(m.g, "timestral.attention.qkv_head_major", false); }, false, "qkv_head_major");
    variant(src, "stabilize_with_global", [](Mut& m) { gguf_set_val_bool(m.g, "timestral.scaler.stabilize_with_global", true); }, false, "stabilize_with_global");
    variant(src, "converter version", [](Mut& m) { gguf_set_val_u32(m.g, "timestral.converter_version", 2); }, false, "converter_version");
    variant(src, "unknown gate", [](Mut& m) { gguf_set_val_str(m.g, "timestral.attention.time_gate", "relu"); }, false, "unknown attention gate");
    variant(src, "unknown layer kind", [](Mut& m) { const char* v[4] = {"time", "variate", "time_variate", "oops"}; gguf_remove_key(m.g, "timestral.block_schedule"); gguf_set_arr_str(m.g, "timestral.block_schedule", v, 4); }, false, "unknown layer kind");
    variant(src, "block_count", [](Mut& m) { gguf_set_val_u32(m.g, "timestral.block_count", 3); }, false, "block_count");
    variant(src, "rope dim", [](Mut& m) { gguf_set_val_u32(m.g, "timestral.rope.dimension_count", 2); }, false, "rope.dimension_count");
    variant(src, "table_pos_min", [](Mut& m) { gguf_set_val_i32(m.g, "timestral.rope.table_pos_min", -10); }, false, "table_pos_min");
    variant(src, "knots without 0.5", [](Mut& m) { gguf_remove_key(m.g, "timestral.quantile_knots"); double v[3] = {0.1, 0.4, 0.9}; gguf_set_arr_data(m.g, "timestral.quantile_knots", GGUF_TYPE_FLOAT64, v, 3); }, false, "0.5");
    variant(src, "knots not increasing", [](Mut& m) { gguf_remove_key(m.g, "timestral.quantile_knots"); double v[3] = {0.9, 0.5, 0.1}; gguf_set_arr_data(m.g, "timestral.quantile_knots", GGUF_TYPE_FLOAT64, v, 3); }, false, "increasing");
    variant(src, "head count does not divide d_model", [](Mut& m) { gguf_set_val_u32(m.g, "timestral.attention.head_count", 5); }, false, "divisible");
    // a KV that changes the schema while the tensors stay: the shapes no longer match
    variant(src, "patch_size changed", [](Mut& m) { gguf_set_val_u32(m.g, "timestral.patch_size", 16); }, false, "table_pos_min");
    // crafted: 32-bit arithmetic would wrap to a table of length 1 (patch_size 1, huge context, pos_min INT32_MIN)
    variant(src, "overflowing rope table (wraps in 32 bits)", [](Mut& m) {
        gguf_set_val_u32(m.g, "timestral.patch_size", 1);
        gguf_set_val_u32(m.g, "timestral.context_length", 2147483644u);
        gguf_set_val_u32(m.g, "timestral.block_size", 4);
        gguf_set_val_i32(m.g, "timestral.rope.table_pos_min", INT32_MIN); }, false, "exceeds the supported maximum");
    variant(src, "pos_min INT32_MIN with ordinary sizes", [](Mut& m) { gguf_set_val_i32(m.g, "timestral.rope.table_pos_min", INT32_MIN); }, false, "table_pos_min");
    variant(src, "context too large for the table bound", [](Mut& m) {
        gguf_set_val_u32(m.g, "timestral.patch_size", 1);
        gguf_set_val_u32(m.g, "timestral.context_length", 2000000u); }, false, "exceeds the supported maximum");
    variant(src, "feed_forward_length changed", [](Mut& m) { gguf_set_val_u32(m.g, "timestral.feed_forward_length", 48); }, false, "shape");
    // bias flags that do not match the tensors in the file
    // flipping a bias flag: a file without bias tensors now lacks them, a file with them has extras
    const char* flip_needle = Model::load(src).config().attn_bias ? "unexpected tensor" : "missing tensor";
    variant(src, "bias flag flipped", [](Mut& m) {
        bool cur = gguf_get_val_bool(m.g, gguf_find_key(m.g, "timestral.attention.bias"));
        gguf_set_val_bool(m.g, "timestral.attention.bias", !cur); }, false, flip_needle);
    variant(src, "mlp bias flag flipped", [](Mut& m) {
        bool cur = gguf_get_val_bool(m.g, gguf_find_key(m.g, "timestral.mlp.bias"));
        gguf_set_val_bool(m.g, "timestral.mlp.bias", !cur); }, false, flip_needle);
    variant(src, "gate turned off while gate tensors exist", [](Mut& m) { gguf_set_val_str(m.g, "timestral.attention.variate_gate", "none"); }, false, "unexpected tensor");

    // tensors
    variant(src, "missing tensor", [](Mut& m) { auto& v = *m.tensors; for (size_t i = 0; i < v.size(); ++i) if (!std::strcmp(v[i]->name, "blk.0.ffn_norm.weight")) { v.erase(v.begin() + i); break; } }, false, "missing tensor: blk.0.ffn_norm.weight");
    variant(src, "extra tensor", [](Mut& m) { ggml_tensor* t = ggml_new_tensor_1d(m.ctx, GGML_TYPE_F32, 4); ggml_set_name(t, "extra.weight"); std::memset(t->data, 0, 16); m.tensors->push_back(t); }, false, "unexpected tensor");
    variant(src, "F16 instead of F32", [](Mut& m) {
        ggml_tensor*& t = find(m, "out_norm.weight");
        ggml_tensor* n = clone_with_ne(m.ctx, t, GGML_TYPE_F16, t->ne[0], 1, 1, 1);
        ggml_fp32_to_fp16_row((const float*)t->data, (ggml_fp16_t*)n->data, t->ne[0]);
        t = n; }, false, "has type f16");
    variant(src, "I32 instead of F32", [](Mut& m) {
        ggml_tensor*& t = find(m, "out_norm.weight");
        ggml_tensor* n = clone_with_ne(m.ctx, t, GGML_TYPE_I32, t->ne[0], 1, 1, 1);
        std::memcpy(n->data, t->data, ggml_nbytes(t));   // same bytes, other type
        t = n; }, false, "has type i32");
    variant(src, "qkv transposed (same bytes)", [](Mut& m) {
        ggml_tensor*& t = find(m, "blk.0.attn.0.qkv.weight");
        ggml_tensor* n = clone_with_ne(m.ctx, t, GGML_TYPE_F32, t->ne[1], t->ne[0], 1, 1);
        std::memcpy(n->data, t->data, ggml_nbytes(t));
        t = n; }, false, "blk.0.attn.0.qkv.weight has shape");
    variant(src, "vector stored as a column matrix (same bytes)", [](Mut& m) {
        ggml_tensor*& t = find(m, "out_norm.weight");
        ggml_tensor* n = clone_with_ne(m.ctx, t, GGML_TYPE_F32, t->ne[0], 1, 1, 1);
        n->ne[0] = 1; n->ne[1] = t->ne[0]; n->nb[1] = n->nb[0] * n->ne[0]; n->nb[2] = n->nb[3] = n->nb[1] * n->ne[1];
        std::memcpy(n->data, t->data, ggml_nbytes(t));
        t = n; }, false, "out_norm.weight has shape");
    variant(src, "rope.cos with one position too few", [](Mut& m) {
        ggml_tensor*& t = find(m, "rope.cos");
        ggml_tensor* n = clone_with_ne(m.ctx, t, GGML_TYPE_F32, t->ne[0], t->ne[1] - 1, 1, 1);
        std::memcpy(n->data, t->data, ggml_nbytes(n));
        t = n; }, false, "rope.cos has shape");
    variant(src, "rope.inv_freq of the wrong length", [](Mut& m) {
        ggml_tensor*& t = find(m, "rope.inv_freq");
        ggml_tensor* n = clone_with_ne(m.ctx, t, GGML_TYPE_F32, t->ne[0] + 1, 1, 1, 1);
        std::memset(n->data, 0, ggml_nbytes(n)); std::memcpy(n->data, t->data, ggml_nbytes(t));
        t = n; }, false, "rope.inv_freq has shape");
    variant(src, "head.out wrong width", [](Mut& m) {
        ggml_tensor*& t = find(m, "head.out.weight");
        ggml_tensor* n = clone_with_ne(m.ctx, t, GGML_TYPE_F32, t->ne[0], t->ne[1] + 1, 1, 1);
        std::memset(n->data, 0, ggml_nbytes(n));
        t = n; }, false, "head.out.weight has shape");
    // data of one tensor cut short: file smaller than the tensor table says
    {
        std::string full = make_variant(src, "variant.gguf", [](Mut&) {});
        std::string bytes = read_file(full);
        std::string cut = tmp_path("variant_cut.gguf");
        std::ofstream(cut, std::ios::binary).write(bytes.data(), (std::streamsize)bytes.size() - 64);
        try { Model m = Model::load(cut); EXPECT(false, "truncated file loads"); }
        catch (const Error& e) { EXPECT(true, "truncated file refused"); std::printf("  refused [truncated file]: %s\n", e.what()); }
    }
    {   // not a gguf at all
        std::string junk = tmp_path("variant_junk.gguf");
        std::ofstream(junk, std::ios::binary) << "this is not a gguf file";
        try { Model m = Model::load(junk); EXPECT(false, "junk file loads"); }
        catch (const Error&) { EXPECT(true, "junk file refused"); }
        try { Model m = Model::load(tmp_path("does_not_exist.gguf")); EXPECT(false, "missing file loads"); }
        catch (const Error&) { EXPECT(true, "missing file refused"); }
    }
}

int main(int argc, char** argv) {
    bool real = argc > 1 && std::string(argv[1]) == "--real";
    try {
        if (real) {
            const char* p = std::getenv("TIMESTRAL_TEST_GGUF");
            if (!p || !*p) { std::fprintf(stderr, "SKIP: TIMESTRAL_TEST_GGUF is not set\n"); return 77; }
            check_file(p, "209M", /*compare_bytes=*/false, /*print_names=*/true);
            {   // the count the plan states for the 209M model
                Model m = Model::load(p);
                EXPECT(m.tensor_names().size() == 206, "209M has 206 tensors");
                EXPECT(m.config().n_layers == 24 && m.config().d_model == 768 && m.config().num_heads == 12, "209M config is the released one");
            }
        } else {
            const char* d = std::getenv("TIMESTRAL_TEST_TINY");
            std::string dir = d && *d ? d : std::string(TS_SOURCE_DIR) + "/tests/fixtures/tiny";
            for (const char* v : {"tiny", "tiny-bias"}) {
                std::string path = dir + "/timestral-" + v + "-f32.gguf";
                check_file(path, v, true, true);
                negative_tests(path, v);
            }
            // the two fixtures must differ in the bias tensors only
            Model a = Model::load(dir + "/timestral-tiny-f32.gguf");
            Model b = Model::load(dir + "/timestral-tiny-bias-f32.gguf");
            EXPECT(!a.config().attn_bias && !a.config().mlp_bias && b.config().attn_bias && b.config().mlp_bias, "tiny has no biases, tiny-bias has both");
            EXPECT(b.has("blk.0.attn.0.qkv.bias") && !a.has("blk.0.attn.0.qkv.bias"), "bias tensors exist only in tiny-bias");
            EXPECT(b.has("blk.0.ffn_w12.bias") && b.has("blk.0.ffn_w3.bias") && !a.has("blk.0.ffn_w3.bias"), "mlp bias tensors");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "unexpected error: %s\n", e.what());
        return 1;
    }
    std::fprintf(stderr, "\n%d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
