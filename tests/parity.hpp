#pragma once
// Parity helpers for timestral.cpp tests (plan section D).
//
// Comparison kinds (D.1):
//   EXACT   compare_exact_i32 / _i8 / _f32   every element equal
//   ULP(k)  compare_ulp                      distance in float32 bit patterns <= k
//   MAG(k)  compare_mag                      |c - r| <= k * u * M_i, u = 2^-24
//   NORM    compare_norm                     max|c - r| / rms(r) <= max(2 * floor, 16 u)
//
// Rules that apply to every helper:
//   - sizes must be equal; a size mismatch fails.
//   - a float comparison of zero elements fails (nothing was compared).
//   - NaN equals NaN only at the same position; +inf and -inf equal only themselves.
//     A NaN or inf that the reference does not have in the same place always fails.
//   - NaN in any tolerance (floor, k, M) fails; a comparison never passes by accident
//     through a false `>` test on NaN.
// Every function returns a Result that converts to bool and prints one line to stderr.

#include "ggml.h"
#include "gguf.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#if defined(__GLIBC__)
#include <gnu/libc-version.h>
#endif

namespace tstest {

constexpr int kSkip = 77;                 // ctest SKIP_RETURN_CODE
constexpr double kU = 5.9604644775390625e-8; // 2^-24

struct Result {
    bool ok = false;
    size_t n = 0;
    double err = 0.0;        // max error in the unit of the comparison kind
    double tol = 0.0;
    size_t worst = 0;
    size_t nbad = 0;
    bool cap_warning = false;
    explicit operator bool() const { return ok; }
};

namespace detail {

inline Result fail_msg(const char* label, const char* why) {
    std::fprintf(stderr, "[%s] %s -> FAIL\n", label, why);
    return Result{};
}

inline uint32_t bits_of(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }

// Monotonic integer key of a non-NaN float: adjacent floats differ by 1, +0 and -0 are both 0.
inline int64_t ulp_key(float f) {
    uint32_t b = bits_of(f);
    int64_t mag = (int64_t)(b & 0x7fffffffu);
    return (b & 0x80000000u) ? -mag : mag;
}

// Non-finite agreement. Returns 0 if both finite, 1 if both non-finite and equal in class,
// -1 if they disagree (one finite and one not, NaN vs inf, +inf vs -inf).
inline int nonfinite_agree(float c, float r) {
    bool cf = std::isfinite(c), rf = std::isfinite(r);
    if (cf && rf) return 0;
    if (cf != rf) return -1;
    if (std::isnan(c) && std::isnan(r)) return 1;
    if (std::isnan(c) || std::isnan(r)) return -1;
    return (c == r) ? 1 : -1;
}

} // namespace detail

// ---------------------------------------------------------------------------
// Baseline GGUF loading
// ---------------------------------------------------------------------------

struct Baseline {
    gguf_context* g = nullptr;
    ggml_context* ctx = nullptr;
    Baseline() = default;
    Baseline(const Baseline&) = delete;
    Baseline& operator=(const Baseline&) = delete;
    ~Baseline() { if (g) gguf_free(g); if (ctx) ggml_free(ctx); }
    bool open(const std::string& path, bool load_data) {
        gguf_init_params p{ /*no_alloc=*/!load_data, /*ctx=*/load_data ? &ctx : nullptr };
        g = gguf_init_from_file(path.c_str(), p);
        if (!g) std::fprintf(stderr, "[parity] failed to open baseline: %s\n", path.c_str());
        return g != nullptr;
    }
};

// Copy a tensor of exactly type `want` into `out` (flattened, row-major as written by torch;
// GGUF stores the shape reversed). `shape` is reported slowest to fastest. Fails when the
// tensor is missing or has another type: a silent reinterpretation would hide a bad dump.
template <typename T>
inline bool load_typed(const std::string& path, const std::string& name, ggml_type want,
                       std::vector<T>& out, std::vector<int64_t>* shape) {
    Baseline b;
    if (!b.open(path, true)) return false;
    ggml_tensor* t = ggml_get_tensor(b.ctx, name.c_str());
    if (!t) {
        std::fprintf(stderr, "[parity] tensor '%s' not found in %s\n", name.c_str(), path.c_str());
        return false;
    }
    if (t->type != want) {
        std::fprintf(stderr, "[parity] tensor '%s' has type %s, expected %s\n", name.c_str(),
                     ggml_type_name(t->type), ggml_type_name(want));
        return false;
    }
    if (shape) {
        shape->clear();
        for (int i = ggml_n_dims(t) - 1; i >= 0; --i) shape->push_back(t->ne[i]);
    }
    size_t n = (size_t)ggml_nelements(t);
    out.resize(n);
    if (n) std::memcpy(out.data(), t->data, n * sizeof(T));
    return true;
}

inline bool load_baseline(const std::string& path, const std::string& name,
                          std::vector<float>& out, std::vector<int64_t>& shape) {
    return load_typed<float>(path, name, GGML_TYPE_F32, out, &shape);
}
inline bool load_baseline_f64(const std::string& path, const std::string& name,
                              std::vector<double>& out, std::vector<int64_t>& shape) {
    return load_typed<double>(path, name, GGML_TYPE_F64, out, &shape);
}
inline bool load_baseline_i32(const std::string& path, const std::string& name,
                              std::vector<int32_t>& out) {
    return load_typed<int32_t>(path, name, GGML_TYPE_I32, out, nullptr);
}
inline bool load_baseline_i8(const std::string& path, const std::string& name,
                             std::vector<int8_t>& out) {
    return load_typed<int8_t>(path, name, GGML_TYPE_I8, out, nullptr);
}

// KV readers. Each returns false when the file or the key is missing, or the type is wrong.
inline bool load_kv_str(const std::string& path, const std::string& key, std::string& out) {
    Baseline b;
    if (!b.open(path, false)) return false;
    int64_t id = gguf_find_key(b.g, key.c_str());
    if (id < 0) {
        std::fprintf(stderr, "[parity] key '%s' not found in %s\n", key.c_str(), path.c_str());
        return false;
    }
    if (gguf_get_kv_type(b.g, id) != GGUF_TYPE_STRING) {
        std::fprintf(stderr, "[parity] key '%s' is not a string\n", key.c_str());
        return false;
    }
    out = gguf_get_val_str(b.g, id);
    return true;
}

// Any scalar numeric KV, read as double.
inline bool load_kv_num(const std::string& path, const std::string& key, double& out,
                        bool quiet = false) {
    Baseline b;
    if (!b.open(path, false)) return false;
    int64_t id = gguf_find_key(b.g, key.c_str());
    if (id < 0) {
        if (!quiet) std::fprintf(stderr, "[parity] key '%s' not found in %s\n", key.c_str(), path.c_str());
        return false;
    }
    switch (gguf_get_kv_type(b.g, id)) {
        case GGUF_TYPE_FLOAT32: out = gguf_get_val_f32(b.g, id); return true;
        case GGUF_TYPE_FLOAT64: out = gguf_get_val_f64(b.g, id); return true;
        case GGUF_TYPE_UINT8:   out = gguf_get_val_u8(b.g, id);  return true;
        case GGUF_TYPE_INT8:    out = gguf_get_val_i8(b.g, id);  return true;
        case GGUF_TYPE_UINT16:  out = gguf_get_val_u16(b.g, id); return true;
        case GGUF_TYPE_INT16:   out = gguf_get_val_i16(b.g, id); return true;
        case GGUF_TYPE_UINT32:  out = gguf_get_val_u32(b.g, id); return true;
        case GGUF_TYPE_INT32:   out = gguf_get_val_i32(b.g, id); return true;
        case GGUF_TYPE_UINT64:  out = (double)gguf_get_val_u64(b.g, id); return true;
        case GGUF_TYPE_INT64:   out = (double)gguf_get_val_i64(b.g, id); return true;
        default:
            std::fprintf(stderr, "[parity] key '%s' is not a number\n", key.c_str());
            return false;
    }
}

// uint32 KV, 0 when absent.
inline uint32_t read_kv_u32(const std::string& path, const std::string& key) {
    double v = 0;
    return load_kv_num(path, key, v, true) ? (uint32_t)v : 0u;
}

// ---------------------------------------------------------------------------
// EXACT
// ---------------------------------------------------------------------------

template <typename T>
inline Result compare_exact_int(const std::vector<T>& got, const std::vector<T>& ref,
                                const char* label) {
    Result r;
    r.n = got.size();
    if (got.size() != ref.size()) {
        std::fprintf(stderr, "[%s] size mismatch got=%zu ref=%zu -> FAIL\n", label, got.size(), ref.size());
        return r;
    }
    bool first = true;
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != ref[i]) {
            ++r.nbad;
            if (first) { r.worst = i; first = false; }
        }
    }
    r.ok = (r.nbad == 0);
    if (r.ok) std::fprintf(stderr, "[%s] n=%zu exact -> OK\n", label, r.n);
    else std::fprintf(stderr, "[%s] n=%zu %zu differ, first@%zu got=%lld ref=%lld -> FAIL\n", label,
                      r.n, r.nbad, r.worst, (long long)got[r.worst], (long long)ref[r.worst]);
    return r;
}

inline Result compare_exact_i32(const std::vector<int32_t>& got, const std::vector<int32_t>& ref,
                                const char* label) { return compare_exact_int(got, ref, label); }
inline Result compare_exact_i8(const std::vector<int8_t>& got, const std::vector<int8_t>& ref,
                               const char* label) { return compare_exact_int(got, ref, label); }

// Float tensors made only by copies and selects: every element equal (+0 == -0, NaN equal
// to NaN at the same position). Same as ULP(0).
inline Result compare_exact_f32(const std::vector<float>& got, const std::vector<float>& ref,
                                const char* label);

// ---------------------------------------------------------------------------
// ULP(k)
// ---------------------------------------------------------------------------

inline Result compare_ulp(const std::vector<float>& got, const std::vector<float>& ref,
                          int64_t k, const char* label) {
    Result r;
    r.n = got.size();
    r.tol = (double)k;
    if (got.size() != ref.size()) {
        std::fprintf(stderr, "[%s] size mismatch got=%zu ref=%zu -> FAIL\n", label, got.size(), ref.size());
        return r;
    }
    if (got.empty()) return detail::fail_msg(label, "no elements");
    if (k < 0) return detail::fail_msg(label, "negative k");
    int64_t maxd = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        int nf = detail::nonfinite_agree(got[i], ref[i]);
        int64_t d;
        if (nf < 0) d = INT64_MAX;                 // disagreement on NaN/inf
        else if (nf > 0) d = 0;                    // same non-finite value
        else {
            // finite vs finite; inf cannot reach here
            d = std::llabs(detail::ulp_key(got[i]) - detail::ulp_key(ref[i]));
        }
        if (d > k) ++r.nbad;
        if (d > maxd) { maxd = d; r.worst = i; }
    }
    r.err = (maxd == INT64_MAX) ? INFINITY : (double)maxd;
    r.ok = (r.nbad == 0);
    std::fprintf(stderr, "[%s] n=%zu ulp max=%.0f k=%lld bad=%zu (worst@%zu got=%.9g ref=%.9g) -> %s\n",
                 label, r.n, r.err, (long long)k, r.nbad, r.worst, got[r.worst], ref[r.worst],
                 r.ok ? "OK" : "FAIL");
    return r;
}

inline Result compare_exact_f32(const std::vector<float>& got, const std::vector<float>& ref,
                                const char* label) {
    return compare_ulp(got, ref, 0, label);
}

// ---------------------------------------------------------------------------
// MAG(k, M): |c_i - r_i| <= k * u * M_i
// ---------------------------------------------------------------------------

inline Result compare_mag(const std::vector<float>& got, const std::vector<float>& ref,
                          const std::vector<double>& mag, double k, const char* label) {
    Result r;
    r.n = got.size();
    r.tol = k;
    if (got.size() != ref.size() || mag.size() != got.size()) {
        std::fprintf(stderr, "[%s] size mismatch got=%zu ref=%zu mag=%zu -> FAIL\n", label,
                     got.size(), ref.size(), mag.size());
        return r;
    }
    if (got.empty()) return detail::fail_msg(label, "no elements");
    if (!(k >= 0.0) || !std::isfinite(k)) return detail::fail_msg(label, "k must be finite and >= 0");
    double worst_ratio = 0.0; // |d| / (u * M), in units of u*M
    for (size_t i = 0; i < got.size(); ++i) {
        if (!(mag[i] >= 0.0) || !std::isfinite(mag[i])) {
            ++r.nbad; r.worst = i; worst_ratio = INFINITY; continue;   // bad magnitude never passes
        }
        int nf = detail::nonfinite_agree(got[i], ref[i]);
        if (nf < 0) { ++r.nbad; r.worst = i; worst_ratio = INFINITY; continue; }
        if (nf > 0) continue;
        double d = std::fabs((double)got[i] - (double)ref[i]);
        double bound = k * kU * mag[i];
        double ratio = (mag[i] > 0.0) ? d / (kU * mag[i]) : (d == 0.0 ? 0.0 : INFINITY);
        if (!(d <= bound)) ++r.nbad;
        if (ratio > worst_ratio) { worst_ratio = ratio; r.worst = i; }
    }
    r.err = worst_ratio;
    r.ok = (r.nbad == 0);
    std::fprintf(stderr, "[%s] n=%zu mag max|d|/(u*M)=%.4g k=%.4g bad=%zu (worst@%zu got=%.9g ref=%.9g) -> %s\n",
                 label, r.n, r.err, k, r.nbad, r.worst, got[r.worst], ref[r.worst],
                 r.ok ? "OK" : "FAIL");
    return r;
}

// Same magnitude for every element.
inline Result compare_mag(const std::vector<float>& got, const std::vector<float>& ref,
                          double mag, double k, const char* label) {
    return compare_mag(got, ref, std::vector<double>(got.size(), mag), k, label);
}

// ---------------------------------------------------------------------------
// NORM: err = max|c - r| / rms(r), gate err <= tol, tol = max(2 * floor, 16 u)
// ---------------------------------------------------------------------------

enum class FloorKind { TeacherForced, FreeRunning };

inline const char* floor_prefix(FloorKind k) {
    return k == FloorKind::TeacherForced ? "floor_tf." : "floor_fr.";
}

// `floor` and `cap` are given directly. `cap` may be NaN (printed n/a): then no warning.
// When 2 * floor > cap the result carries cap_warning and a warning is printed, but the
// verdict still uses tol = max(2 * floor, 16 u) (D.1: the cap is a sanity check, not a limit).
inline Result compare_norm(const std::vector<float>& got, const std::vector<float>& ref,
                           double floor_value, double cap, const char* label) {
    Result r;
    r.n = got.size();
    if (got.size() != ref.size()) {
        std::fprintf(stderr, "[%s] size mismatch got=%zu ref=%zu -> FAIL\n", label, got.size(), ref.size());
        return r;
    }
    if (got.empty()) return detail::fail_msg(label, "no elements");
    if (!(floor_value >= 0.0) || !std::isfinite(floor_value))
        return detail::fail_msg(label, "floor must be finite and >= 0");
    r.tol = std::max(2.0 * floor_value, 16.0 * kU);

    double maxd = 0.0, sumsq = 0.0;
    size_t nfin = 0;
    bool nonfinite_bad = false;
    size_t worst = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        int nf = detail::nonfinite_agree(got[i], ref[i]);
        if (nf < 0) { nonfinite_bad = true; ++r.nbad; worst = i; continue; }
        if (nf > 0) continue;
        double d = std::fabs((double)got[i] - (double)ref[i]);
        sumsq += (double)ref[i] * (double)ref[i];
        ++nfin;
        if (d > maxd) { maxd = d; worst = i; }
    }
    r.worst = worst;
    double rms = nfin ? std::sqrt(sumsq / (double)nfin) : 0.0;
    if (nonfinite_bad) r.err = INFINITY;
    else if (rms > 0.0) r.err = maxd / rms;
    else r.err = (maxd == 0.0) ? 0.0 : INFINITY;   // an all-zero reference tolerates no difference
    r.ok = !nonfinite_bad && (r.err <= r.tol);      // NaN err cannot pass
    if (!r.ok && r.nbad == 0) r.nbad = 1;
    r.cap_warning = std::isfinite(cap) && (2.0 * floor_value > cap);

    char capbuf[32];
    if (std::isfinite(cap)) std::snprintf(capbuf, sizeof capbuf, "%.3e", cap);
    else std::snprintf(capbuf, sizeof capbuf, "n/a");
    std::fprintf(stderr, "[%s] n=%zu err=%.3e tol=%.3e floor=%.3e cap=%s -> %s\n", label, r.n,
                 r.err, r.tol, floor_value, capbuf, r.ok ? "OK" : "FAIL");
    if (r.cap_warning)
        std::fprintf(stderr, "[%s] WARNING: 2*floor=%.3e exceeds cap=%.3e; record the case in docs/parity.md\n",
                     label, 2.0 * floor_value, cap);
    return r;
}

// Reads `floor_tf.<tensor>` or `floor_fr.<tensor>` and `cap.<tensor>` (optional) from the
// baseline file. A missing floor fails: without it the tolerance is unknown, and guessing
// one would let a bad port pass or a good port fail.
inline Result compare_norm(const std::vector<float>& got, const std::vector<float>& ref,
                           const std::string& baseline_path, const std::string& tensor,
                           FloorKind kind, const char* label) {
    double fl = 0.0, cap = std::nan("");
    if (!load_kv_num(baseline_path, std::string(floor_prefix(kind)) + tensor, fl))
        return detail::fail_msg(label, "floor key missing in the baseline");
    load_kv_num(baseline_path, "cap." + tensor, cap, /*quiet=*/true);
    return compare_norm(got, ref, fl, cap, label);
}

// ---------------------------------------------------------------------------
// Machine check (D.4)
// ---------------------------------------------------------------------------

// The CPU capability name that torch.backends.cpu.get_cpu_capability() reports on x86.
inline std::string running_cpu_capability() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl") &&
        __builtin_cpu_supports("avx512cd"))
        return "AVX512";
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return "AVX2";
    return "DEFAULT";
#else
    return "DEFAULT";
#endif
}

inline std::string running_glibc() {
#if defined(__GLIBC__)
    return gnu_get_libc_version();
#else
    return "";
#endif
}

// Keep the first "digits.digits..." token, so "2.39", "glibc 2.39" and "('glibc', '2.39')" agree.
inline std::string normalize_glibc(const std::string& s) {
    size_t i = 0;
    while (i < s.size() && !(s[i] >= '0' && s[i] <= '9')) ++i;
    size_t j = i;
    while (j < s.size() && ((s[j] >= '0' && s[j] <= '9') || s[j] == '.')) ++j;
    return s.substr(i, j - i);
}

inline std::string normalize_cap(const std::string& s) {
    std::string o;
    for (char c : s) if (!std::isspace((unsigned char)c)) o += (char)std::toupper((unsigned char)c);
    return o;
}

// Pure comparison: both fields must be known and equal.
inline bool host_matches(const std::string& base_cap, const std::string& base_glibc,
                         const std::string& run_cap, const std::string& run_glibc) {
    std::string bc = normalize_cap(base_cap), rc = normalize_cap(run_cap);
    std::string bg = normalize_glibc(base_glibc), rg = normalize_glibc(run_glibc);
    if (bc.empty() || rc.empty() || bg.empty() || rg.empty()) return false;
    return bc == rc && bg == rg;
}

// True when the baseline was made on a machine like this one, so the host float gates
// (ULP, MAG, host EXACT) apply. A baseline without env keys counts as a different machine.
inline bool host_gates_available(const std::string& baseline_path) {
    std::string cap, glibc;
    bool a = load_kv_str(baseline_path, "env.cpu_capability", cap);
    bool b = load_kv_str(baseline_path, "env.glibc", glibc);
    return a && b && host_matches(cap, glibc, running_cpu_capability(), running_glibc());
}

// Call at the start of a host float gate. Returns true when the gate may run; otherwise
// prints the D.4 message and the caller returns kSkip.
inline bool host_gates_or_skip(const std::string& baseline_path) {
    if (host_gates_available(baseline_path)) return true;
    std::fprintf(stderr, "[parity] SKIP: host gates need baselines from this machine (%s)\n",
                 baseline_path.c_str());
    return false;
}

} // namespace tstest
