// Tests of the comparison helpers in parity.hpp, on hand-made vectors. Each helper gets
// known-good inputs (must pass) and known-bad inputs (must fail): a helper that always
// passes would make every later gate meaningless.
#include "parity.hpp"
#include <cstdlib>
#include <limits>

using namespace tstest;

static int g_fail = 0;
static int g_checks = 0;

#define EXPECT(cond, what) do { ++g_checks; if (!(cond)) { ++g_fail; \
    std::fprintf(stderr, "TEST FAILED line %d: %s\n", __LINE__, what); } } while (0)
#define PASS(expr, what) EXPECT(static_cast<bool>(expr), what)
#define REJECT(expr, what) EXPECT(!static_cast<bool>(expr), what)

static const float NANF = std::numeric_limits<float>::quiet_NaN();
static const float INFF = std::numeric_limits<float>::infinity();

static float next_up(float f, int n = 1) {
    for (int i = 0; i < n; ++i) f = std::nextafterf(f, INFF);
    return f;
}
static float next_down(float f, int n = 1) {
    for (int i = 0; i < n; ++i) f = std::nextafterf(f, -INFF);
    return f;
}

static void test_exact() {
    PASS(compare_exact_i32({1, 2, 3}, {1, 2, 3}, "i32 equal"), "i32 equal passes");
    REJECT(compare_exact_i32({}, {}, "i32 empty"), "i32 empty comparison fails");
    REJECT(compare_exact_i8({}, {}, "i8 empty"), "i8 empty comparison fails");
    REJECT(compare_exact_i32({1, 2, 3}, {1, 2, 4}, "i32 one off"), "i32 one differs fails");
    REJECT(compare_exact_i32({1, 2, 3}, {1, 2}, "i32 size"), "i32 size mismatch fails");
    REJECT(compare_exact_i32({0}, {-1}, "i32 sign"), "i32 sign differs fails");
    Result r = compare_exact_i32({1, 9, 3, 9}, {1, 2, 3, 4}, "i32 two off");
    EXPECT(r.nbad == 2 && r.worst == 1, "i32 reports count and first position");

    PASS(compare_exact_i8({0, 1, 1, 0}, {0, 1, 1, 0}, "i8 equal"), "i8 equal passes");
    REJECT(compare_exact_i8({0, 1, 1, 0}, {0, 1, 0, 0}, "i8 one off"), "i8 one differs fails");
    REJECT(compare_exact_i8({0, 1}, {0, 1, 1}, "i8 size"), "i8 size mismatch fails");

    PASS(compare_exact_f32({1.f, -0.f}, {1.f, 0.f}, "f32 zero sign"), "exact f32: +0 == -0");
    REJECT(compare_exact_f32({1.f, 2.f}, {1.f, next_up(2.f)}, "f32 1ulp"), "exact f32: 1 ulp fails");
}

static void test_ulp() {
    const float one = 1.0f;
    PASS(compare_ulp({one, 5.f, -3.f}, {one, 5.f, -3.f}, 0, "ulp identical"), "identical at k=0");
    PASS(compare_ulp({0.f}, {-0.f}, 0, "ulp +0/-0"), "+0 equals -0 at k=0");
    PASS(compare_ulp({-0.f}, {0.f}, 0, "ulp -0/+0"), "-0 equals +0 at k=0");

    REJECT(compare_ulp({next_up(one)}, {one}, 0, "ulp 1 at k0"), "1 ulp fails at k=0");
    PASS(compare_ulp({next_up(one)}, {one}, 1, "ulp 1 at k1"), "1 ulp passes at k=1");
    PASS(compare_ulp({next_down(one)}, {one}, 1, "ulp -1 at k1"), "-1 ulp passes at k=1 (below, other binade)");
    REJECT(compare_ulp({next_up(one, 2)}, {one}, 1, "ulp 2 at k1"), "2 ulp fails at k=1");
    PASS(compare_ulp({next_up(one, 2)}, {one}, 2, "ulp 2 at k2"), "2 ulp passes at k=2");
    REJECT(compare_ulp({next_up(one, 1000)}, {one}, 999, "ulp 1000 at k999"), "1000 ulp fails at k=999");
    PASS(compare_ulp({next_up(one, 1000)}, {one}, 1000, "ulp 1000 at k1000"), "1000 ulp passes at k=1000");
    REJECT(compare_ulp({1.5f}, {1.0f}, 100, "ulp far"), "far values fail");
    // the one failing element among good ones must be found
    REJECT(compare_ulp({1.f, 2.f, next_up(3.f, 5), 4.f}, {1.f, 2.f, 3.f, 4.f}, 4, "ulp one bad"),
           "one bad element among good ones fails");

    // across zero the distance is the sum on both sides
    float tiny = std::numeric_limits<float>::denorm_min();
    PASS(compare_ulp({tiny}, {-tiny}, 2, "ulp across zero k2"), "denorm_min vs -denorm_min is 2 ulp");
    REJECT(compare_ulp({tiny}, {-tiny}, 1, "ulp across zero k1"), "denorm_min vs -denorm_min fails at k=1");
    PASS(compare_ulp({tiny}, {0.f}, 1, "ulp denorm vs zero"), "denorm_min vs 0 is 1 ulp");

    // NaN rules
    PASS(compare_ulp({NANF, 1.f}, {NANF, 1.f}, 0, "ulp nan nan"), "NaN equals NaN at the same position");
    REJECT(compare_ulp({NANF}, {1.f}, 1000, "ulp nan vs num"), "NaN vs number fails at any k");
    REJECT(compare_ulp({1.f}, {NANF}, 1000, "ulp num vs nan"), "number vs NaN fails at any k");
    REJECT(compare_ulp({NANF, 1.f}, {1.f, NANF}, 1000, "ulp nan moved"), "NaN at different positions fails");
    REJECT(compare_ulp({NANF}, {INFF}, 1000, "ulp nan vs inf"), "NaN vs inf fails");
    // inf rules
    PASS(compare_ulp({INFF, -INFF}, {INFF, -INFF}, 0, "ulp inf"), "equal infs pass");
    REJECT(compare_ulp({INFF}, {-INFF}, 1000, "ulp +inf -inf"), "+inf vs -inf fails");
    REJECT(compare_ulp({INFF}, {std::numeric_limits<float>::max()}, 1000, "ulp inf vs max"),
           "inf vs the largest finite value fails");

    REJECT(compare_ulp({1.f, 2.f}, {1.f}, 0, "ulp size"), "size mismatch fails");
    REJECT(compare_ulp({}, {}, 0, "ulp empty"), "empty float comparison fails");
    REJECT(compare_ulp({1.f}, {1.f}, -1, "ulp negative k"), "negative k fails");
}

static void test_mag() {
    const double u = kU;
    // Exactly representable bound: M = 1024, k = 4 -> bound = 4 * 2^-24 * 2^10 = 2^-12.
    const float bound = std::ldexp(1.0f, -12);
    PASS(compare_mag({bound}, {0.f}, 1024.0, 4.0, "mag at bound"), "difference equal to the bound passes");
    REJECT(compare_mag({next_up(bound)}, {0.f}, 1024.0, 4.0, "mag over bound"),
           "difference one ulp over the bound fails");
    PASS(compare_mag({-bound}, {0.f}, 1024.0, 4.0, "mag neg"), "negative difference at the bound passes");
    REJECT(compare_mag({-next_up(bound)}, {0.f}, 1024.0, 4.0, "mag neg over"), "negative difference over fails");
    PASS(compare_mag({1.f}, {1.f}, 1024.0, 0.0, "mag k0 equal"), "equal values pass at k=0");
    REJECT(compare_mag({next_up(1.f)}, {1.f}, 1.0, 0.0, "mag k0 off"), "any difference fails at k=0");

    // Cancellation case: a sum of terms of size ~1000 gives a result near 1e-3 (a stored
    // result whose ulp is 1e-10). The port's value differs by 3 * u * 1000 = 1.8e-4, a huge
    // number of ulps of the result, but within MAG(4, 1000) and outside MAG(2, 1000).
    float ref = 1.0e-3f;
    float got = (float)((double)ref + 3.0 * u * 1000.0);
    PASS(compare_mag({got}, {ref}, 1000.0, 4.0, "mag cancel k4"), "cancellation: passes MAG(4)");
    REJECT(compare_mag({got}, {ref}, 1000.0, 2.0, "mag cancel k2"), "cancellation: fails MAG(2)");
    REJECT(compare_ulp({got}, {ref}, 4, "ulp cancel k4"), "cancellation: the same pair fails ULP(4), so ULP is the wrong unit here");
    // a result that is not near the cancellation scale is still caught by a small M
    REJECT(compare_mag({got}, {ref}, 1.0, 4.0, "mag small M"), "same error fails with the small magnitude");

    // per-element magnitudes
    PASS(compare_mag({0.f, bound}, {0.f, 0.f}, std::vector<double>{1.0, 1024.0}, 4.0, "mag vec"),
         "per-element magnitude passes");
    REJECT(compare_mag({bound, 0.f}, {0.f, 0.f}, std::vector<double>{1.0, 1024.0}, 4.0, "mag vec bad"),
           "the error placed where M is small fails");
    REJECT(compare_mag({0.f}, {0.f}, std::vector<double>{1.0, 2.0}, 4.0, "mag vec size"),
           "magnitude vector of the wrong size fails");
    REJECT(compare_mag({0.f}, {0.f}, -1.0, 4.0, "mag negative M"), "negative magnitude fails");
    REJECT(compare_mag({0.f}, {0.f}, (double)NANF, 4.0, "mag nan M"), "NaN magnitude fails");
    REJECT(compare_mag({0.f}, {0.f}, 1.0, (double)NANF, "mag nan k"), "NaN k fails");
    REJECT(compare_mag({0.f}, {0.f}, 1.0, -1.0, "mag negative k"), "negative k fails");
    PASS(compare_mag({0.f}, {0.f}, 0.0, 4.0, "mag zero M equal"), "M = 0 with equal values passes");
    REJECT(compare_mag({next_up(0.f)}, {0.f}, 0.0, 4.0, "mag zero M off"), "M = 0 with a difference fails");

    PASS(compare_mag({NANF}, {NANF}, 1.0, 4.0, "mag nan nan"), "NaN equals NaN at the same position");
    REJECT(compare_mag({NANF}, {1.f}, 1e30, 1e30, "mag nan vs num"), "NaN vs number fails");
    REJECT(compare_mag({1.f}, {NANF}, 1e30, 1e30, "mag num vs nan"), "number vs NaN fails");
    PASS(compare_mag({INFF}, {INFF}, 1.0, 4.0, "mag inf"), "equal infs pass");
    REJECT(compare_mag({INFF}, {-INFF}, 1e30, 1e30, "mag inf sign"), "opposite infs fail");
    REJECT(compare_mag({1.f}, {1.f, 2.f}, 1.0, 4.0, "mag size"), "size mismatch fails");
    REJECT(compare_mag({}, {}, 1.0, 4.0, "mag empty"), "empty comparison fails");
}

static void test_norm() {
    // reference with rms exactly 2: {2, 2, 2, 2}
    std::vector<float> ref = {2.f, 2.f, 2.f, 2.f};
    auto with_err = [&](double e) { std::vector<float> g = ref; g[2] = (float)(2.0 + e * 2.0); return g; };
    const double nan_cap = std::nan("");

    PASS(compare_norm(ref, ref, 0.0, nan_cap, "norm identical"), "identical passes with floor 0");
    // floor = 1e-4: tol = 2e-4
    PASS(compare_norm(with_err(1.9e-4), ref, 1e-4, nan_cap, "norm below tol"), "error below 2*floor passes");
    REJECT(compare_norm(with_err(2.2e-4), ref, 1e-4, nan_cap, "norm above tol"), "error above 2*floor fails");
    // a larger floor accepts what a smaller one rejects (the floor really is used)
    PASS(compare_norm(with_err(2.2e-4), ref, 2e-4, nan_cap, "norm bigger floor"), "same error passes with a bigger floor");
    // 16u minimum: floor 0 -> tol 9.5e-7
    PASS(compare_norm(with_err(8e-7), ref, 0.0, nan_cap, "norm 16u pass"), "error below 16u passes with floor 0");
    REJECT(compare_norm(with_err(1.2e-6), ref, 0.0, nan_cap, "norm 16u fail"), "error above 16u fails with floor 0");
    // the error is a max over elements, normalized by rms of the reference (not by max)
    std::vector<float> rr = {0.f, 0.f, 0.f, 4.f}; // rms = 2
    std::vector<float> gg = {0.f, 0.f, 0.f, 4.f}; gg[0] = 1e-3f;
    Result rn = compare_norm(gg, rr, 1e-4, nan_cap, "norm rms scale");
    EXPECT(!rn.ok && std::fabs(rn.err - 0.5e-3) < 1e-9, "err = max|d| / rms(ref) = 1e-3 / 2");

    REJECT(compare_norm({NANF, 2.f, 2.f, 2.f}, ref, 1.0, nan_cap, "norm got nan"), "NaN in the result fails");
    REJECT(compare_norm(ref, {NANF, 2.f, 2.f, 2.f}, 1.0, nan_cap, "norm ref nan"), "NaN only in the reference fails");
    PASS(compare_norm({NANF, 2.f, 2.f, 2.f}, {NANF, 2.f, 2.f, 2.f}, 1e-4, nan_cap, "norm nan nan"),
         "NaN at the same position passes");
    REJECT(compare_norm({INFF, 2.f, 2.f, 2.f}, ref, 1.0, nan_cap, "norm inf"), "inf vs finite fails");
    REJECT(compare_norm({1.f}, {1.f, 2.f}, 1.0, nan_cap, "norm size"), "size mismatch fails");
    REJECT(compare_norm({}, {}, 1.0, nan_cap, "norm empty"), "empty comparison fails");
    REJECT(compare_norm({NANF, NANF}, {NANF, NANF}, 1.0, nan_cap, "norm all nan"), "all-NaN inputs fail (no finite element)");
    REJECT(compare_norm({INFF, -INFF}, {INFF, -INFF}, 1.0, nan_cap, "norm all inf"), "all-inf inputs fail (no finite element)");
    REJECT(compare_norm(ref, ref, -1.0, nan_cap, "norm neg floor"), "negative floor fails");
    REJECT(compare_norm(ref, ref, (double)NANF, nan_cap, "norm nan floor"), "NaN floor fails");
    // all-zero reference: rms is 0, so a difference must fail and equality must pass
    PASS(compare_norm({0.f, 0.f}, {0.f, 0.f}, 1e-4, nan_cap, "norm zero ref equal"), "zero reference, equal passes");
    REJECT(compare_norm({1e-30f, 0.f}, {0.f, 0.f}, 1.0, nan_cap, "norm zero ref off"), "zero reference, any difference fails");

    // the cap is a warning, not a limit
    Result w = compare_norm(with_err(1e-4), ref, 1e-4, /*cap=*/1e-5, "norm cap warn pass");
    EXPECT(w.ok && w.cap_warning, "2*floor above cap: warns and still passes when the error is within tol");
    Result w2 = compare_norm(with_err(3e-4), ref, 1e-4, /*cap=*/1e-5, "norm cap warn fail");
    EXPECT(!w2.ok && w2.cap_warning, "2*floor above cap does not rescue a failing error");
    Result w3 = compare_norm(with_err(1e-4), ref, 1e-6, /*cap=*/1e-3, "norm cap fine");
    EXPECT(!w3.ok && !w3.cap_warning, "cap above 2*floor: no warning; error above tol fails");
}

static std::string tmp_path(const char* name) { return std::string(TS_TEST_TMP_DIR) + "/" + name; }

static void test_baseline_io_and_floors() {
    const std::string path = tmp_path("parity_helpers_baseline.gguf");

    gguf_context* g = gguf_init_empty();
    gguf_set_val_str(g, "env.cpu_capability", "AVX2");
    gguf_set_val_str(g, "env.glibc", "2.39");
    gguf_set_val_u32(g, "case.count", 7);
    gguf_set_val_f64(g, "floor_tf.x", 1e-4);
    gguf_set_val_f64(g, "floor_fr.x", 1e-2);
    gguf_set_val_f32(g, "cap.x", 1e-5f);
    gguf_set_val_f64(g, "floor_tf.nocap", 1e-4);
    gguf_set_val_f64(g, "floor_fr.onlyfr", 1e-4);

    ggml_init_params ip{ 1 << 20, nullptr, false };
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* tf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);   // torch shape (2, 3)
    ggml_set_name(tf, "t.f32");
    for (int i = 0; i < 6; ++i) ((float*)tf->data)[i] = 0.5f * (float)i;
    ggml_tensor* t3 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 1, 1);  // torch shape (1, 1, 3)
    ggml_set_name(t3, "t.lead1");
    for (int i = 0; i < 3; ++i) ((float*)t3->data)[i] = (float)i;
    ggml_tensor* td = ggml_new_tensor_1d(ctx, GGML_TYPE_F64, 2);
    ggml_set_name(td, "t.f64");
    ((double*)td->data)[0] = 1.25; ((double*)td->data)[1] = -2.5;
    ggml_tensor* ti = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);
    ggml_set_name(ti, "t.i32");
    ((int32_t*)ti->data)[0] = 7; ((int32_t*)ti->data)[1] = -8; ((int32_t*)ti->data)[2] = 9;
    ggml_tensor* tb = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, 4);
    ggml_set_name(tb, "t.i8");
    for (int i = 0; i < 4; ++i) ((int8_t*)tb->data)[i] = (int8_t)(i & 1);
    for (ggml_tensor* t : {tf, t3, td, ti, tb}) gguf_add_tensor(g, t);
    bool wrote = gguf_write_to_file(g, path.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
    EXPECT(wrote, "baseline fixture written");
    if (!wrote) return;

    std::vector<float> f; std::vector<int64_t> shape;
    EXPECT(load_baseline(path, "t.f32", f, shape) && f.size() == 6 && f[5] == 2.5f &&
           shape == std::vector<int64_t>({1, 1, 2, 3}), "f32 tensor loads with shape (2, 3), padded to 4 dims");
    EXPECT(load_baseline(path, "t.lead1", f, shape) && f.size() == 3 &&
           shape == std::vector<int64_t>({1, 1, 1, 3}), "torch shape (1, 1, 3) keeps its leading 1s (padded form)");
    std::vector<double> d;
    EXPECT(load_baseline_f64(path, "t.f64", d, shape) && d.size() == 2 && d[1] == -2.5, "f64 tensor loads");
    std::vector<int32_t> i32;
    EXPECT(load_baseline_i32(path, "t.i32", i32) && i32.size() == 3 && i32[1] == -8, "i32 tensor loads");
    std::vector<int8_t> i8;
    EXPECT(load_baseline_i8(path, "t.i8", i8) && i8.size() == 4 && i8[3] == 1, "i8 tensor loads");
    EXPECT(!load_baseline(path, "missing", f, shape), "missing tensor is an error");
    EXPECT(!load_baseline(path, "t.i32", f, shape), "wrong tensor type is an error, not a reinterpretation");
    EXPECT(!load_baseline_i32(path, "t.f32", i32), "f32 tensor read as i32 is an error");
    EXPECT(!load_baseline(tmp_path("does_not_exist.gguf"), "t.f32", f, shape), "missing file is an error");

    std::string s;
    EXPECT(load_kv_str(path, "env.glibc", s) && s == "2.39", "string KV loads");
    EXPECT(!load_kv_str(path, "case.count", s), "number read as string is an error");
    EXPECT(!load_kv_str(path, "nope", s), "missing string KV is an error");
    EXPECT(read_kv_u32(path, "case.count") == 7 && read_kv_u32(path, "nope") == 0, "u32 KV");
    double v = 0;
    EXPECT(load_kv_num(path, "floor_tf.x", v) && v == 1e-4, "f64 KV loads as number");
    EXPECT(load_kv_num(path, "case.count", v) && v == 7.0, "u32 KV loads as number");
    EXPECT(!load_kv_num(path, "env.glibc", v), "string KV is not a number");

    // floors from the file: err = 5e-4 passes with floor_fr = 1e-2 (tol 2e-2),
    // fails with floor_tf = 1e-4 (tol 2e-4)
    std::vector<float> ref = {2.f, 2.f, 2.f, 2.f};
    std::vector<float> got = ref; got[1] = 2.f + 2.f * 5e-4f;
    PASS(compare_norm(got, ref, path, "x", FloorKind::FreeRunning, "file fr"), "free-running floor from the file");
    REJECT(compare_norm(got, ref, path, "x", FloorKind::TeacherForced, "file tf"), "teacher-forced floor is the smaller one and rejects");
    Result noc = compare_norm(ref, ref, path, "nocap", FloorKind::TeacherForced, "file nocap");
    EXPECT(noc.ok && !noc.cap_warning, "absent cap is allowed and gives no warning");
    Result cw = compare_norm(ref, ref, path, "x", FloorKind::TeacherForced, "file cap");
    EXPECT(cw.ok && cw.cap_warning, "cap read from the file triggers the warning (2e-4 > 1e-5)");
    REJECT(compare_norm(ref, ref, path, "onlyfr", FloorKind::TeacherForced, "file no floor"),
           "a missing floor fails instead of defaulting to a tolerance");
    REJECT(compare_norm(ref, ref, path, "unknown", FloorKind::FreeRunning, "file unknown"), "unknown tensor name fails");
    REJECT(compare_norm(ref, ref, tmp_path("does_not_exist.gguf"), "x", FloorKind::FreeRunning, "file nofile"),
           "missing baseline file fails");

    // machine check on the file
    EXPECT(!host_gates_available(tmp_path("does_not_exist.gguf")), "missing file: host gates unavailable");
}

static void test_host_check() {
    EXPECT(host_matches("AVX2", "2.39", "AVX2", "2.39"), "same capability and glibc match");
    EXPECT(host_matches(" avx2 ", "glibc 2.39", "AVX2", "2.39"), "case, spaces and a glibc prefix are normalized");
    EXPECT(host_matches("AVX512", "('glibc', '2.39')", "AVX512", "2.39"), "python tuple string normalizes");
    EXPECT(!host_matches("AVX2", "2.39", "AVX512", "2.39"), "different capability does not match");
    EXPECT(!host_matches("AVX2", "2.39", "AVX2", "2.35"), "different glibc does not match");
    EXPECT(!host_matches("AVX2", "2.3", "AVX2", "2.39"), "2.3 is not 2.39");
    EXPECT(!host_matches("", "2.39", "AVX2", "2.39"), "missing capability does not match");
    EXPECT(!host_matches("AVX2", "", "AVX2", ""), "unknown glibc on both sides does not match");
    EXPECT(!running_cpu_capability().empty(), "running capability is detected");
    std::fprintf(stderr, "[host] cpu_capability=%s glibc=%s\n", running_cpu_capability().c_str(),
                 running_glibc().c_str());

    // file-based: a baseline made for this machine passes, one for another machine does not
    for (int same = 0; same < 2; ++same) {
        std::string path = tmp_path(same ? "host_same.gguf" : "host_other.gguf");
        gguf_context* g = gguf_init_empty();
        gguf_set_val_str(g, "env.cpu_capability", same ? running_cpu_capability().c_str() : "NOT-A-CPU");
        gguf_set_val_str(g, "env.glibc", same ? running_glibc().c_str() : "0.0");
        bool ok = gguf_write_to_file(g, path.c_str(), true);
        gguf_free(g);
        EXPECT(ok, "host fixture written");
        if (!ok) continue;
        if (running_glibc().empty() && same) continue; // no glibc here: cannot match by design
        EXPECT(host_gates_available(path) == (same == 1), same ? "own machine: gates run" : "other machine: gates skip");
    }
    // baseline without env keys counts as another machine
    std::string path = tmp_path("host_nokeys.gguf");
    gguf_context* g = gguf_init_empty();
    gguf_set_val_u32(g, "something", 1);
    gguf_write_to_file(g, path.c_str(), true);
    gguf_free(g);
    EXPECT(!host_gates_available(path), "baseline without env keys: gates skip");
}

int main() {
    test_exact();
    test_ulp();
    test_mag();
    test_norm();
    test_baseline_io_and_floors();
    test_host_check();
    std::fprintf(stderr, "\n%d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
