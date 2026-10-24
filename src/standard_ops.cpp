#include "standard_ops.hpp"
#include "common.hpp"
#include "backend.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
namespace ts {
namespace {
constexpr int64_t kReducePanel = 192;
std::atomic<int>& state() {
    // -1 = not yet resolved.
    static std::atomic<int> value{-1};
    return value;
}
int resolve_default() {
#ifdef TS_STANDARD_OPS_DEFAULT
    int v = 1;
#else
    int v = 0;
#endif
    if (const char* e = std::getenv("TIMESTRAL_STANDARD_OPS")) {
        if (!std::strcmp(e, "1")) v = 1;
        else if (!std::strcmp(e, "0")) v = 0;
    }
    return v;
}
}
bool standard_ops() {
    int v = state().load();
    if (v < 0) {
        int d = resolve_default();
        state().compare_exchange_strong(v, d);
        v = state().load();
    }
    return v != 0;
}
void set_standard_ops(bool enabled) { state().store(enabled ? 1 : 0); }

ggml_tensor* standard_linear(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, ggml_tensor* bias) {
    if (!w || !x || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[3] != 1 || w->ne[0] != x->ne[0] ||
        (bias && (bias->type != GGML_TYPE_F32 || ggml_nelements(bias) != w->ne[1])))
        throw Error("invalid standard linear layout");
    // The reference F32 GEMM reduces K in panels of kReducePanel with one
    // accumulator chain per panel and adds the panels in order. A single
    // mul_mat over all of K has a different (and, with outlier activations,
    // noticeably different) rounding path, so contractions of K >= panel are
    // split the same way. Short dots stay one mul_mat.
    ggml_tensor* y = nullptr;
    if (w->ne[0] < kReducePanel) {
        y = ggml_mul_mat(ctx, w, x);
    } else {
        for (int64_t off = 0; off < w->ne[0]; off += kReducePanel) {
            const int64_t n = std::min<int64_t>(kReducePanel, w->ne[0] - off);
            auto wv = ggml_view_2d(ctx, w, n, w->ne[1], w->nb[1], size_t(off) * w->nb[0]);
            auto xv = ggml_view_3d(ctx, x, n, x->ne[1], x->ne[2], x->nb[1], x->nb[2], size_t(off) * x->nb[0]);
            auto part = ggml_mul_mat(ctx, wv, xv);
            y = y ? ggml_add(ctx, y, part) : part;
        }
    }
    return bias ? ggml_add(ctx, y, bias) : y;
}
// The reference sums x*x in F32 with a fixed cascade: 32 independent lanes
// (four groups of eight) accumulate group after group, partial lane sets are
// merged in power-of-two blocks, then the lanes are folded in a fixed order.
// With outlier activations the output error is dominated by the last bit of
// 1/sqrt(mean + eps), so the order of this sum matters. ggml_rms_norm sums in
// double, which is more exact but rounds differently. This builds the same
// cascade from element-wise operations, whose results are exact IEEE F32 on
// every backend.
ggml_tensor* standard_rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* scale, float eps) {
    const int64_t D = x->ne[0];
    if (D % 32 != 0 || D < 32 || !ggml_is_contiguous(x))
        return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), scale);
    const int64_t G = D / 32;
    auto sq = ggml_sqr(ctx, x);
    auto group = [&](int64_t g) {
        return ggml_view_3d(ctx, sq, 32, x->ne[1], x->ne[2], sq->nb[1], sq->nb[2], size_t(g) * 32 * sizeof(float));
    };
    ggml_tensor* acc[4] = {nullptr, nullptr, nullptr, nullptr}; // null is an all-zero lane set
    auto accumulate = [&](ggml_tensor*& into, ggml_tensor* v) { into = into ? ggml_add(ctx, into, v) : v; };
    int log = 0;
    for (int64_t n = G - 1; n > 0; n >>= 1) ++log;
    const int power = std::max(4, log / 4);
    const int64_t step = int64_t(1) << power, mask = step - 1;
    int64_t i = 0;
    while (i + step <= G) {
        for (int64_t j = 0; j < step; ++j, ++i) accumulate(acc[0], group(i));
        for (int j = 1; j < 4; ++j) {
            if (acc[j - 1]) accumulate(acc[j], acc[j - 1]);
            acc[j - 1] = nullptr;
            if (i & (mask << (j * power))) break;
        }
    }
    for (; i < G; ++i) accumulate(acc[0], group(i));
    for (int j = 1; j < 4; ++j) if (acc[j]) accumulate(acc[0], acc[j]);
    // Fold the four groups of eight lanes, then add the eight lanes in order.
    auto lanes = [&](ggml_tensor* t, int64_t count, int64_t first) {
        return ggml_view_3d(ctx, t, count, t->ne[1], t->ne[2], t->nb[1], t->nb[2], size_t(first) * sizeof(float));
    };
    auto lane8 = lanes(acc[0], 8, 0);
    for (int k = 1; k < 4; ++k) lane8 = ggml_add(ctx, lane8, lanes(acc[0], 8, 8 * k));
    auto sum = lanes(lane8, 1, 0);
    for (int l = 1; l < 8; ++l) sum = ggml_add(ctx, sum, lanes(lane8, 1, l));
    auto mean = ggml_div(ctx, sum, graph_constant(ctx, float(D)));
    auto root = ggml_sqrt(ctx, ggml_scale_bias(ctx, mean, 1.0f, eps));
    auto inv = ggml_div(ctx, ggml_repeat(ctx, graph_constant(ctx, 1.0f), root), root);
    return ggml_mul(ctx, ggml_mul(ctx, x, inv), scale);
}
ggml_tensor* standard_attention(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* mask) {
    const int64_t hd = q->ne[0], H = q->ne[1], S = q->ne[2], B = q->ne[3];
    // [hd, S, H, B] views: heads become a batch dimension of mul_mat.
    auto qp = ggml_permute(ctx, q, 0, 2, 1, 3);
    auto kp = ggml_permute(ctx, k, 0, 2, 1, 3);
    auto vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)); // [S, hd, H, B]
    auto kq = ggml_mul_mat(ctx, kp, qp);                         // [S keys, S queries, H, B]
    auto m = ggml_reshape_4d(ctx, mask, mask->ne[0], mask->ne[1], 1, mask->ne[2]);
    auto p = ggml_soft_max_ext(ctx, kq, m, float(1.0 / std::sqrt(double(hd))), 0.0f);
    auto o = ggml_mul_mat(ctx, vt, p);                           // [hd, S, H, B]
    auto r = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));   // [hd, H, S, B]
    return ggml_reshape_3d(ctx, r, hd * H, S, B);
}
}
