#include "standard_ops.hpp"
#include "common.hpp"
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
ggml_tensor* standard_rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* scale, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), scale);
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
