// Smoke test: the library links, and ggml runs one graph on the CPU backend.
#include "common.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstring>
#include <vector>

int main() {
    const char* v = ts::version();
    if (v == nullptr || std::strlen(v) == 0) {
        std::fprintf(stderr, "version string is empty\n");
        return 1;
    }
    std::printf("timestral.cpp version: %s\n", v);

    ggml_init_params ip{ ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, /*no_alloc=*/true };
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor* c = ggml_add(ctx, a, b);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { std::fprintf(stderr, "alloc failed\n"); return 1; }
    const float av[4] = {1.f, 2.f, 3.f, 4.f}, bv[4] = {10.f, 20.f, 30.f, 40.f};
    ggml_backend_tensor_set(a, av, 0, sizeof av);
    ggml_backend_tensor_set(b, bv, 0, sizeof bv);
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { std::fprintf(stderr, "compute failed\n"); return 1; }
    std::vector<float> out(4);
    ggml_backend_tensor_get(c, out.data(), 0, 4 * sizeof(float));
    int rc = 0;
    for (int i = 0; i < 4; ++i) {
        if (out[i] != av[i] + bv[i]) { std::fprintf(stderr, "out[%d]=%g wrong\n", i, out[i]); rc = 1; }
    }
    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
    return rc;
}
