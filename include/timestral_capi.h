#ifndef TIMESTRAL_CAPI_H
#define TIMESTRAL_CAPI_H
#include <stdint.h>
#if defined(_WIN32) && defined(TIMESTRAL_CAPI_SHARED)
# ifdef TIMESTRAL_CAPI_BUILD
#  define TIMESTRAL_API __declspec(dllexport)
# else
#  define TIMESTRAL_API __declspec(dllimport)
# endif
#elif defined(__GNUC__)
# define TIMESTRAL_API __attribute__((visibility("default")))
#else
# define TIMESTRAL_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
/* Non-reentrant handles: callers serialize operations, including free.
   Independent handles may run independently. ABI version starts at 1. */
typedef struct timestral_ctx timestral_ctx;
TIMESTRAL_API int timestral_capi_abi_version(void);
TIMESTRAL_API timestral_ctx* timestral_capi_load(const char* gguf_path, int n_threads);
TIMESTRAL_API void timestral_capi_free(timestral_ctx* ctx);
/* Owned text, valid until a modifying call or owner destruction. NULL selects
   thread-local load/null-handle errors. These queries do not clear errors. */
TIMESTRAL_API const char* timestral_capi_last_error(timestral_ctx* ctx);
TIMESTRAL_API int timestral_capi_num_quantiles(timestral_ctx* ctx);
TIMESTRAL_API double timestral_capi_quantile_level(timestral_ctx* ctx, int i);
typedef struct timestral_forecast_request {
    const float* context;           /* B*V*T, NaN = missing */
    const uint8_t* observation_mask;/* B*V*T or NULL; boolean bytes */
    int32_t B, V, T, prediction_length;
    const float* past_cov; int32_t c_past;       /* B*c_past*T */
    const float* known_cov; int32_t c_known;     /* B*c_known*(T+H) */
    const double* cat_past; int32_t cc_past;     /* B*cc_past*T */
    const double* cat_known; int32_t cc_known;   /* B*cc_known*(T+H) */
    int32_t symmetric_averaging; /* -1=model default, 0, 1 */
    int32_t clip_nonnegative;    /* -1=model default, 0, 1 */
    int32_t max_context;         /* 0=model default */
} timestral_forecast_request;
/* Caller buffers must hold the implied lengths. Returns malloc-owned F32
   [B,V,H,Q]; NULL on error. out_q is required, set to zero on failure.
   Free output only with timestral_capi_free_buffer. */
TIMESTRAL_API float* timestral_capi_forecast(timestral_ctx* ctx,
    const timestral_forecast_request* req, int32_t* out_q);
TIMESTRAL_API void timestral_capi_free_buffer(void* p);
#ifdef __cplusplus
}
#endif
#endif
