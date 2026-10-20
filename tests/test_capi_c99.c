#include "timestral_capi.h"
#include <stddef.h>
size_t capi_c_size(void) { return sizeof(timestral_forecast_request); }
size_t capi_c_offset(int i) {
#define O(x) offsetof(timestral_forecast_request,x)
    const size_t offsets[]={O(context),O(observation_mask),O(B),O(V),O(T),O(prediction_length),O(past_cov),O(c_past),O(known_cov),O(c_known),O(cat_past),O(cc_past),O(cat_known),O(cc_known),O(symmetric_averaging),O(clip_nonnegative),O(max_context)};
    return offsets[i];
}
int capi_c_smoke(void) { timestral_capi_free(0);timestral_capi_free_buffer(0);return timestral_capi_abi_version(); }
