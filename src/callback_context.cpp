#include "callback_context.hpp"
#include "common.hpp"
#include "ggml-backend.h"
#include <cstdint>
#include <limits>
#include <set>
#include <algorithm>
namespace ts {
namespace detail {
#ifdef TS_CALLBACK_TESTING
namespace {thread_local CallbackFault fault=CallbackFault::None;}
void inject_callback_fault(CallbackFault f)noexcept{fault=f;}
bool consume_callback_fault(CallbackFault f)noexcept{if(fault!=f)return false;fault=CallbackFault::None;return true;}
#else
bool consume_callback_fault(CallbackFault)noexcept{return false;}
#endif
}
static size_t mul(size_t a,size_t b){if(b&&a>SIZE_MAX/b)throw Error("callback size overflow");return a*b;}
static size_t extent(const ggml_tensor* t){if(!t||t->type!=GGML_TYPE_F32)throw Error("callback tensor type");size_t bytes=4;for(int i=0;i<4;++i){if(t->ne[i]<=0||t->nb[i]!=bytes)throw Error("callback tensor layout");bytes=mul(bytes,size_t(t->ne[i]));}if(bytes>size_t(PTRDIFF_MAX))throw Error("callback tensor byte bounds");return bytes;}
static void span(ggml_tensor*t){
 std::vector<ggml_tensor*> chain;std::set<ggml_tensor*> seen;auto owner=t;
 for(;;){extent(owner);if(!seen.insert(owner).second)throw Error("callback cyclic view");chain.push_back(owner);if(!owner->view_src)break;auto parent=owner->view_src;size_t bytes=extent(owner),parent_bytes=extent(parent);if(owner->view_offs>parent_bytes||bytes>parent_bytes-owner->view_offs)throw Error("callback view extent");owner=parent;}
 auto buffer=owner->buffer;if(!buffer||!ggml_backend_buffer_is_host(buffer))throw Error("callback host storage required");uintptr_t base=reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer));size_t size=ggml_backend_buffer_get_size(buffer);
 for(auto node:chain){size_t bytes=extent(node);uintptr_t data=reinterpret_cast<uintptr_t>(node->data);if(!data||data%alignof(float)||(node->buffer&&node->buffer!=buffer)||data<base||data-base>size||bytes>size-(data-base))throw Error("callback ancestor storage extent");if(node->view_src){uintptr_t parent=reinterpret_cast<uintptr_t>(node->view_src->data);if(node->view_offs>UINTPTR_MAX-parent||parent+node->view_offs!=data)throw Error("callback ancestor data offset");}}
}
void callback_freeze(CallbackContext& c){for(auto t:c.tensors){extent(t);TensorSnapshot s;s.tensor=t;s.type=t->type;s.op=t->op;std::copy_n(t->ne,4,s.ne.begin());std::copy_n(t->nb,4,s.nb.begin());std::copy_n(t->src,GGML_MAX_SRC,s.src.begin());std::copy_n(t->op_params,s.op_params.size(),s.op_params.begin());c.metadata.push_back(s);}}
void callback_validate_metadata(const CallbackContext& context){for(auto& s:context.metadata){if(!s.tensor||s.tensor->type!=s.type||s.tensor->op!=s.op||!std::equal(s.ne.begin(),s.ne.end(),s.tensor->ne)||!std::equal(s.nb.begin(),s.nb.end(),s.tensor->nb)||!std::equal(s.src.begin(),s.src.end(),s.tensor->src)||!std::equal(s.op_params.begin(),s.op_params.end(),s.tensor->op_params))throw Error("callback frozen operation metadata changed");}}
void callback_preflight(CallbackContext& context){context.storage_valid=false;callback_validate_metadata(context);for(auto t:context.tensors)span(t);auto d=context.tensors.back();uintptr_t y=reinterpret_cast<uintptr_t>(d->data);size_t bytes=extent(d);for(size_t i=0;i+1<context.tensors.size();++i){auto t=context.tensors[i];uintptr_t x=reinterpret_cast<uintptr_t>(t->data);size_t n=extent(t);if(y<=x?(x-y<bytes):(y-x<n))throw Error("callback output aliases input");}context.inject_scratch_failure=detail::consume_callback_fault(detail::CallbackFault::ScratchAllocation);context.output_bytes=bytes;context.output_data=d->data;context.storage_valid=true;}
}
