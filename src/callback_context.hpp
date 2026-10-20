#pragma once
#include "ggml.h"
#include <atomic>
#include <array>
#include <vector>
namespace ts {
namespace detail { enum class CallbackFault { None, ScratchAllocation, BeginRegistry };
#ifdef TS_CALLBACK_TESTING
void inject_callback_fault(CallbackFault) noexcept;
#endif
bool consume_callback_fault(CallbackFault) noexcept; }
enum class Publication { Empty, Reserved, Published };
struct CallbackStatus {
 std::atomic<Publication> slot{Publication::Empty};std::array<char,192> text{};
 void publish(const char* message) noexcept {auto expected=Publication::Empty;if(!slot.compare_exchange_strong(expected,Publication::Reserved,std::memory_order_relaxed))return;size_t i=0;for(;i+1<text.size()&&message[i];++i)text[i]=message[i];text[i]=0;slot.store(Publication::Published,std::memory_order_release);}
 bool failed()const noexcept{return slot.load(std::memory_order_acquire)!=Publication::Empty;}
 const char* message()const noexcept {return slot.load(std::memory_order_acquire)==Publication::Published?text.data():"callback error publication incomplete";}
 void reset()noexcept{text[0]=0;slot.store(Publication::Empty,std::memory_order_relaxed);}
};
struct TensorSnapshot {ggml_tensor* tensor=nullptr;ggml_type type=GGML_TYPE_F32;ggml_op op=GGML_OP_NONE;std::array<int64_t,4> ne{};std::array<size_t,4> nb{};std::array<ggml_tensor*,GGML_MAX_SRC> src{};std::array<int32_t,GGML_MAX_OP_PARAMS/sizeof(int32_t)> op_params{};};
struct CallbackContext {CallbackStatus* status=nullptr;size_t split=0;std::vector<ggml_tensor*> tensors;std::vector<TensorSnapshot> metadata;size_t output_bytes=0;void* output_data=nullptr;bool inject_scratch_failure=false;bool storage_valid=false;};
// Graph construction is synchronous on the owning calling thread.
CallbackContext* register_callback(ggml_context*,const std::vector<ggml_tensor*>&,size_t split=0);
void callback_freeze(CallbackContext&);
void callback_validate_metadata(const CallbackContext&);
void callback_preflight(CallbackContext&);
}
