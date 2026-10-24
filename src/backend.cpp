#include "backend.hpp"
#include "common.hpp"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <functional>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ts {
namespace { thread_local int execution_threads = 1; thread_local std::map<ggml_context*,GraphRunner*> building_runners; }
void set_cpu_threads(int n) { if(n<1 || n>1024)throw Error("invalid CPU thread count"); execution_threads=n; }
int cpu_threads() { return execution_threads; }

MappedFile::MappedFile(const std::string& path) {
#if !defined(_WIN32)
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw Error("cannot open " + path);
    struct stat st;
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) { ::close(fd); throw Error("cannot stat or empty file: " + path); }
    void* p = ::mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) throw Error("mmap failed for " + path);
    base_ = p;
    size_ = (size_t)st.st_size;
    heap_ = false;
#else
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw Error("cannot open " + path);
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n <= 0) { std::fclose(f); throw Error("empty file: " + path); }
    size_t sz = ((size_t)n + 31) & ~(size_t)31;
    void* p = _aligned_malloc(sz, 64);
    if (!p || std::fread(p, 1, (size_t)n, f) != (size_t)n) { std::fclose(f); throw Error("cannot read " + path); }
    std::fclose(f);
    base_ = p; size_ = (size_t)n; heap_ = true;
#endif
}

void MappedFile::release() {
    if (!base_) return;
#if !defined(_WIN32)
    ::munmap(base_, size_);
#else
    _aligned_free(base_);
#endif
    base_ = nullptr; size_ = 0;
}

MappedFile::MappedFile(MappedFile&& o) noexcept : base_(o.base_), size_(o.size_), heap_(o.heap_) {
    o.base_ = nullptr; o.size_ = 0;
}
MappedFile& MappedFile::operator=(MappedFile&& o) noexcept {
    if (this != &o) { release(); base_ = o.base_; size_ = o.size_; heap_ = o.heap_; o.base_ = nullptr; o.size_ = 0; }
    return *this;
}
MappedFile::~MappedFile() { release(); }

ggml_backend_t backend_init_cpu() {
    ggml_backend_t be = ggml_backend_cpu_init();
    if (!be) throw Error("cannot initialise the CPU backend");
    return be;
}

ggml_backend_buffer_t backend_buffer_from_mapping(MappedFile& file) {
    if (!file.data()) throw Error("no mapping");
    ggml_backend_buffer_t b = ggml_backend_cpu_buffer_from_ptr(const_cast<unsigned char*>(file.data()), file.size());
    if (!b) throw Error("cannot wrap the mapping as a backend buffer");
    return b;
}

} // namespace ts

namespace ts {
GraphRunner::GraphRunner(size_t capacity) : capacity_(capacity) {
    if (capacity < 16 || capacity > 1000000) throw Error("invalid graph capacity");
    backend_ = backend_init_cpu();
    ggml_backend_cpu_set_n_threads(backend_, 1);
    allocator_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (!allocator_) { ggml_backend_free(backend_); throw Error("cannot create graph allocator"); }
}
GraphRunner::~GraphRunner() {
    if(ctx_)building_runners.erase(ctx_);
    if (ctx_) ggml_free(ctx_);
    ggml_gallocr_free(allocator_);
    ggml_backend_free(backend_);
}
void GraphRunner::transition(lifecycle::Event event){auto result=lifecycle::reduce(state_,event);state_=result.state;if(result.effect==lifecycle::Effect::Reject)throw Error(result.rule);}
CallbackContext* register_callback(ggml_context* ctx,const std::vector<ggml_tensor*>& tensors,size_t split){
 auto found=building_runners.find(ctx);auto building_runner=found==building_runners.end()?nullptr:found->second;
 if(!building_runner||building_runner->ctx_!=ctx||!std::holds_alternative<lifecycle::Building>(building_runner->state_.phase))throw Error("callback requires owning GraphRunner");
 auto context=std::make_unique<CallbackContext>();context->status=&building_runner->callback_status_;context->split=split;context->tensors=tensors;
 auto result=context.get();building_runner->callbacks_.push_back(std::move(context));return result;
}
ggml_tensor* graph_constant(ggml_context* ctx,float value){
 auto found=building_runners.find(ctx);auto runner=found==building_runners.end()?nullptr:found->second;
 if(!runner||runner->ctx_!=ctx||!std::holds_alternative<lifecycle::Building>(runner->state_.phase))throw Error("graph constant requires owning GraphRunner");
 for(auto& c:runner->constants_)if(std::memcmp(&c.second,&value,sizeof value)==0)return c.first;
 auto t=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,1);ggml_set_input(t);runner->constants_.emplace_back(t,value);return t;
}
ggml_context* GraphRunner::begin() {
    transition(lifecycle::Event::Begin);constants_.clear();callbacks_.clear();callback_status_.reset();if(ctx_)building_runners.erase(ctx_);
    if (ctx_) ggml_free(ctx_);
    ctx_ = nullptr; graph_ = nullptr; allocated_ = false; computed_ = false;
    ggml_init_params p{capacity_*ggml_tensor_overhead()+ggml_graph_overhead_custom(capacity_,false), nullptr, true};
    ctx_ = ggml_init(p);
    if (!ctx_) {state_=lifecycle::reduce(state_,lifecycle::Event::BeginFail).state;throw Error("cannot allocate graph metadata");}
    try{graph_ = ggml_new_graph_custom(ctx_, capacity_, false);if(detail::consume_callback_fault(detail::CallbackFault::BeginRegistry))throw std::bad_alloc();building_runners[ctx_]=this;}catch(...){state_=lifecycle::reduce(state_,lifecycle::Event::BeginFail).state;building_runners.erase(ctx_);ggml_free(ctx_);ctx_=nullptr;graph_=nullptr;throw;}
    return ctx_;
}
void GraphRunner::allocate(const std::vector<ggml_tensor*>& outputs) {
    computed_=false;
    try {
      if(!std::holds_alternative<lifecycle::Building>(state_.phase)||!graph_||allocated_||outputs.empty())throw Error("graph allocation order");
      std::set<ggml_tensor*> visited,active;
      std::function<void(ggml_tensor*)> validate=[&](ggml_tensor* t){if(!t)return;if(active.count(t))throw Error("cyclic graph");if(!visited.insert(t).second)return;active.insert(t);for(auto& c:callbacks_)if(c->tensors.back()==t)callback_validate_metadata(*c);for(auto src:t->src)validate(src);active.erase(t);};
      for(auto t:outputs){if(!t)throw Error("null graph output");validate(t);}
    }catch(...){state_=lifecycle::reduce(state_,lifecycle::Event::ValidationRejected).state;throw;}
    try {
      for(auto t:outputs){ggml_set_output(t);ggml_build_forward_expand(graph_,t);}
      if(!ggml_gallocr_reserve(allocator_,graph_)||!ggml_gallocr_alloc_graph(allocator_,graph_))throw Error("cannot allocate graph buffers");
      for(auto& c:constants_)if(c.first->buffer)ggml_backend_tensor_set(c.first,&c.second,0,sizeof(float));
      transition(lifecycle::Event::Allocate);allocated_=true;
    }catch(...){state_=lifecycle::reduce(state_,lifecycle::Event::AllocationFail).state;throw;}
}
void GraphRunner::upload(ggml_tensor* t,const std::vector<float>& values) {
    computed_=false;
    try {
      if(!std::holds_alternative<lifecycle::Allocated>(state_.phase)||!allocated_||!t||!(t->flags&GGML_TENSOR_FLAG_INPUT)||t->type!=GGML_TYPE_F32||values.size()>SIZE_MAX/sizeof(float)||ggml_nbytes(t)!=values.size()*sizeof(float))throw Error("graph input size/type/order mismatch");
    }catch(...){state_=lifecycle::reduce(state_,lifecycle::Event::ValidationRejected).state;throw;}
    try {ggml_backend_tensor_set(t,values.data(),0,ggml_nbytes(t));transition(lifecycle::Event::Upload);}
    catch(...){state_=lifecycle::reduce(state_,lifecycle::Event::UploadFail).state;throw;}
}
void GraphRunner::compute() {
    computed_=false;
    try {for(auto& context:callbacks_){bool reachable=false;for(int i=0;i<ggml_graph_n_nodes(graph_);++i)if(ggml_graph_node(graph_,i)==context->tensors.back()){reachable=true;break;}if(reachable)callback_preflight(*context);}}
    catch(...){state_=lifecycle::reduce(state_,lifecycle::Event::PreflightFail).state;throw;}
    transition(lifecycle::Event::Compute);
    try {
    ggml_backend_cpu_set_n_threads(backend_, cpu_threads());
    if (!allocated_ || !graph_ || ggml_backend_graph_compute(backend_, graph_) != GGML_STATUS_SUCCESS || callback_status_.failed()) {
      state_=lifecycle::reduce(state_,lifecycle::Event::WorkerFailure).state;
      throw Error(callback_status_.failed()?callback_status_.message():"graph compute failed");
    }
    transition(lifecycle::Event::Completed);computed_=true;
    }catch(...){state_=lifecycle::reduce(state_,lifecycle::Event::WorkerFailure).state;throw;}
}
std::vector<float> GraphRunner::read(ggml_tensor* t) const {
    if(lifecycle::reduce(state_,lifecycle::Event::Read).effect!=lifecycle::Effect::ReturnOutput)throw Error("graph result unavailable");
    if (!computed_ || !t || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) throw Error("invalid graph read");
    std::vector<float> result(ggml_nelements(t));
    ggml_backend_tensor_get(t, result.data(), 0, ggml_nbytes(t)); return result;
}
}
