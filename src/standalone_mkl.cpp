#include "standalone_mkl.hpp"
#include "standalone_mkl_algorithm.hpp"
#include "callback_context.hpp"
#include "common.hpp"
#include <set>
#include <climits>
namespace ts {
std::string standalone_mkl_identity(){standalone_mkl_init();MKLVersion v;mkl_get_version(&v);return std::string("standalone-mkl-lp64-sequential-")+std::to_string(v.MajorVersion)+"."+std::to_string(v.MinorVersion)+"."+std::to_string(v.UpdateVersion)+"-"+v.Build;}
void standalone_mkl_init(){MKLVersion v;mkl_get_version(&v);if(v.MajorVersion!=2024||v.MinorVersion!=0||v.UpdateVersion!=2||std::strcmp(v.Build,"20240605"))throw Error("unsupported standalone MKL version: expected 2024.0 update2 build20240605");}
namespace {
size_t extent(ggml_tensor*t){if(!t||t->type!=GGML_TYPE_F32)throw Error("MKL tensor type");size_t n=4;for(int i=0;i<4;++i){if(t->ne[i]<=0||t->nb[i]!=n)throw Error("MKL tensor layout");n=standalone_algorithm::multiply(n,size_t(t->ne[i]));}if(n>size_t(PTRDIFF_MAX))throw Error("MKL tensor byte bounds");return n;}
void disjoint(ggml_tensor*d){uintptr_t y=reinterpret_cast<uintptr_t>(d->data);size_t bytes=extent(d);for(int i=0;i<4;++i)if(d->src[i]){auto t=d->src[i];uintptr_t x=reinterpret_cast<uintptr_t>(t->data);size_t n=extent(t);if(y<=x?(x-y<bytes):(y-x<n))throw Error("MKL output aliases input");}}
void patch_shape(ggml_tensor*w,ggml_tensor*x,ggml_tensor*b,ggml_tensor*d=nullptr){extent(w);extent(x);if(w->ne[2]!=1||w->ne[3]!=1||x->ne[3]!=1||w->ne[0]!=x->ne[0])throw Error("MKL patch rank/K mismatch");for(auto a:{w->ne[0],w->ne[1],x->ne[1],x->ne[2]})if(a>INT_MAX)throw Error("MKL patch LP64 bounds");size_t rows=standalone_algorithm::multiply(size_t(x->ne[1]),size_t(x->ne[2]));if(rows>INT_MAX)throw Error("MKL patch row bounds");if(b){extent(b);if(b->ne[0]!=w->ne[1]||b->ne[1]!=1||b->ne[2]!=1||b->ne[3]!=1)throw Error("MKL patch bias shape");}if(d){extent(d);if(d->ne[0]!=w->ne[1]||d->ne[1]!=x->ne[1]||d->ne[2]!=x->ne[2]||d->ne[3]!=1)throw Error("MKL patch output shape");}}
void patch_callback(ggml_tensor*d,int ith,int nth,void*userdata)noexcept{
 auto c=static_cast<CallbackContext*>(userdata);if(!c)return;
 try{if(!d||ith<0||nth<=0||ith>=nth||!c->storage_valid)throw Error("MKL patch callback boundary");if(ith)return;
 patch_shape(d->src[0],d->src[1],d->src[2],d);disjoint(d);if(c->status->failed()){std::memset(c->output_data,0,c->output_bytes);return;}
 if(c->inject_scratch_failure)throw std::bad_alloc();
 auto w=d->src[0],x=d->src[1];int K=int(w->ne[0]),N=int(w->ne[1]),M=int(x->ne[1]*x->ne[2]);standalone_algorithm::Aligned output(size_t(N)*size_t(M));if(d->src[2])for(int r=0;r<M;++r)std::copy_n((const float*)d->src[2]->data,N,output.data+size_t(r)*N);
 cblas_sgemm(CblasColMajor,CblasTrans,CblasNoTrans,N,M,K,1,(const float*)w->data,K,(const float*)x->data,K,d->src[2]?1.f:0.f,output.data,N);output.check();std::memcpy(d->data,output.data,size_t(N)*M*4);
 }catch(...){c->status->publish("standalone MKL patch execution failed");if(c->storage_valid&&d&&ith==0)std::memset(c->output_data,0,c->output_bytes);}
}
void sdpa_callback(ggml_tensor*d,int ith,int nth,void*userdata)noexcept{
 auto c=static_cast<CallbackContext*>(userdata);if(!c)return;
 // A single worker owns this node, retaining every accepted whole GEMM.
 if(ith<0||nth<=0||ith>=nth){c->status->publish("MKL SDPA worker bounds");return;}
 if(ith!=0)return;
 try{if(!d||nth<=0||!c->storage_valid)throw Error("MKL SDPA callback boundary");disjoint(d);if(c->status->failed()){std::memset(c->output_data,0,c->output_bytes);return;}
 auto mask=d->src[3];for(size_t i=0;i<extent(mask)/4;++i){float v=((float*)mask->data)[i];if(std::isnan(v)||v==INFINITY)throw Error("MKL SDPA mask payload");}
 if(c->inject_scratch_failure)throw std::bad_alloc();
 standalone_algorithm::sdpa(d,0,1,reinterpret_cast<void*>(c->split));
 }catch(...){c->status->publish("standalone MKL SDPA execution failed");if(c->storage_valid&&d)std::memset(c->output_data,0,c->output_bytes);}
}
}
ggml_tensor* standalone_patch(ggml_context*ctx,ggml_tensor*w,ggml_tensor*x,ggml_tensor*b){standalone_mkl_init();patch_shape(w,x,b);ggml_tensor*args[]={w,x,b};auto c=register_callback(ctx,b?std::vector<ggml_tensor*>{w,x,b}:std::vector<ggml_tensor*>{w,x});auto d=ggml_custom_4d(ctx,GGML_TYPE_F32,w->ne[1],x->ne[1],x->ne[2],1,args,b?3:2,patch_callback,1,c);c->tensors.push_back(d);callback_freeze(*c);return d;}
ggml_tensor* standalone_sdpa(ggml_context*ctx,ggml_tensor*q,ggml_tensor*k,ggml_tensor*v,ggml_tensor*m,size_t split){standalone_mkl_init();for(auto t:{q,k,v,m})extent(t);if(!ggml_are_same_shape(q,k)||!ggml_are_same_shape(q,v)||m->ne[0]!=q->ne[2]||m->ne[1]!=q->ne[2]||m->ne[2]!=q->ne[3]||m->ne[3]!=1||split>size_t(q->ne[2])||q->ne[0]>INT_MAX||q->ne[2]>INT_MAX||q->ne[3]>INT_MAX||q->ne[1]>INT_MAX/q->ne[0])throw Error("MKL SDPA shape");auto c=register_callback(ctx,{q,k,v,m},split);ggml_tensor*args[]={q,k,v,m};auto d=ggml_custom_4d(ctx,GGML_TYPE_F32,q->ne[0]*q->ne[1],q->ne[2],q->ne[3],1,args,4,sdpa_callback,1,c);c->tensors.push_back(d);callback_freeze(*c);return d;}
}
