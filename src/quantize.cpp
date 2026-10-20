#include "quantize.hpp"
#include "quantize_internal.hpp"
#include "model_loader.hpp"
#include "weight_policy.hpp"
#include <openssl/evp.h>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#ifndef __linux__
#error "Atomic quantization currently requires Linux"
#endif
#ifndef TS_QUANTIZE_COMMIT
#error "Explicit quantizer commit required"
#endif
#ifndef TS_QUANTIZE_SOURCE_SHA256
#error "Explicit quantizer source hash required"
#endif
namespace ts {
namespace {
void need(bool ok,const std::string& why){if(!ok)throw Error("quantize: "+why);}
bool hex(const std::string& s,size_t n){return s.size()==n&&s.find_first_not_of("0123456789abcdef")==std::string::npos;}
struct FD {int n=-1;explicit FD(int value=-1):n(value){}~FD(){if(n>=0)::close(n);}FD(const FD&)=delete;};
struct Temp {
 int dir;std::string name;FD fd;
 explicit Temp(int directory):dir(directory){
  static unsigned long serial=0;
  for(int i=0;i<100;++i){name=".timestral-quant-"+std::to_string(getpid())+"-"+std::to_string(__atomic_add_fetch(&serial,1,__ATOMIC_RELAXED));fd.n=openat(dir,name.c_str(),O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);if(fd.n>=0)return;need(errno==EEXIST,"create temporary file");}
  throw Error("quantize: temporary name exhaustion");
 }
 ~Temp(){if(!name.empty())unlinkat(dir,name.c_str(),0);}
 std::string path()const{return "/proc/self/fd/"+std::to_string(fd.n);}
};
struct Digest {
 std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> ctx{EVP_MD_CTX_new(),EVP_MD_CTX_free};
 Digest(){need(ctx&&EVP_DigestInit_ex(ctx.get(),EVP_sha256(),nullptr)==1,"SHA init");}
 void add(const void* p,size_t n){need(EVP_DigestUpdate(ctx.get(),p,n)==1,"SHA update");}
 std::string finish(){std::array<unsigned char,32> b{};unsigned n=0;need(EVP_DigestFinal_ex(ctx.get(),b.data(),&n)==1&&n==32,"SHA final");std::string out;for(auto v:b){out+="0123456789abcdef"[v>>4];out+="0123456789abcdef"[v&15];}return out;}
};
std::string hash_fd(int fd){need(lseek(fd,0,SEEK_SET)==0,"hash seek");Digest d;char b[65536];for(;;){ssize_t n=read(fd,b,sizeof b);if(n<0&&errno==EINTR)continue;need(n>=0,"hash read");if(!n)break;d.add(b,size_t(n));}return d.finish();}
void write_all(int fd,const char* p,size_t n){while(n){ssize_t w=write(fd,p,n);if(w<0&&errno==EINTR)continue;need(w>0,"snapshot write");p+=w;n-=size_t(w);}}
struct GG {gguf_context* g=nullptr;ggml_context* c=nullptr;explicit GG(const std::string& p){g=gguf_init_from_file(p.c_str(),{true,&c});need(g!=nullptr,"read GGUF metadata");}~GG(){if(g)gguf_free(g);if(c)ggml_free(c);}};
struct Out {gguf_context* g=gguf_init_empty();~Out(){gguf_free(g);}};
size_t scalar_size(gguf_type type){switch(type){case GGUF_TYPE_UINT8:case GGUF_TYPE_INT8:case GGUF_TYPE_BOOL:return 1;case GGUF_TYPE_UINT16:case GGUF_TYPE_INT16:return 2;case GGUF_TYPE_UINT32:case GGUF_TYPE_INT32:case GGUF_TYPE_FLOAT32:return 4;case GGUF_TYPE_UINT64:case GGUF_TYPE_INT64:case GGUF_TYPE_FLOAT64:return 8;default:throw Error("quantize: invalid scalar KV type");}}
void compare_kv(const gguf_context* a,const gguf_context* b){
 for(int64_t i=0;i<gguf_get_n_kv(a);++i){const char* key=gguf_get_key(a,i);auto j=gguf_find_key(b,key);need(j>=0,"missing copied KV");auto type=gguf_get_kv_type(a,i);need(type==gguf_get_kv_type(b,j),"copied KV type");
  if(type==GGUF_TYPE_STRING)need(std::strcmp(gguf_get_val_str(a,i),gguf_get_val_str(b,j))==0,"copied string KV");
  else if(type==GGUF_TYPE_ARRAY){auto t=gguf_get_arr_type(a,i);size_t n=gguf_get_arr_n(a,i);need(t==gguf_get_arr_type(b,j)&&n==gguf_get_arr_n(b,j),"copied array KV type/length");if(t==GGUF_TYPE_STRING){for(size_t k=0;k<n;++k)need(std::strcmp(gguf_get_arr_str(a,i,k),gguf_get_arr_str(b,j,k))==0,"copied string array");}else{size_t width=scalar_size(t);need(n<=SIZE_MAX/width,"KV size overflow");if(n)need(!std::memcmp(gguf_get_arr_data(a,i),gguf_get_arr_data(b,j),n*width),"copied numeric array");}}
  else need(!std::memcmp(gguf_get_val_data(a,i),gguf_get_val_data(b,j),scalar_size(type)),"copied scalar KV");
 }
}
bool absent(int dir,const std::string& name){struct stat st{};if(fstatat(dir,name.c_str(),&st,AT_SYMLINK_NOFOLLOW)==0)return false;need(errno==ENOENT,"inspect destination");return true;}
}
namespace quantize_detail {
QuantizeResult run(const std::string& input,const std::string& output,QuantizeType type,const Hooks& hooks){
 need(hex(TS_QUANTIZE_COMMIT,40)&&hex(TS_QUANTIZE_SOURCE_SHA256,64),"invalid build provenance");
 ggml_type target;const char* name;
 switch(type){case QuantizeType::Q6_K:target=GGML_TYPE_Q6_K;name="q6_k";break;case QuantizeType::Q5_K:target=GGML_TYPE_Q5_K;name="q5_k";break;case QuantizeType::Q4_K:target=GGML_TYPE_Q4_K;name="q4_k";break;default:throw Error("quantize: unsupported output type");}
 need(!ggml_quantize_requires_imatrix(target),"importance matrix required");
 std::filesystem::path destination(output);auto leaf=destination.filename().string();need(!leaf.empty()&&leaf!="."&&leaf!="..","invalid destination name");auto parent=destination.parent_path();if(parent.empty())parent=".";
 FD dir(open(parent.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW));need(dir.n>=0,"open destination directory");need(absent(dir.n,leaf),"destination exists; no clobber");
 FD source(open(input.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK));struct stat st{};need(source.n>=0&&fstat(source.n,&st)==0&&S_ISREG(st.st_mode)&&st.st_size>0,"source must be a non-symlink regular file");
 Temp snapshot(dir.n);Digest digest;char chunk[65536];uint64_t copied=0;
 for(;;){ssize_t n=read(source.n,chunk,sizeof chunk);if(n<0&&errno==EINTR)continue;need(n>=0,"source read");if(!n)break;need(copied<=UINT64_MAX-uint64_t(n),"source size overflow");copied+=uint64_t(n);digest.add(chunk,size_t(n));write_all(snapshot.fd.n,chunk,size_t(n));}
 struct stat after{};need(fstat(source.n,&after)==0&&after.st_size==st.st_size&&after.st_mtim.tv_sec==st.st_mtim.tv_sec&&after.st_mtim.tv_nsec==st.st_mtim.tv_nsec&&copied==uint64_t(st.st_size),"source changed during snapshot");
 QuantizeResult result;result.input_bytes=copied;result.input_sha256=digest.finish();need(!hooks.fail_after_snapshot,"injected snapshot failure");
 auto model=Model::load(snapshot.path());GG metadata(snapshot.path());Out out;
 constexpr const char* prefix="timestral.quantization.";
 for(int64_t i=0;i<gguf_get_n_kv(metadata.g);++i)need(std::string(gguf_get_key(metadata.g,i)).rfind(prefix,0)!=0,"existing quantization namespace");
 gguf_set_kv(out.g,metadata.g);
 auto set=[&](const char* key,const std::string& value){gguf_set_val_str(out.g,(std::string(prefix)+key).c_str(),value.c_str());};
 set("version","1");set("type",name);set("policy","b4-k-v1");set("input_gguf_sha256",result.input_sha256);set("tool_version","1");set("runtime_commit",TS_QUANTIZE_COMMIT);set("source_sha256",TS_QUANTIZE_SOURCE_SHA256);set("ggml_commit","e705c5fed490514458bdd2eaddc43bd098fcce9b");
 std::vector<std::vector<unsigned char>> encoded;
 for(const auto& tensor_name:model.tensor_names()){
  auto w=model.tensor(tensor_name);need(w->type==GGML_TYPE_F32,"input tensors must all be F32");
  auto descriptor=*w;descriptor.buffer=nullptr;gguf_add_tensor(out.g,&descriptor);
  const auto* values=static_cast<const float*>(w->data);for(int64_t k=0;k<ggml_nelements(w);++k)need(std::isfinite(values[k]),"nonfinite source tensor");
  if(reduced_weight_name(tensor_name)){
   need(w->ne[0]>0&&w->ne[0]%256==0&&w->ne[2]==1&&w->ne[3]==1,"incompatible K-quant row");
   size_t row=ggml_row_size(target,w->ne[0]);need(uint64_t(w->ne[1])<=SIZE_MAX/row,"encoded size overflow");encoded.emplace_back(row*size_t(w->ne[1]));auto& bytes=encoded.back();
   need(ggml_quantize_chunk(target,values,bytes.data(),0,w->ne[1],w->ne[0],nullptr)==bytes.size(),"quantizer byte count");need(ggml_validate_row_data(target,bytes.data(),bytes.size()),"invalid encoded rows");
   gguf_set_tensor_type(out.g,tensor_name.c_str(),target);gguf_set_tensor_data(out.g,tensor_name.c_str(),bytes.data());++result.converted_tensors;
  }else{gguf_set_tensor_data(out.g,tensor_name.c_str(),w->data);++result.copied_tensors;}
 }
 Temp temporary(dir.n);int copy=dup(temporary.fd.n);need(copy>=0,"duplicate output fd");FILE* stream=fdopen(copy,"wb");if(!stream){close(copy);throw Error("quantize: output stream");}
 bool written=gguf_write_to_file_ptr(out.g,stream,false);bool flushed=fflush(stream)==0;bool synced=fsync(copy)==0;bool closed=fclose(stream)==0;need(written&&flushed&&synced&&closed,"write/flush/sync/close output");need(!hooks.fail_after_write,"injected write failure");
 {
  auto check=Model::load(temporary.path());GG actual(temporary.path());compare_kv(metadata.g,actual.g);need(gguf_get_n_kv(actual.g)==gguf_get_n_kv(metadata.g)+8,"unexpected output KV");compare_kv(out.g,actual.g);
  need(check.tensor_names()==model.tensor_names(),"output tensor order/names");size_t index=0;
  for(const auto& n:model.tensor_names()){auto a=model.tensor(n),b=check.tensor(n);for(int k=0;k<GGML_MAX_DIMS;++k)need(a->ne[k]==b->ne[k],"output shape");bool quant=reduced_weight_name(n);need(b->type==(quant?target:GGML_TYPE_F32),"output tensor type");if(quant){auto& bytes=encoded.at(index++);need(ggml_nbytes(b)==bytes.size()&&!std::memcmp(b->data,bytes.data(),bytes.size()),"encoded output bytes");need(ggml_validate_row_data(target,b->data,ggml_nbytes(b)),"output row validation");}else need(ggml_nbytes(a)==ggml_nbytes(b)&&!std::memcmp(a->data,b->data,ggml_nbytes(a)),"excluded output bytes");}
 }
 struct stat output_stat{};need(fstat(temporary.fd.n,&output_stat)==0&&output_stat.st_size>0,"output size");result.output_bytes=uint64_t(output_stat.st_size);result.output_sha256=hash_fd(temporary.fd.n);
 if(hooks.before_publish)hooks.before_publish();
 bool linked=hooks.force_link_publish;
 if(!linked){long rc=syscall(SYS_renameat2,dir.n,temporary.name.c_str(),dir.n,leaf.c_str(),1u);if(rc==0)temporary.name.clear();else {need(errno==ENOSYS||errno==EINVAL||errno==EOPNOTSUPP,"no-replace publication failed");linked=true;}}
 if(linked){need(linkat(dir.n,temporary.name.c_str(),dir.n,leaf.c_str(),0)==0,"no-replace link publication failed");if(unlinkat(dir.n,temporary.name.c_str(),0)==0)temporary.name.clear();}
 if(hooks.fail_directory_sync||fsync(dir.n)!=0)result.publication=QuantizePublication::PublishedDurabilityUnconfirmed;
 return result;
}
}
QuantizeResult quantize_gguf(const std::string& input,const std::string& output,QuantizeType type){return quantize_detail::run(input,output,type,{});}
}
