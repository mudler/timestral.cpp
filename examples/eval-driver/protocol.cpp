#include "protocol.hpp"
#include "weight_policy.hpp"
#include <openssl/evp.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <cerrno>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
namespace tsq {
namespace {
void need(bool b,const char* msg){if(!b)throw std::runtime_error(msg);}
size_t mul(size_t a,size_t b){need(!b||a<=SIZE_MAX/b,"protocol product overflow");return a*b;}
size_t add(size_t a,size_t b){need(a<=SIZE_MAX-b,"protocol sum overflow");return a+b;}
uint64_t get(const std::vector<unsigned char>& b,size_t p,size_t n){need(p<=b.size()&&n<=b.size()-p,"truncated header");uint64_t v=0;for(size_t i=0;i<n;++i)v|=uint64_t(b[p+i])<<(8*i);return v;}
void put(std::vector<unsigned char>& b,size_t p,uint64_t v,size_t n){for(size_t i=0;i<n;++i)b.at(p+i)=unsigned((v>>(8*i))&255);}
double dbl(const std::vector<unsigned char>& b,size_t p){uint64_t v=get(b,p,8);double x;std::memcpy(&x,&v,8);return x;}
Digest digest_at(const std::vector<unsigned char>& b,size_t p){need(p+32<=b.size(),"digest bounds");Digest d;std::copy_n(b.begin()+p,32,d.begin());return d;}
bool safe(const std::string& s){return !s.empty()&&s.size()<=256&&std::all_of(s.begin(),s.end(),[](unsigned char c){return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='-'||c=='.';});}
template<class T> void append(std::vector<unsigned char>& b,const std::vector<T>& a){size_t n=mul(a.size(),sizeof(T));need(add(b.size(),n)<=limit,"result exceeds limit");const auto* p=reinterpret_cast<const unsigned char*>(a.data());if(n)b.insert(b.end(),p,p+n);}
}
Format parse_format(const std::string& value){
 const char* names[]={"f32","f16","q8_0","q6_k","q5_k","q4_k"};for(unsigned i=0;i<6;++i)if(value==names[i])return static_cast<Format>(i);throw std::runtime_error("unsupported driver format");
}
Digest hash_bytes(const void* p,size_t n){Digest d;unsigned int out=0;need(EVP_Digest(p,n,d.data(),&out,EVP_sha256(),nullptr)==1&&out==32,"SHA256 failure");return d;}
Digest hash_file(const std::string& path){std::ifstream f(path,std::ios::binary);need(bool(f),"cannot hash file");std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> c(EVP_MD_CTX_new(),EVP_MD_CTX_free);need(bool(c)&&EVP_DigestInit_ex(c.get(),EVP_sha256(),nullptr)==1,"SHA init");char buf[65536];while(f){f.read(buf,sizeof buf);auto n=f.gcount();if(n)need(EVP_DigestUpdate(c.get(),buf,size_t(n))==1,"SHA update");}need(f.eof(),"hash read failed");Digest d;unsigned int n;need(EVP_DigestFinal_ex(c.get(),d.data(),&n)==1&&n==32,"SHA final");return d;}
Digest unhex(const std::string& s){need(s.size()==64,"SHA length");Digest d{};for(size_t i=0;i<64;++i){char c=s[i];need((c>='0'&&c<='9')||(c>='a'&&c<='f'),"SHA syntax");d[i/2]|=unsigned(c<='9'?c-'0':c-'a'+10)<<(i%2?0:4);}return d;}
std::vector<unsigned char> read_file(const std::string& p){struct stat st{};need(lstat(p.c_str(),&st)==0&&S_ISREG(st.st_mode)&&st.st_size>=0&&uint64_t(st.st_size)<=limit,"input size/type");std::vector<unsigned char> b(size_t(st.st_size));std::ifstream f(p,std::ios::binary);need(bool(f)&&bool(f.read(reinterpret_cast<char*>(b.data()),b.size())),"input read");need(f.peek()==EOF,"input changed length");return b;}
Request decode(const std::vector<unsigned char>& b){
 uint16_t endian=1;need(*reinterpret_cast<unsigned char*>(&endian)==1&&sizeof(float)==4&&sizeof(double)==8&&std::numeric_limits<float>::is_iec559&&std::numeric_limits<double>::is_iec559,"unsupported host");
 need(b.size()>=288&&b.size()<=limit&&!std::memcmp(b.data(),"TSQREQ01",8)&&(get(b,8,4)==1||get(b,8,4)==2)&&get(b,16,8)==b.size(),"request header");
 size_t d[8];for(int i=0;i<8;++i){uint64_t x=get(b,24+i*8,8);need(x<=SIZE_MAX,"dimension range");d[i]=size_t(x);}need(d[0]&&d[1]&&d[2]&&d[3],"zero dimension");
 need(get(b,88,8)==32768&&get(b,96,8)==32&&get(b,104,8)==32&&get(b,112,4)==3&&dbl(b,120)==10000&&dbl(b,128)==1e30&&dbl(b,136)==1e-6&&get(b,144,4)==1&&get(b,148,4)==0,"frozen options/format");
 const std::string hf="9874d252ab0aae16f7f7c999f98f98d6ad7d8365",ref="a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff";
 need(digest_at(b,192)==hash_bytes(hf.data(),hf.size())&&digest_at(b,224)==hash_bytes(ref.data(),ref.size()),"source identities");
 size_t nc=get(b,152,4),header=add(add(288,nc),192);need(nc<=256&&header<=b.size()&&get(b,12,4)==header,"case/header bounds");
 size_t outputs=mul(mul(mul(d[0],d[1]),d[3]),21);
 need(add(mul(outputs,4),add(mul(mul(d[0],d[1]),8),168+512))<=limit,"forecast result exceeds limit");
 auto version=get(b,8,4),format=get(b,116,4),flags=get(b,156,4);
 need(format<=5,"unknown format");if(version==1)need(format==0&&flags==0,"v1 F32 only");else need(flags==(format==0?1u:0u),"invalid anchor capture flags");
 Request out;out.version=uint32_t(version);out.format=static_cast<Format>(format);out.capture_anchors=format==0;out.id=std::string(b.begin()+288,b.begin()+288+nc);need(safe(out.id),"case ID");out.model=digest_at(b,160);out.fixture_input=digest_at(b,256);
 size_t TH=add(d[2],d[3]);size_t counts[]={mul(mul(d[0],d[1]),d[2]),0,mul(mul(d[0],d[4]),d[2]),mul(mul(d[0],d[5]),TH),mul(mul(d[0],d[6]),d[2]),mul(mul(d[0],d[7]),TH)};
 size_t widths[]={4,1,4,4,8,8};unsigned types[]={1,2,1,1,3,3};size_t cursor=header,offsets[6];
 for(int i=0;i<6;++i){size_t p=288+nc+i*32,n=mul(counts[i],widths[i]);need(get(b,p,4)==types[i]&&get(b,p+4,4)==0&&get(b,p+8,8)==counts[i]&&get(b,p+16,8)==n,"array descriptor");offsets[i]=get(b,p+24,8);need(offsets[i]==(n?cursor:0),"array offset/overlap");cursor=add(cursor,n);need(cursor<=b.size(),"array bounds");}need(cursor==b.size(),"trailing bytes");
 auto& r=out.request;r.batch=d[0];r.targets=d[1];r.length=d[2];r.horizon=d[3];r.past_count=d[4];r.known_count=d[5];r.categorical_past_count=d[6];r.categorical_known_count=d[7];r.max_context=32768;r.per_shot_horizon_patches=32;r.quantile_real_cap_k=10000;r.symmetric_averaging=true;r.clip_nonnegative=true;
 auto copy=[&](auto& v,int i){using T=typename std::decay_t<decltype(v)>::value_type;need(counts[i]<=v.max_size(),"vector limit");v.resize(counts[i]);if(counts[i])std::memcpy(v.data(),b.data()+offsets[i],mul(counts[i],sizeof(T)));};copy(r.context,0);copy(r.past,2);copy(r.known,3);copy(r.categorical_past,4);copy(r.categorical_known,5);return out;
}
void validate_storage(const ts::Model& m,Format format){
 const ggml_type types[]={GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_Q8_0,GGML_TYPE_Q6_K,GGML_TYPE_Q5_K,GGML_TYPE_Q4_K};auto index=static_cast<unsigned>(format);need(index<6,"unknown model format");
 for(const auto& n:m.tensor_names()){auto t=m.tensor(n);auto want=ts::reduced_weight_name(n)?types[index]:GGML_TYPE_F32;need(t->type==want&&ts::weight_type_allowed(n,t->type,t->ne[0]),"requested model storage mismatch");}
}
void validate_provenance(const std::string& path,Format format){
 ggml_context* ctx=nullptr;auto g=gguf_init_from_file(path.c_str(),{true,&ctx});need(g!=nullptr,"model provenance read");
 struct Cleanup {gguf_context* g;ggml_context* c;~Cleanup(){gguf_free(g);if(c)ggml_free(c);}} cleanup{g,ctx};
 auto text=[&](const char* key){auto i=gguf_find_key(g,key);need(i>=0&&gguf_get_kv_type(g,i)==GGUF_TYPE_STRING,"required provenance string");return std::string(gguf_get_val_str(g,i));};
 auto ishex=[](const std::string& v,size_t n){return v.size()==n&&v.find_first_not_of("0123456789abcdef")==std::string::npos;};
 auto converted=gguf_find_key(g,"timestral.conversion.outtype");auto quantized=gguf_find_key(g,"timestral.quantization.type");
 if(format==Format::F32){need(converted<0&&quantized<0,"F32 conversion metadata conflict");return;}
 if(format==Format::F16||format==Format::Q8_0){
  const char* name=format==Format::F16?"f16":"q8_0";need(quantized<0&&text("timestral.conversion.outtype")==name,"conversion format metadata");need(text("timestral.conversion.type_policy")==std::string("b4-")+name+"-v1"&&ishex(text("timestral.conversion.converter_sha256"),64),"conversion policy/hash");
 }else{
  const char* names[]={"q6_k","q5_k","q4_k"};need(converted<0&&text("timestral.quantization.type")==names[static_cast<unsigned>(format)-3],"K format metadata");
  need(text("timestral.quantization.version")=="1"&&text("timestral.quantization.policy")=="b4-k-v1"&&text("timestral.quantization.tool_version")=="1","K policy/version");
  need(ishex(text("timestral.quantization.input_gguf_sha256"),64)&&ishex(text("timestral.quantization.source_sha256"),64)&&ishex(text("timestral.quantization.runtime_commit"),40),"K provenance hashes");
  need(text("timestral.quantization.ggml_commit")=="e705c5fed490514458bdd2eaddc43bd098fcce9b","K ggml pin");
 }
}
void validate_model(const ts::Model& m,Format format){const auto& c=m.config();need(c.hf_revision=="9874d252ab0aae16f7f7c999f98f98d6ad7d8365"&&c.code_sha=="a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff","model source pin");need(c.patch_size==32&&c.context_length==32768&&c.per_shot_horizon_patches==32&&c.quantile_real_cap_k==10000&&c.extreme_value_threshold==1e30&&c.minimum_scale==1e-6&&c.use_asinh&&!c.stabilize_with_global&&c.symmetric_averaging&&c.clip_nonnegative,"model frozen options");std::vector<double> levels{.01,.05};for(int i=2;i<=18;++i)levels.push_back(double(i)/20);levels.push_back(.95);levels.push_back(.99);need(c.quantile_knots==levels,"model quantile levels");validate_storage(m,format);}
std::vector<unsigned char> result(const Request& r,const Identity& id,const ts::Forecast* f,const ts::ForecastAnchors* a,const std::string& diagnostic){
 need(r.version==1||r.version==2,"result version");auto format=static_cast<unsigned>(r.format);need(format<=5,"result format");need(r.capture_anchors==(format==0),"result capture flag");need(r.version!=1||format==0,"v1 format");
 std::string msg=diagnostic.substr(0,4096);for(char& c:msg)if(static_cast<unsigned char>(c)>=128)c='?';std::string name=safe(r.id)?r.id:"";
 bool ok=f&&msg.empty();need(ok||(!msg.empty()&&!f&&!a),"failure arrays/diagnostic");need(!ok||bool(a)==r.capture_anchors,"anchor payload presence");
 std::vector<unsigned char> b(256+name.size()+msg.size());std::memcpy(b.data(),"TSQRES01",8);put(b,8,r.version,4);put(b,12,ok?0:1,4);std::copy(id.model.begin(),id.model.end(),b.begin()+56);std::copy(id.executable.begin(),id.executable.end(),b.begin()+88);std::copy(id.source.begin(),id.source.end(),b.begin()+120);
 need(id.commit.size()==40&&std::all_of(id.commit.begin(),id.commit.end(),[](char c){return(c>='0'&&c<='9')||(c>='a'&&c<='f');}),"runtime commit");std::copy(id.commit.begin(),id.commit.end(),b.begin()+152);std::copy(id.input.begin(),id.input.end(),b.begin()+192);put(b,224,name.size(),4);put(b,228,msg.size(),4);put(b,232,b.size(),8);std::copy(name.begin(),name.end(),b.begin()+256);std::copy(msg.begin(),msg.end(),b.begin()+256+name.size());
 if(r.version==2){put(b,240,format,4);put(b,244,ok&&a?1:0,4);}
 if(ok){size_t BV=mul(r.request.batch,r.request.targets),n=mul(mul(BV,r.request.horizon),f->quantile_levels.size());need(f->batch==r.request.batch&&f->targets==r.request.targets&&f->horizon==r.request.horizon&&f->quantiles.size()==n&&!f->quantile_levels.empty(),"result shape");
  for(float x:f->quantiles)need(std::isfinite(x),"nonfinite forecast");for(double x:f->quantile_levels)need(std::isfinite(x),"nonfinite level");
  if(a){need(a->batch==f->batch&&a->targets==f->targets&&a->loc.size()==BV&&a->scale.size()==BV,"anchor shape");for(size_t i=0;i<BV;++i)need(std::isfinite(a->loc[i])&&std::isfinite(a->scale[i])&&a->scale[i]>0,"invalid anchor");}
  put(b,24,f->batch,8);put(b,32,f->targets,8);put(b,40,f->horizon,8);put(b,48,f->quantile_levels.size(),8);append(b,f->quantile_levels);append(b,f->quantiles);if(a){append(b,a->loc);append(b,a->scale);}
 }
 put(b,16,b.size(),8);return b;
}

void publish(const std::string& path,const std::vector<unsigned char>& b){need(b.size()<=limit,"output limit");struct stat st{};need(lstat(path.c_str(),&st)!=0&&errno==ENOENT,"output exists or inaccessible");std::string pattern=path+".tmp.XXXXXX";std::vector<char> tmp(pattern.begin(),pattern.end());tmp.push_back(0);int fd=mkstemp(tmp.data());need(fd>=0,"exclusive temp create");try{size_t off=0;while(off<b.size()){ssize_t n=write(fd,b.data()+off,b.size()-off);need(n>0,"output write");off+=size_t(n);}need(fsync(fd)==0,"output flush");int rc=close(fd);fd=-1;need(rc==0,"output close");need(link(tmp.data(),path.c_str())==0,"refuse existing output/publication failure");unlink(tmp.data());}catch(...){if(fd>=0)close(fd);unlink(tmp.data());throw;}}
}
