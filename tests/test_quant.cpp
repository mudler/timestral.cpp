#include "quantize.hpp"
#include "quantize_internal.hpp"
#include "model_loader.hpp"
#include "weight_policy.hpp"
#include "layers.hpp"
#include <openssl/evp.h>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <unistd.h>
using namespace ts;
namespace fs=std::filesystem;
namespace ts {ggml_tensor* detail_head_projection(ggml_context*,const Model&,const char*,ggml_tensor*,bool);}
static void check(bool ok,const char* why){if(!ok)throw Error(why);}
template<class F>void rejects(F f){bool rejected=false;try{f();}catch(const Error&){rejected=true;}check(rejected,"expected refusal");}
static std::vector<char> read(const fs::path& p){std::ifstream f(p,std::ios::binary);return std::vector<char>(std::istreambuf_iterator<char>(f),{});}
static std::string hash(const fs::path& p){auto b=read(p);unsigned char d[32];unsigned n=0;check(EVP_Digest(b.data(),b.size(),d,&n,EVP_sha256(),nullptr)==1&&n==32,"hash oracle");std::string s;for(auto v:d){s+="0123456789abcdef"[v>>4];s+="0123456789abcdef"[v&15];}return s;}
static void fixture(const fs::path& output,bool bias,bool conflict=false,bool nonfinite=false,ggml_type reduced=GGML_TYPE_F32){
 auto path=std::string(TS_SOURCE_DIR)+"/tests/fixtures/tiny/timestral-tiny"+(bias?"-bias":"")+"-f32.gguf";
 ggml_context* original=nullptr;auto g=gguf_init_from_file(path.c_str(),{true,&original});check(g,"fixture metadata");
 gguf_set_val_u32(g,"timestral.embedding_length",256);gguf_set_val_u32(g,"timestral.feed_forward_length",256);gguf_set_val_u32(g,"timestral.attention.head_count",4);gguf_set_val_u32(g,"timestral.rope.dimension_count",32);gguf_set_val_u32(g,"timestral.context_length",64);gguf_set_val_i32(g,"timestral.rope.table_pos_min",-12);
 auto c=read_config(g);auto out=gguf_init_empty();gguf_set_kv(out,g);
 gguf_set_val_u8(out,"custom.u8",255);gguf_set_val_i8(out,"custom.i8",-7);gguf_set_val_u16(out,"custom.u16",60000);gguf_set_val_i16(out,"custom.i16",-30000);gguf_set_val_u32(out,"custom.u32",4000000000u);gguf_set_val_i32(out,"custom.i32",-123456);gguf_set_val_u64(out,"custom.u64",UINT64_MAX);gguf_set_val_i64(out,"custom.i64",INT64_MIN);gguf_set_val_f32(out,"custom.f32",-0.f);gguf_set_val_f64(out,"custom.f64",1.125);gguf_set_val_bool(out,"custom.bool",true);gguf_set_val_str(out,"custom.string","preserve me");
 const char* words[]={"one","two"};gguf_set_arr_str(out,"custom.strings",words,2);double numbers[]={-0.,1.25};gguf_set_arr_data(out,"custom.numbers",GGUF_TYPE_FLOAT64,numbers,2);
 if(conflict)gguf_set_val_str(out,"timestral.quantization.type","prior");
 auto ctx=ggml_init({16u<<20,nullptr,false});check(ctx,"fixture scratch");
 std::vector<std::vector<unsigned char>> encoded;
 for(auto& spec:expected_schema(c)){
  auto t=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,spec.ne[0],spec.ne[1],spec.ne[2],spec.ne[3]);ggml_set_name(t,spec.name.c_str());auto p=(float*)t->data;
  for(int64_t i=0;i<ggml_nelements(t);++i)p[i]=float(int(i%127)-63)/4096.f;
  if(spec.name.find("norm")!=std::string::npos)for(int64_t i=0;i<ggml_nelements(t);++i)p[i]=1.f;
  if(spec.name=="rope.inv_freq")for(uint32_t i=0;i<c.head_dim()/2;++i)p[i]=i<c.rope_dim/2?1.f/std::pow(float(c.rope_theta),float(2*i)/float(c.rope_dim)):0.f;
  if(spec.name=="rope.cos"||spec.name=="rope.sin")for(int64_t pos=0;pos<c.rope_table_len();++pos)for(uint32_t d=0;d<c.head_dim();++d){uint32_t index=d%(c.head_dim()/2);float freq=index<c.rope_dim/2?1.f/std::pow(float(c.rope_theta),float(2*index)/float(c.rope_dim)):0.f;float angle=float(pos+c.rope_table_pos_min)*freq;p[pos*c.head_dim()+d]=spec.name=="rope.cos"?std::cos(angle):std::sin(angle);}
  if(nonfinite&&spec.name=="head.out.weight")p[0]=INFINITY;
  gguf_add_tensor(out,t);
  if(reduced!=GGML_TYPE_F32&&reduced_weight_name(spec.name)){
   size_t size=ggml_row_size(reduced,t->ne[0])*size_t(t->ne[1]);encoded.emplace_back(size);auto& bytes=encoded.back();check(ggml_quantize_chunk(reduced,p,bytes.data(),0,t->ne[1],t->ne[0],nullptr)==size,"fixture quantization");gguf_set_tensor_type(out,spec.name.c_str(),reduced);gguf_set_tensor_data(out,spec.name.c_str(),bytes.data());
  }
 }
 check(gguf_write_to_file(out,output.c_str(),false),"fixture write");gguf_free(out);ggml_free(ctx);gguf_free(g);ggml_free(original);
}
static void projection(const Model& model){
 for(auto base:{"head.hidden","head.out","head.res","patch.out"}){
  auto w=model.tensor(std::string(base)+".weight");bool bias=std::string(base)=="patch.out";GraphRunner run;auto ctx=run.begin();auto x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,w->ne[0],3,2);ggml_set_input(x);
  auto y=detail_head_projection(ctx,model,base,x,bias);auto oracle=ggml_mul_mat(ctx,w,x);if(bias)oracle=ggml_add(ctx,oracle,model.tensor("patch.out.bias"));run.allocate({y,oracle});run.upload(x,std::vector<float>(w->ne[0]*6,.12345f));run.compute();check(run.read(y)==run.read(oracle),"head GGML oracle");
 }
 GraphRunner run;auto ctx=run.begin();auto x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,16,3,2);ggml_set_input(x);auto y=residual_block(ctx,model,x);auto projection=ggml_mul_mat(ctx,model.tensor("patch.out.weight"),ggml_silu(ctx,y.hidden));projection=ggml_add(ctx,projection,model.tensor("patch.out.bias"));auto oracle=ggml_add(ctx,projection,y.output->src[1]);run.allocate({y.output,oracle});run.upload(x,std::vector<float>(96,.01f));run.compute();check(run.read(y.output)==run.read(oracle),"residual oracle");
 GraphRunner affine_run;ctx=affine_run.begin();auto a=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,256,3,2);ggml_set_input(a);auto got=affine(ctx,model,"blk.0.attn.0.qkv",a,model.config().attn_bias);auto want=ggml_mul_mat(ctx,model.tensor("blk.0.attn.0.qkv.weight"),a);if(model.config().attn_bias)want=ggml_add(ctx,want,model.tensor("blk.0.attn.0.qkv.bias"));affine_run.allocate({got,want});affine_run.upload(a,std::vector<float>(1536,.01f));affine_run.compute();check(affine_run.read(got)==affine_run.read(want),"affine oracle");
}
int main(){try{
 for(auto type:{GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K}){
  for(int width:{256,512,768})check(weight_type_allowed("head.out.weight",type,width),"K compatible row");
  for(int width:{0,16,32,192,255,257})check(!weight_type_allowed("head.out.weight",type,width),"K incompatible row");
  for(auto name:{"patch.hidden.weight","patch.res.weight","out_norm.weight","rope.cos","head.out.bias"})check(!weight_type_allowed(name,type,256),"K forbidden tensor");
 }
 fs::path dir=fs::path(TS_TEST_TMP_DIR)/"quant-transaction";fs::remove_all(dir);fs::create_directories(dir);auto input=dir/"input.gguf";
 auto clean=[&]{for(auto& e:fs::directory_iterator(dir))check(e.path().filename().string().rfind(".timestral-quant-",0)!=0,"temporary leak");};
 for(bool bias:{false,true})for(auto type:{QuantizeType::Q6_K,QuantizeType::Q5_K,QuantizeType::Q4_K}){
  fixture(input,bias);auto output=dir/"output.gguf";fs::remove(output);auto before=read(input);auto result=quantize_gguf(input,output,type);check(result.input_sha256==hash(input)&&result.output_sha256==hash(output),"actual hash oracle");check(result.publication==QuantizePublication::Durable&&result.converted_tensors>0&&result.copied_tensors>0,"result accounting");check(before==read(input),"input changed");auto source=Model::load(input),out=Model::load(output);
  for(auto& name:source.tensor_names()){auto a=source.tensor(name),b=out.tensor(name);if(!reduced_weight_name(name))check(!std::memcmp(a->data,b->data,ggml_nbytes(a)),"excluded copy");else{std::vector<unsigned char> encoded(ggml_nbytes(b));check(ggml_quantize_chunk(b->type,(float*)a->data,encoded.data(),0,a->ne[1],a->ne[0],nullptr)==encoded.size(),"reference quant bytes");check(!std::memcmp(encoded.data(),b->data,encoded.size()),"reference quant oracle");}}
  projection(out);auto saved=read(output);rejects([&]{quantize_gguf(input,output,type);});check(saved==read(output),"existing destination changed");clean();
 }
 fixture(input,false);auto output=dir/"failure.gguf";
 for(int stage:{0,1,2}){quantize_detail::Hooks h;h.fail_after_snapshot=stage==0;h.fail_after_write=stage==1;if(stage==2)h.before_publish=[](){throw Error("injected before publish");};rejects([&]{quantize_detail::run(input,output,QuantizeType::Q4_K,h);});check(!fs::exists(output),"failure published");clean();}
 quantize_detail::Hooks race;race.before_publish=[&]{std::ofstream(output)<<"race-winner";};rejects([&]{quantize_detail::run(input,output,QuantizeType::Q4_K,race);});check(read(output)==std::vector<char>({'r','a','c','e','-','w','i','n','n','e','r'}),"race destination clobbered");fs::remove(output);clean();
 quantize_detail::Hooks durable;durable.fail_directory_sync=true;durable.force_link_publish=true;auto uncertain=quantize_detail::run(input,output,QuantizeType::Q4_K,durable);check(fs::exists(output)&&uncertain.publication==QuantizePublication::PublishedDurabilityUnconfirmed,"publication distinction");fs::remove(output);clean();
 auto link=dir/"link.gguf";fs::create_symlink(input.filename(),link);rejects([&]{quantize_gguf(link,output,QuantizeType::Q4_K);});fs::remove(link);fs::create_hard_link(input,output);rejects([&]{quantize_gguf(input,output,QuantizeType::Q4_K);});fs::remove(output);fs::create_symlink(input.filename(),output);rejects([&]{quantize_gguf(input,output,QuantizeType::Q4_K);});fs::remove(output);rejects([&]{quantize_gguf(input,input,QuantizeType::Q4_K);});
 fixture(input,false,true);rejects([&]{quantize_gguf(input,output,QuantizeType::Q4_K);});fixture(input,false,false,true);rejects([&]{quantize_gguf(input,output,QuantizeType::Q4_K);});
 for(auto storage:{GGML_TYPE_F16,GGML_TYPE_Q8_0,GGML_TYPE_Q4_K}){fixture(input,false,false,false,storage);rejects([&]{quantize_gguf(input,output,QuantizeType::Q4_K);});}
 fixture(input,false);fs::resize_file(input,fs::file_size(input)-100);rejects([&]{quantize_gguf(input,output,QuantizeType::Q4_K);});rejects([&]{quantize_gguf(std::string(TS_SOURCE_DIR)+"/tests/fixtures/tiny/timestral-tiny-f32.gguf",output,QuantizeType::Q4_K);});clean();
 std::cout<<"PASS synthetic K quantization, all routes, provenance, atomic failures/races\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
