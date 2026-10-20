#include "weight_policy.hpp"
#include "model_loader.hpp"
#include "layers.hpp"
#include "head.hpp"
#include <cstring>
#include <cmath>
namespace ts { ggml_tensor* detail_head_projection(ggml_context*,const Model&,const char*,ggml_tensor*,bool); }
#include <fstream>
#include <iostream>
#include <vector>
using namespace ts;
static void check(bool ok,const char* why){if(!ok)throw Error(why);}
template<class F>static void rejects(F f){bool bad=false;try{f();}catch(const Error&){bad=true;}check(bad,"expected rejection");}
static std::string fixture(bool bias){return std::string(TS_SOURCE_DIR)+"/tests/fixtures/tiny/timestral-tiny"+(bias?"-bias":"")+"-f32.gguf";}
// Rewrite random test weights only. Hold converted bytes through the GGUF write.
static void rewrite(const std::string& src,const std::string& dst,const std::string& bad="",ggml_type badtype=GGML_TYPE_Q8_0){
 auto model=Model::load(src);ggml_context* ctx=nullptr;auto source=gguf_init_from_file(src.c_str(),{true,&ctx});check(source,"read fixture");
 auto g=gguf_init_empty();gguf_set_kv(g,source);
 std::vector<std::vector<uint8_t>> data;
 for(auto& name:model.tensor_names()){
  auto w=model.tensor(name);auto descriptor=*w;descriptor.buffer=nullptr;gguf_add_tensor(g,&descriptor);auto type=name==bad?badtype:(bad.empty()&&reduced_weight_name(name)?GGML_TYPE_Q8_0:GGML_TYPE_F32);
  if(type==GGML_TYPE_F32)gguf_set_tensor_data(g,name.c_str(),w->data);
  else {
   // Malformed type fixtures need only enough bytes for their requested row format.
   data.emplace_back(ggml_nbytes(w));
   if(type==GGML_TYPE_Q8_0)check(ggml_quantize_chunk(type,(const float*)w->data,data.back().data(),0,w->ne[1],w->ne[0],nullptr)==ggml_row_size(type,w->ne[0])*w->ne[1],"quantized bytes");
   gguf_set_tensor_type(g,name.c_str(),type);gguf_set_tensor_data(g,name.c_str(),data.back().data());
  }
 }
 check(gguf_write_to_file(g,dst.c_str(),false),"write fixture");gguf_free(g);gguf_free(source);ggml_free(ctx);
}
static ggml_tensor* manual(ggml_context* ctx,ggml_tensor* w,ggml_tensor* x,ggml_tensor* b=nullptr){auto y=ggml_mul_mat(ctx,w,x);return b?ggml_add(ctx,y,b):y;}
static void same(const std::vector<float>& a,const std::vector<float>& b){check(a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*4),"GGML oracle bytes differ");}
static void projections(const Model& model,bool reduced){
 const auto& c=model.config();size_t D=c.d_model,R=2,S=3;
 // General affine including explicit bias and multiple rows/tokens.
 {
  GraphRunner run;auto ctx=run.begin();auto x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,4*D,S,R);ggml_set_input(x);
  auto got=affine(ctx,model,"patch.out",x,true);
  check(got->op==(reduced?GGML_OP_ADD:GGML_OP_CUSTOM),"affine route");
  if(reduced){auto want=manual(ctx,model.tensor("patch.out.weight"),x,model.tensor("patch.out.bias"));run.allocate({got,want});run.upload(x,std::vector<float>(4*D*S*R,.10003f));run.compute();same(run.read(got),run.read(want));}
 }
 // Residual entry: preserve F32 hidden/residual callbacks; only patch.out changes.
 {
  GraphRunner run;auto ctx=run.begin();auto x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,2*c.patch_size,S,R);ggml_set_input(x);
  auto got=residual_block(ctx,model,x);
  check(got.hidden->op==GGML_OP_CUSTOM,"hidden must retain F32 callback");
  check(got.output->src[0]->op==(reduced?GGML_OP_ADD:GGML_OP_CUSTOM),"residual route");
  if(reduced){auto projection=manual(ctx,model.tensor("patch.out.weight"),ggml_silu(ctx,got.hidden),model.tensor("patch.out.bias"));auto want=ggml_add(ctx,projection,got.output->src[1]);run.allocate({got.output,want});run.upload(x,std::vector<float>(2*c.patch_size*S*R,.010003f));run.compute();same(run.read(got.output),run.read(want));}
 }
 // Direct small-head dispatch oracle, including optional bias on patch.out.
 for(auto base:{"head.hidden","head.out","head.res","patch.out"}){
  auto w=model.tensor(std::string(base)+".weight");bool bias=std::string(base)=="patch.out";
  GraphRunner run;auto ctx=run.begin();auto x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,w->ne[0],S,R);ggml_set_input(x);
  auto got=detail_head_projection(ctx,model,base,x,bias);
  check(got->op==(reduced?(bias?GGML_OP_ADD:GGML_OP_MUL_MAT):GGML_OP_CUSTOM),"head route");
  if(reduced){auto want=manual(ctx,w,x,bias?model.tensor("patch.out.bias"):nullptr);run.allocate({got,want});run.upload(x,std::vector<float>(w->ne[0]*S*R,.10003f));run.compute();same(run.read(got),run.read(want));}
  else if(w->ne[0]<192){std::vector<float> values(w->ne[0]*S*R,.10003f),expected(w->ne[1]*S*R);auto weights=(const float*)w->data;auto b=bias?(const float*)model.tensor("patch.out.bias")->data:nullptr;
   for(size_t row=0;row<S*R;++row)for(int64_t col=0;col<w->ne[1];++col){float sum=0;for(int64_t k=0;k<w->ne[0];++k)sum=std::fma(weights[col*w->ne[0]+k],values[row*w->ne[0]+k],sum);expected[row*w->ne[1]+col]=b?sum+b[col]:sum;}
   run.allocate({got});run.upload(x,values);run.compute();same(run.read(got),expected);}
 }

}
int main(int argc,char** argv){try{
 if(argc==3 && std::string(argv[1])=="--encode-blocks"){
  std::vector<float> x(3*64);for(size_t row=0;row<3;++row)for(size_t k=0;k<64;++k)x[row*64+k]=row==0?0.f:row==1?(k==31||k==63?127.f:float(int(k%31)-15)+.5f):float(int(k)-32)*.125f;
  std::vector<uint8_t> bytes(3*2*34);check(ggml_quantize_chunk(GGML_TYPE_Q8_0,x.data(),bytes.data(),0,3,64,nullptr)==bytes.size(),"block oracle length");
  std::ofstream out(argv[2],std::ios::binary);out.write((const char*)bytes.data(),bytes.size());check(bool(out),"write block oracle");return 0;
 }

 for(auto name:{"patch.hidden.weight","patch.res.weight","patch.forecast_query","rope.cos","rope.inv_freq","out_norm.weight","head.out.bias"})check(!weight_type_allowed(name,GGML_TYPE_Q8_0,64),"Q8 exclusion");
 for(int width:{32,64,96})check(weight_type_allowed("head.out.weight",GGML_TYPE_Q8_0,width),"Q8 row accepted");
 for(int width:{0,1,31,33,63})check(!weight_type_allowed("head.out.weight",GGML_TYPE_Q8_0,width),"Q8 row rejected");
 for(auto type:{GGML_TYPE_BF16,GGML_TYPE_Q8_1,GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K})check(!weight_type_allowed("head.out.weight",type,256),"future type rejected");
 for(bool bias:{false,true}){
  auto original=Model::load(fixture(bias));projections(original,false);
  auto path=std::string(TS_TEST_TMP_DIR)+(bias?"/tiny-bias-q8.gguf":"/tiny-q8.gguf");rewrite(fixture(bias),path);auto m=Model::load(path);
  for(auto& name:m.tensor_names()){
   auto a=original.tensor(name),b=m.tensor(name);
   if(!reduced_weight_name(name))check(a->type==b->type&&!std::memcmp(a->data,b->data,ggml_nbytes(a)),"excluded bytes changed");
   else {check(b->type==GGML_TYPE_Q8_0,"allowlisted tensor not Q8");std::vector<uint8_t> bytes(ggml_nbytes(b));ggml_quantize_chunk(GGML_TYPE_Q8_0,(const float*)a->data,bytes.data(),0,a->ne[1],a->ne[0],nullptr);check(!std::memcmp(bytes.data(),b->data,bytes.size()),"quantized storage oracle");}
  }
  projections(m,true);
  auto bad=std::string(TS_TEST_TMP_DIR)+"/bad-q8.gguf";
  // These malformed rows are block-compatible; rejection must be policy-based.
  for(auto name:{"out_norm.weight","patch.hidden.bias"}){rewrite(fixture(bias),bad,name);rejects([&]{Model::load(bad);});}
  for(auto type:{GGML_TYPE_BF16,GGML_TYPE_Q8_1}){rewrite(fixture(bias),bad,"head.out.weight",type);rejects([&]{Model::load(bad);});}
 }
 std::cout<<"PASS Q8 policy, loader, storage bytes, all projection routes\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
