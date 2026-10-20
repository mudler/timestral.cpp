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
static void rewrite(const std::string& src,const std::string& dst,const std::string& bad="",ggml_type badtype=GGML_TYPE_F16){
 auto model=Model::load(src);ggml_context* ctx=nullptr;auto source=gguf_init_from_file(src.c_str(),{true,&ctx});check(source,"read fixture");
 auto g=gguf_init_empty();gguf_set_kv(g,source);
 std::vector<std::vector<ggml_fp16_t>> data;
 for(auto& name:model.tensor_names()){
  auto w=model.tensor(name);auto descriptor=*w;descriptor.buffer=nullptr;gguf_add_tensor(g,&descriptor);auto type=name==bad?badtype:(bad.empty()&&reduced_weight_name(name)?GGML_TYPE_F16:GGML_TYPE_F32);
  if(type==GGML_TYPE_F32)gguf_set_tensor_data(g,name.c_str(),w->data);
  else {
   // Malformed type fixtures need only enough bytes for their requested row format.
   data.emplace_back((ggml_nbytes(w)+1)/2);
   if(type==GGML_TYPE_F16)ggml_fp32_to_fp16_row((const float*)w->data,data.back().data(),ggml_nelements(w));
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
int main(){try{
 for(auto name:{"patch.out.weight","head.hidden.weight","head.out.weight","head.res.weight","blk.0.attn.0.qkv.weight","blk.1.ffn_w3.weight"})check(weight_type_allowed(name,GGML_TYPE_F16,7),"F16 scalar block");
 for(auto name:{"patch.hidden.weight","patch.res.weight","patch.forecast_query","rope.cos","rope.inv_freq","out_norm.weight","blk.0.attn_norm.0.weight","head.out.bias","blk.x.ffn_w3.weight"})check(!weight_type_allowed(name,GGML_TYPE_F16,32),"forbidden F16 name");
 for(auto type:{GGML_TYPE_BF16,GGML_TYPE_Q8_1,GGML_TYPE_Q2_K,GGML_TYPE_Q3_K})check(!weight_type_allowed("head.out.weight",type,256),"future type enabled early");
 check(!weight_type_allowed("head.out.weight",GGML_TYPE_F16,0),"zero row accepted");
 for(bool bias:{false,true}){
  auto original=Model::load(fixture(bias));projections(original,false);
  auto path=std::string(TS_TEST_TMP_DIR)+(bias?"/tiny-bias-f16.gguf":"/tiny-f16.gguf");rewrite(fixture(bias),path);auto m=Model::load(path);
  for(auto& name:m.tensor_names()){auto a=original.tensor(name),b=m.tensor(name);if(!reduced_weight_name(name))check(a->type==b->type&&!std::memcmp(a->data,b->data,ggml_nbytes(a)),"excluded bytes changed");else{check(b->type==GGML_TYPE_F16,"allowlisted tensor not F16");for(int64_t i=0;i<ggml_nelements(a);++i)check(((ggml_fp16_t*)b->data)[i]==ggml_fp32_to_fp16(((float*)a->data)[i]),"cast byte oracle");}}
  projections(m,true);
  auto bad=std::string(TS_TEST_TMP_DIR)+"/bad-f16.gguf";
  for(auto name:{"patch.hidden.weight","patch.res.weight","rope.cos","out_norm.weight","patch.hidden.bias"}){rewrite(fixture(bias),bad,name);rejects([&]{Model::load(bad);});}
  rewrite(fixture(bias),bad,"head.out.weight",GGML_TYPE_BF16);rejects([&]{Model::load(bad);});
 }
 std::cout<<"PASS F16 policy, loader, storage bytes, projection routes\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
