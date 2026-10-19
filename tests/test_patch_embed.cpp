#include "fixture.hpp"
#include "encoder.hpp"
#include <iostream>
template<class F> void rejects(F f) {bool bad=false;try{f();}catch(const std::exception&){bad=true;}require(bad,"expected rejection");}
void units(const Model& m) {
 Encoder e(m); size_t ps=m.config().patch_size;
 std::vector<float> x(ps);for(size_t i=0;i<ps;++i)x[i]=float(i+1);
 std::vector<int8_t> mask(ps,1);mask[0]=0;
 auto a=e.embed(x,mask,1,ps);require(a.input[0]==1 && a.input[ps]==1 && a.input[ps+1]==0,"values then missing flags");
 auto b=e.embed(x,mask,1,ps);require(a.output==b.output,"repeat graph reuse");
 auto big=e.embed(std::vector<float>(4*ps,1),std::vector<int8_t>(4*ps,1),2,2*ps);
 require(big.output.size()==4*m.config().d_model,"shape growth");
 require(e.embed(x,mask,1,ps).output==a.output,"shrink preserves owned output");
 rejects([&]{e.embed({1},{1},1,1);});rejects([&]{e.embed({}, {},0,0);});
 mask[0]=2;rejects([&]{e.embed(x,mask,1,ps);});
 rejects([&]{e.project(x,1,1);});
 GraphRunner runner; auto ctx=runner.begin();auto bad=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,3);
 rejects([&]{residual_block(ctx,m,bad);});
 auto bias=m.tensor("patch.out.bias"); auto saved=bias->ne[0]; bias->ne[0]--;
 auto in=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,2*ps,1);
 rejects([&]{residual_block(ctx,m,in);});bias->ne[0]=saved;
}
int main(int argc,char** argv) {try {
 const char* mr=getenv("TIMESTRAL_TEST_MODEL_ROOT");auto m=Model::load(std::string(mr?mr:TS_SOURCE_DIR "/tests/fixtures/tiny")+"/timestral-tiny-f32.gguf"); units(m);
 if(argc>1 && std::string(argv[1])=="--unit")return 0;
 return run_component(argc,argv,[](const Model& m,Fixture& f,const std::string& p){
   Encoder e(m);auto t=f.tensor(p+"patch.in");auto a=e.project(f.f(p+"patch.in"),t->ne[2],t->ne[1]);
   auto known=ggml_get_tensor(f.b.ctx,(p+"known.rows").c_str());
   auto scaled=p+(m.config().covariate_enabled && known?"known.scaled":"scaled");
   auto v=f.tensor(scaled); auto boundary=e.embed(f.f(scaled),f.mask(p+"scale.observed"),v->ne[1],v->ne[0]);
   require(tstest::compare_exact_f32(boundary.input,f.f(p+"patch.in"),"patch feature boundary").ok,"patch feature boundary");
   for(auto name:{"hidden_out","out"}) {
      std::string n=p+"patch."+name;
      require(tstest::compare_norm(std::string(name)=="out"?a.output:a.hidden,f.f(n),f.number("floor_tf."+n),f.number("cap."+n,true),n.c_str()).ok,n);

   }
 });
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return std::string(e.what())=="HOST_SKIP"?77:1;}}
