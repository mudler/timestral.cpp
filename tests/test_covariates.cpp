#include "covariates.hpp"
#include "numpy_sum.hpp"
#include "fixture.hpp"
#include <limits>
template<class F> void rejects(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}require(bad,"invalid covariate input accepted");}
void units(){
 require(!tstest::compare_exact_f32({1},{2},"negative G5 exact").ok,"nonvacuous exact");
 std::vector<double> cancel(256,0);for(size_t i=0;i<256;i+=8){cancel[i]=1e16;cancel[i+1]=1;cancel[i+2]=-1e16;cancel[i+3]=1;}
 require(numpy_sum(cancel)==0,"recursive pairwise split");
 auto missing=target_encode({NAN,NAN},{NAN,NAN},{1,NAN});require(missing.past==std::vector<float>({0,0})&&missing.future==std::vector<float>({0,0}),"empty categories and observations");
 auto e=target_encode({2,4,NAN},{16777216.,16777217.,NAN},{16777216.,16777217.,99,NAN});
 require(e.past==std::vector<float>({2.5f,3.5f,3})&&e.future==std::vector<float>({2.5f,3.5f,3,3}),"f64 codes/unseen/nan");
 require(numpy_sum(std::vector<double>{1e16,1,-1e16,1,1,1,1,1})==4,"numpy eight lanes");
 Config c{};c.target_encoding_smooth=1;c.patch_size=2;c.context_length=4;c.covariate_enabled=true;c.block_schedule={LayerKind::Time,LayerKind::TimeVariate};
 c.target_encoding_smooth=3;
 auto weighted=assemble_rows(c,{2,4},{1,1},1,1,2,1,{},0,{},0,0,{10,20},1);
 require(weighted.rows==std::vector<float>({2,4,2.75f,3.25f}),"metadata smoothing reaches categorical assembly");
 for(double invalid:{0.,-1.,double(NAN),double(INFINITY)}){c.target_encoding_smooth=invalid;rejects([&]{assemble_rows(c,{2,4},{1,1},1,1,2,1,{},0,{},0,0,{10,20},1);});rejects([&]{target_encode({2,4},{10,20},{},invalid);});}
 c.target_encoding_smooth=1;
 auto a=assemble_rows(c,{NAN,2,NAN,4,5},{1,1,1,1,1},1,1,5,3,{1,NAN,3,INFINITY,5},1,{0,1,2,3,4,5,6,NAN},1);
 require(a.rows==std::vector<float>({2,2,4,5,0,3,std::numeric_limits<float>::max(),5,1,2,3,4}),"cap after ffill and finite saturation");
 require(a.roles==std::vector<int64_t>({0,1,2})&&a.groups==std::vector<int64_t>({0,0,0}),"roles/groups");
 require(a.future==std::vector<float>({0,0,0,0,0,0,0,0,5,6,0,0}),"future padding");
 auto padded=assemble_rows(c,{NAN,2,NAN},{1,1,0},1,1,3,1,{},0,{},0);
 require(padded.rows==std::vector<float>({0,0,2,2})&&padded.mask==std::vector<int8_t>({0,0,1,0}),"sanitize, user mask, and left padding");
 rejects([&]{assemble_rows(c,{1,2},{1,2},1,1,2,1,{},0,{},0);});
 rejects([&]{assemble_rows(c,{1,2},{1,1},1,1,2,SIZE_MAX,{},0,{},0);});
 rejects([&]{target_encode({1},{INFINITY},{});});rejects([&]{target_encode({1},{1,2},{});});
 rejects([&]{assemble_rows(c,{}, {},1,1,0,1,{},0,{},0);});
 rejects([&]{assemble_rows(c,{}, {},SIZE_MAX,2,2,1,{},0,{},0);});
 c.block_schedule={LayerKind::Time};
 try{assemble_rows(c,{1,2},{1,1},1,1,2,1,{},0,{1,2,3},1);throw std::runtime_error("schedule accepted");}catch(const ts::Error& x){require(std::string(x.what())=="this checkpoint cannot condition on known-future covariates","reference schedule error");}
}

size_t comparisons=0,encodings=0;
void exact(const std::vector<float>& a,const std::vector<float>& b,const std::string& label){
 require(tstest::compare_exact_f32(a,b,label.c_str()).ok,label);++comparisons;
}
void check(const Model& m,Fixture& f,const std::string& p){
 if(p!="pos.b0.")return;
 auto t=f.tensor("input.context");size_t B=t->ne[2],V=t->ne[1],T=t->ne[0];
 auto has=[&](const std::string& n){return ggml_get_tensor(f.b.ctx,n.c_str())!=nullptr;};
 auto floats=[&](const std::string& n){return has(n)?f.f(n):std::vector<float>{};};
 auto doubles=[&](const std::string& n){return has(n)?f.get<double>(n,GGML_TYPE_F64):std::vector<double>{};};
 auto count=[&](const std::string& n){return has(n)?size_t(f.tensor(n)->ne[1]):0;};
 auto id=gguf_find_key(f.b.g,"case.H");require(id>=0,"case.H");size_t H=gguf_get_val_u32(f.b.g,id);
 for(auto pass:{"pos.","neg."}){
  auto x=f.f("input.context"),past=floats("input.past"),known=floats("input.known");auto cp=doubles("input.cat_past"),ck=doubles("input.cat_known");
  // Negation after encoding is linear in the target. Raw multivariate codes
  // are also negated by the forecaster, so assemble positive then mirror.
  auto a=assemble_rows(m.config(),x,{},B,V,T,H,past,count("input.past"),known,count("input.known"),0,cp,count("input.cat_past"),ck,count("input.cat_known"));
  if(std::string(pass)=="neg."){for(auto& v:a.rows)v=-v;for(auto& v:a.future)v=-v;}
  exact(a.rows,f.f(std::string(pass)+"assemble.rows"),std::string(pass)+"G5.2 rows");
  require(a.mask==f.mask(std::string(pass)+"assemble.mask"),"assembled mask");++comparisons;
  if(!a.roles.empty()){auto roles=f.get<int32_t>(std::string(pass)+"assemble.role_ids",GGML_TYPE_I32);require(std::vector<int64_t>(roles.begin(),roles.end())==a.roles,"roles");++comparisons;}
  if(!a.future.empty()){exact(a.future,f.f(std::string(pass)+"assemble.known_future"),"known future");require(a.future_mask==f.mask(std::string(pass)+"assemble.known_future_mask"),"future mask");++comparisons;}
  auto g=f.get<int32_t>(std::string(pass)+"b0.args.group_ids",GGML_TYPE_I32);require(std::vector<int64_t>(g.begin(),g.end())==a.groups,"groups");++comparisons;
 }
 for(size_t i=0;has("target_encode."+std::to_string(i)+".ctx");++i){
  std::string q="target_encode."+std::to_string(i)+".";
  auto e=target_encode(doubles(q+"ctx"),doubles(q+"past"),doubles(q+"future"));
  exact(e.past,f.f(q+"out_past"),"G5.3 past");if(has(q+"out_future"))exact(e.future,f.f(q+"out_future"),"G5.3 future");++encodings;
 }
}
int main(int argc,char**argv){try{units();if(argc>1&&std::string(argv[1])=="--unit")return 0;int status=run_component(argc,argv,check);std::cerr<<"COV comparisons="<<comparisons<<" encodings="<<encodings<<" ULP=0\n";return status;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
