#include "layer_test_helpers.hpp"
size_t selections=0,missing=0,amplified=0;double max_amplification=0;
void units(const Model& m) {
 Encoder e(m);size_t D=m.config().d_model;std::vector<float>x(6*D);
 for(size_t i=0;i<x.size();++i)x[i]=std::sin(float(i)*.17f);
 std::vector<int64_t>pos={0,1,0,1,0,1};std::vector<size_t>all;
 for(size_t l=0;l<m.config().n_layers;++l)all.push_back(l);
 auto a=e.run(x,3,2,{},pos,all);require(a.size()==all.size()+1,"selected outputs");
 auto y=x;for(size_t l:all){y=e.run_layer(l,y,3,2,{},pos).at("out");require(y==a.at("layer."+std::to_string(l)+".out"),"single graph equals sequential layers");}
 require(a==e.run(x,3,2,{},pos,all),"reuse after layer replay");
 auto minimal=e.run(x,3,2,{},pos);require(minimal.size()==1&&minimal.at("out_norm")==a.at("out_norm"),"optional dumps");
 e.run(std::vector<float>(D,1),1,1,{}, {0});require(a==e.run(x,3,2,{},pos,all),"grow shrink reuse");
 rejects([&]{e.run({},3,2,{},pos);});rejects([&]{e.run(x,0,2,{},pos);});
 rejects([&]{e.run(x,3,2,{},{});});rejects([&]{e.run(x,3,2,{}, {0,1,0,1,0,INT64_MAX});});
 rejects([&]{e.run(x,3,2,{},pos,{m.config().n_layers});});rejects([&]{e.run(x,3,2,{},pos,{0,0});});
 rejects([&]{e.run(x,3,2,{{1},0},pos);});rejects([&]{e.run(x,3,2,{std::vector<int8_t>(12,0),0},pos);});
 rejects([&]{e.run(x,3,2,{std::vector<int8_t>(12,2),0},pos);});
 rejects([&]{e.run(x,3,2,{},pos,{}, {1});});
 rejects([&]{e.run(x,3,2,{},pos,{},std::vector<int8_t>(18,0));});
 rejects([&]{e.run(x,3,2,{},pos,{},std::vector<int8_t>(18,2));});
 rejects([&]{e.run_layer(0,x,3,2,{},pos,{{"bogus",{1}}});});
 rejects([&]{e.run_layer(0,x,3,2,{},pos,{{"mlp_norm",{1}}});});
 auto clean=e.run_layer(0,x,3,2,{},pos).at("out");
 auto bad=e.run_layer(0,x,3,2,{},pos,{{"mlp",std::vector<float>(x.size(),10)}}).at("out");
 require(clean!=bad,"teacher negative affects output");require(a==e.run(x,3,2,{},pos,all),"no leaked teacher overrides");
 rejects([&]{GraphRunner r(1000001);});
}
int main(int argc,char**argv){try{
 const char* mr=getenv("TIMESTRAL_TEST_MODEL_ROOT");std::string root=mr?mr:TS_SOURCE_DIR "/tests/fixtures/tiny";
 for(auto v:{"tiny","tiny-bias"})units(Model::load(root+"/timestral-"+v+"-f32.gguf"));
 if(argc>1&&std::string(argv[1])=="--unit")return 0;
 bool selected_only=argc>2&&std::string(argv[2])=="--selected-reference";
 int rc=run_component(argc,argv,[&](const Model&m,Fixture&f,const std::string&p){
  auto t=f.tensor(p+"tf.in");require(t->type==GGML_TYPE_F32&&t->ne[0]==m.config().d_model&&t->ne[3]==1,"encoder embedding dimensions");
  size_t R=t->ne[2],S=t->ne[1];auto ip=f.get<int32_t>(p+"tf.positions",GGML_TYPE_I32);std::vector<int64_t>pos(ip.begin(),ip.end());TimeMask mask;
  if(ggml_get_tensor(f.b.ctx,(p+"mask.time").c_str()))mask.allowed=f.mask(p+"mask.time");
  if(ggml_get_tensor(f.b.ctx,(p+"tf.enc_len").c_str()))mask.enc_len=f.get<int32_t>(p+"tf.enc_len",GGML_TYPE_I32).at(0);
  std::vector<int8_t>vm;if(ggml_get_tensor(f.b.ctx,(p+"mask.variate").c_str()))vm=f.mask(p+"mask.variate");
  std::vector<size_t>selected;
  for(size_t l=0;l<m.config().n_layers;++l){auto b=p+"layer."+std::to_string(l)+".";
   bool present=ggml_get_tensor(f.b.ctx,(b+"in").c_str())&&ggml_get_tensor(f.b.ctx,(b+"out").c_str());
   if(!present){++missing;std::cerr<<"MISSING G3 layer "<<b<<'\n';require(selected_only,"G3.1 requires EVERY layer reference");continue;}selected.push_back(l);
  }
  Encoder e(m);auto full=e.run(f.f(p+"tf.in"),R,S,mask,pos,selected,vm);double prev=0;size_t prior=0,streak=0;
  for(size_t l:selected){auto b=p+"layer."+std::to_string(l)+".";
   auto local=e.run_layer(l,f.f(b+"in"),R,S,mask,pos,{},vm);compare(f,b+"out",local.at("out"));++layers;
   auto&x=full.at("layer."+std::to_string(l)+".out");compare(f,b+"out",x,"floor_fr.");
   auto ref=f.f(b+"out");double err=0;for(size_t i=0;i<x.size();++i)err=std::max(err,std::abs(double(x[i])-ref[i]));err/=std::max(stat(f,b+"out","rms"),double(std::numeric_limits<float>::min()));
   double ratio=prev>0?err/prev:0;bool adjacent=l==prior+1&&prev>0;
   if(adjacent){max_amplification=std::max(max_amplification,ratio);streak=ratio>1.5?streak+1:0;if(streak>=2){++amplified;std::cerr<<"AMPLIFICATION repeated: investigate accumulated rounding in residual stack; tolerance unchanged\n";}}
   std::cerr<<"G3 "<<b<<" err="<<err<<" err_l/err_prev="<<ratio<<(adjacent?" adjacent":" sparse-or-first")<<'\n';prev=err;prior=l;
  }
  compare(f,p+"out_norm",full.at("out_norm"),"floor_fr.");++selections;
 });
 std::cerr<<"ENCODER selections="<<selections<<" local_layers="<<layers<<" comparisons="<<checks<<" max_norm_error="<<maxerr<<" max_error/tolerance="<<maxratio<<" max_adjacent_amplification="<<max_amplification<<" repeated_amplifications="<<amplified<<" missing_layers="<<missing<<'\n';
 if(selected_only)std::cerr<<"SELECTED REFERENCE DIAGNOSTIC ONLY: NOT G3 ACCEPTANCE\n";
 return rc;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return std::string(e.what())=="HOST_SKIP"?77:1;}}
