#include "fixture.hpp"
#include "encoder.hpp"
#include <set>
template<class F> void rejects(F f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}require(rejected,"expected rejection");}
size_t checks=0,layers=0;double maxerr=0,maxratio=0;
double stat(Fixture& f,const std::string& n,const std::string& key){
 auto name="stat."+n+"."+key;auto id=gguf_find_key(f.b.g,name.c_str());require(id>=0,"missing "+name);
 auto ty=gguf_get_kv_type(f.b.g,id);double v;
 if(ty==GGUF_TYPE_UINT32)v=gguf_get_val_u32(f.b.g,id);
 else if(ty==GGUF_TYPE_UINT64)v=double(gguf_get_val_u64(f.b.g,id));
 else v=f.number(name);
 require(std::isfinite(v)&&v>=0,"invalid "+name);return v;
}
void compare(Fixture& f,const std::string& n,const std::vector<float>& x,const char* floor="floor_tf."){
 auto ref=f.f(n);require(stat(f,n,"numel")==ref.size(),"numel "+n);
 double rms=stat(f,n,"rms");double sqref=0;for(float v:ref){require(std::isfinite(v),"nonfinite reference");sqref+=double(v)*v;}
 require(std::abs(rms-std::sqrt(sqref/ref.size()))<=1e-12*std::max(1.,rms),"rms statistic mismatch");
 bool cap_required=n.find(".norm")==std::string::npos && n.find(".mlp_norm")==std::string::npos && n.find(".gated")==std::string::npos;
 if(cap_required){double cap=f.number("cap."+n);require(std::isfinite(cap)&&cap>=0,"invalid cap");}
 double fl=f.number(std::string(floor)+n);require(std::isfinite(fl)&&fl>=0,"invalid floor");
 auto report=tstest::compare_norm(x,ref,fl,f.number("cap."+n,true),n.c_str());require(report.ok,n);
 double e=0,sq=0;for(size_t i=0;i<ref.size();++i){e=std::max(e,std::abs(double(x[i])-ref[i]));sq+=double(ref[i])*ref[i];}
 double err=e/std::max(std::sqrt(sq/ref.size()),double(std::numeric_limits<float>::min()));
 double tol=std::max(2*fl,16*std::ldexp(1.,-24));maxerr=std::max(maxerr,err);maxratio=std::max(maxratio,err/tol);++checks;
}
void units(const Model& m){
 size_t l=0;while(m.config().block_schedule.at(l)!=LayerKind::Time)++l;
 auto D=m.config().d_model;std::vector<float> x(2*D);for(size_t i=0;i<x.size();++i)x[i]=std::sin(float(i));
 Encoder encoder(m);auto run=[&](const TimeMask& mask=TimeMask{}){return encoder.run_layer(l,x,1,2,mask,{0,1});};
 auto a=run();require(a==run(),"repeat replay");TimeMask causal{{1,0,1,1},1};require(a==run(causal),"causal/split equivalence");
 encoder.run_layer(l,std::vector<float>(D,1),1,1,{}, {0});require(a==run(),"grow/shrink replay owned output");
 rejects([&]{replay_time_layer(m,l,{},0,0,{},{});});
 rejects([&]{replay_time_layer(m,m.config().n_layers,x,1,2,{}, {0,1});});
 rejects([&]{replay_time_layer(m,l,x,1,2,{}, {0,INT64_MAX});});
 rejects([&]{run({{1,0,1},0});});rejects([&]{run({{0,0,1,1},0});});rejects([&]{run({{1,2,1,1},0});});rejects([&]{run({{1,1,1,1},1});});
 rejects([&]{replay_time_layer(m,l,x,1,2,{}, {0,1},{{"attn.0.norm",{1}}});});
 rejects([&]{replay_time_layer(m,l,x,1,2,{}, {0,1},{{"unknown",x}});});
 for(size_t i=0;i<m.config().n_layers;++i)if(m.config().block_schedule[i]!=LayerKind::Time)rejects([&]{replay_time_layer(m,i,x,1,2,{}, {0,1});});
 // All-open mask differs from causal on an asymmetric sequence.
 require(run({{1,1,1,1},0}).at("out")!=a.at("out"),"mask negative control");
 auto q=a.at("attn.0.q");auto wrong=q;size_t hd=m.config().head_dim(),H=m.config().num_heads;
 for(size_t t=0;t<2;++t)for(size_t h=0;h<H;++h)for(size_t d=0;d<hd;++d)wrong[t*D+h*hd+d]=q[t*D+d*H+h];
 require(wrong!=q,"head interleave negative control");
 require(!tstest::compare_norm(wrong,q,0,0,"wrong head layout must fail").ok,"layout comparator negative");
 LayerValues wrongq{{"attn.0.q",wrong}};
 require(replay_time_layer(m,l,x,1,2,{}, {0,1},wrongq).at("attn.0.q_rope")!=a.at("attn.0.q_rope"),"interleaved heads affect rotation");
 auto b="blk."+std::to_string(l)+".attn.0.";GraphRunner runner;auto ctx=runner.begin();auto in=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,2,1);
 auto w=m.tensor(b+"qkv.weight");auto saved=w->ne[0];--w->ne[0];rejects([&]{affine(ctx,m,b+"qkv",in,m.config().attn_bias);});w->ne[0]=saved;
 if(m.config().attn_bias){auto bias=m.tensor(b+"qkv.bias");auto n=bias->ne[0];--bias->ne[0];rejects([&]{affine(ctx,m,b+"qkv",in,true);});bias->ne[0]=n;}
 if(m.config().time_gate==Gate::SdpaSigmoid){LayerValues tf{{"attn.0.gated",a.at("attn.0.sdpa")}};require(replay_time_layer(m,l,x,1,2,{}, {0,1},tf).at("attn.0.out")!=a.at("attn.0.out"),"gate-off negative control");}
}
int main(int argc,char** argv){try{
 const char* mr=getenv("TIMESTRAL_TEST_MODEL_ROOT");std::string root=mr?mr:TS_SOURCE_DIR "/tests/fixtures/tiny";
 for(auto v:{"tiny","tiny-bias"}){auto m=Model::load(root+"/timestral-"+v+"-f32.gguf");units(m);}
 if(argc>1&&std::string(argv[1])=="--unit")return 0;
 bool real=argc>1&&std::string(argv[1])=="--real";
 int result=run_component(argc,argv,[&](const Model& m,Fixture& f,const std::string& p){
  std::vector<size_t> selected;
  if(real){require(m.config().n_layers>22&&m.config().block_schedule[0]==LayerKind::Time&&m.config().block_schedule[22]==LayerKind::Time,"real time schedule 0/22");selected={0,22};}
  else for(size_t l=0;l<m.config().n_layers;++l)if(m.config().block_schedule[l]==LayerKind::Time)selected.push_back(l);
  require(!selected.empty(),"no required time layers");
  for(size_t l:selected){std::string b=p+"layer."+std::to_string(l)+".";auto t=f.tensor(b+"in");size_t R=t->ne[2],S=t->ne[1];
   auto ip=f.get<int32_t>(p+"tf.positions",GGML_TYPE_I32);std::vector<int64_t> pos(ip.begin(),ip.end());TimeMask mask;
   if(ggml_get_tensor(f.b.ctx,(p+"mask.time").c_str()))mask.allowed=f.mask(p+"mask.time");
   if(ggml_get_tensor(f.b.ctx,(p+"tf.enc_len").c_str()))mask.enc_len=f.get<int32_t>(p+"tf.enc_len",GGML_TYPE_I32).at(0);
   std::vector<std::string> names={"attn.0.norm","attn.0.qkv","attn.0.q","attn.0.k","attn.0.v","attn.0.q_rope","attn.0.k_rope","attn.0.sdpa","attn.0.gated","attn.0.out","mlp_norm","mlp"};
   if(m.config().time_gate==Gate::SdpaSigmoid)names.push_back("attn.0.gate");
   LayerValues teacher;for(const auto& n:names)if(n!="attn.0.qkv")teacher[n]=f.f(b+n);
   teacher["attn.0.cos"]=f.f(b+"attn.0.cos");teacher["attn.0.sin"]=f.f(b+"attn.0.sin");
   for(auto n:{"q","k","v","q_rope","k_rope","gate","out"})if(std::string(n)!="gate"||m.config().time_gate==Gate::SdpaSigmoid)stat(f,b+"attn.0."+n,"kappa");
   stat(f,b+"attn.0.sdpa","max_logit");stat(f,b+"attn.0.out","rho");stat(f,b+"mlp","kappa");stat(f,b+"mlp","rho");
   Encoder encoder(m);auto tf=encoder.run_layer(l,f.f(b+"in"),R,S,mask,pos,teacher);
   for(const auto& n:names)compare(f,b+n,tf.at(n));
   auto whole=encoder.run_layer(l,f.f(b+"in"),R,S,mask,pos);
   // The layer is teacher-forced on its dumped input, so its local whole-layer floor_tf applies.
   compare(f,b+"out",whole.at("out"));++layers;
  }
 });std::cerr<<"TIME layers="<<layers<<" tensors="<<checks<<" max_norm_error="<<maxerr<<" max_error/tolerance="<<maxratio<<'\n';return result;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return std::string(e.what())=="HOST_SKIP"?77:1;}}
