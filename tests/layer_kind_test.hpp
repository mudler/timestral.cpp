#include "layer_test_helpers.hpp"
#include <cstring>
inline bool bitwise(const std::vector<float>& a,const std::vector<float>& b){return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0;}
void kind_units(const Model& m){
 size_t l=0;while(l<m.config().n_layers && m.config().block_schedule[l]!=TEST_KIND)++l;
 require(l<m.config().n_layers,"unit kind absent");
 const size_t D=m.config().d_model;Encoder enc(m);std::vector<float> x(6*D);
 for(size_t i=0;i<x.size();++i)x[i]=std::sin(float(i)*.71f)+float(i/D)*.1f;
 std::vector<int64_t> pos={0,1,0,1,0,1};std::vector<int8_t> mask(18,0);
 for(size_t t=0;t<2;++t)for(size_t r=0;r<3;++r)mask[(t*3+r)*3+r]=1;
 auto run=[&](const std::vector<int8_t>& vm){return enc.run_layer(l,x,3,2,{},pos,{},vm);};
 auto open=run({}),diag=run(mask);require(open.at("out")!=diag.at("out"),"variate mask changes output");
 require(open==run({}),"graph reuse after mask change");
 rejects([&]{run({1});});auto bad=mask;bad[0]=2;rejects([&]{run(bad);});
 bad=mask;bad[0]=0;rejects([&]{run(bad);});
 rejects([&]{enc.run_layer(m.config().n_layers,x,3,2,{},pos);});
 rejects([&]{enc.run_layer(l,x,2,3,{},{});});
 rejects([&]{enc.run_layer(l,x,3,2,{},pos,{{"attn.0.norm",{1}}});});
 rejects([&]{enc.run_layer(l,x,3,2,{},pos,{{"unknown",{1}}});});
 rejects([&]{enc.run_layer(l,x,3,2,{}, {0,1,0,1,0,INT64_MAX});});
 auto axes=layer_axes(TEST_KIND);size_t vj=0;while(axes[vj])++vj;auto a="attn."+std::to_string(vj)+".";
 require(!open.count(a+"q_rope")&&!open.count(a+"cos"),"variate has no RoPE");
 auto wrong=open.at(a+"q"),q=wrong;size_t hd=m.config().head_dim(),H=m.config().num_heads;
 for(size_t t=0;t<6;++t)for(size_t h=0;h<H;++h)for(size_t d=0;d<hd;++d)wrong[t*D+h*hd+d]=q[t*D+d*H+h];
 require(!tstest::compare_norm(wrong,q,0,0,"wrong variate head layout").ok,"wrong layout comparator");
 require(enc.run_layer(l,x,3,2,{},pos,{{a+"q",wrong}}).at(a+"sdpa")!=open.at(a+"sdpa"),"wrong heads affect attention");
 // Swapping slot masks is a valid shape but wrong indexing: only slot 0 permits all keys.
 auto slots=mask;for(size_t i=0;i<9;++i)slots[i]=1;auto swapped=slots;
 std::rotate(swapped.begin(),swapped.begin()+9,swapped.end());
 require(run(slots).at("out")!=run(swapped).at("out"),"per-patch mask indexing negative");
 std::vector<float> one(x.begin(),x.begin()+2*D);
 auto singleton=enc.run_layer(l,one,1,2,{}, {0,1});
 require(bitwise(singleton.at(a+"v"),singleton.at(a+"sdpa")),"R=1 full attention bitwise v");
 require(open==run({}),"grow/shrink owned result");
 GraphRunner runner;auto ctx=runner.begin();auto t=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,hd,H,3,2);
 auto maskt=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,3,3,2);
 auto wrongtype=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,hd,H,3,2);
 rejects([&]{attention_variate(ctx,wrongtype,wrongtype,wrongtype,maskt);});
 rejects([&]{attention_variate(ctx,t,t,t,ggml_new_tensor_3d(ctx,GGML_TYPE_F32,2,2,2));});
 rejects([&]{attention_variate(ctx,ggml_permute(ctx,t,0,2,1,3),t,t,maskt);});
 if(m.config().variate_gate==Gate::SdpaSigmoid){auto off=open.at(a+"sdpa");std::vector<float> unpermuted(off.size());
  for(size_t r=0;r<3;++r)for(size_t s=0;s<2;++s)for(size_t d=0;d<D;++d)unpermuted[(r*2+s)*D+d]=off[(s*3+r)*D+d];
  require(enc.run_layer(l,x,3,2,{},pos,{{a+"gated",unpermuted}}).at(a+"out")!=open.at(a+"out"),"variate gate-off negative");
 }
}

int main(int argc,char** argv){try{
 const char* mr=getenv("TIMESTRAL_TEST_MODEL_ROOT");std::string root=mr?mr:TS_SOURCE_DIR "/tests/fixtures/tiny";
 for(auto v:{"tiny","tiny-bias"})kind_units(Model::load(root+"/timestral-"+v+"-f32.gguf"));
 if(argc>1&&std::string(argv[1])=="--unit")return 0;
 bool real=argc>1&&std::string(argv[1])=="--real";
 int result=run_component(argc,argv,[&](const Model& m,Fixture& f,const std::string& p){
  std::vector<size_t> selected;
  if(real){require(TEST_KIND==LayerKind::TimeVariate,"only time_variate exists in real model");require(m.config().block_schedule.at(3)==TEST_KIND,"real layer 3 kind");selected={3};}
  else for(size_t l=0;l<m.config().n_layers;++l)if(m.config().block_schedule[l]==TEST_KIND)selected.push_back(l);
  require(!selected.empty(),"no required layer kind");
  for(size_t l:selected){std::string b=p+"layer."+std::to_string(l)+".";auto t=f.tensor(b+"in");require(t->type==GGML_TYPE_F32&&t->ne[0]==m.config().d_model&&t->ne[3]==1,"layer input layout");size_t R=t->ne[2],S=t->ne[1];
   auto ip=f.get<int32_t>(p+"tf.positions",GGML_TYPE_I32);std::vector<int64_t> pos(ip.begin(),ip.end());TimeMask mask;
   if(ggml_get_tensor(f.b.ctx,(p+"mask.time").c_str()))mask.allowed=f.mask(p+"mask.time");
   if(ggml_get_tensor(f.b.ctx,(p+"tf.enc_len").c_str()))mask.enc_len=f.get<int32_t>(p+"tf.enc_len",GGML_TYPE_I32).at(0);
   std::vector<int8_t> vm;if(ggml_get_tensor(f.b.ctx,(p+"mask.variate").c_str()))vm=f.mask(p+"mask.variate");
   std::vector<std::string> names={"mlp_norm","mlp"};LayerValues teacher;auto axes=layer_axes(TEST_KIND);
   for(size_t j=0;j<axes.size();++j){auto a="attn."+std::to_string(j)+".";bool time=axes[j];
    for(auto n:{"norm","qkv","q","k","v","sdpa","gated","out"})names.push_back(a+n);
    if(time)for(auto n:{"q_rope","k_rope"})names.push_back(a+n);
    bool gate=(time?m.config().time_gate:m.config().variate_gate)==Gate::SdpaSigmoid;if(gate)names.push_back(a+"gate");
    for(auto n:{"q","k","v","out"})stat(f,b+a+n,"kappa");
    if(time)for(auto n:{"q_rope","k_rope"})stat(f,b+a+n,"kappa");
    if(gate)stat(f,b+a+"gate","kappa");stat(f,b+a+"sdpa","max_logit");stat(f,b+a+"out","rho");
    if(time)for(auto n:{"cos","sin"})teacher[a+n]=f.f(b+a+n);
   }
   for(const auto& n:names)if(n.find("qkv")==std::string::npos)teacher[n]=f.f(b+n);
   stat(f,b+"mlp","kappa");stat(f,b+"mlp","rho");Encoder encoder(m);auto x=f.f(b+"in");
   auto tf=encoder.run_layer(l,x,R,S,mask,pos,teacher,vm);for(const auto& n:names)compare(f,b+n,tf.at(n));
   if(R==1)for(size_t j=0;j<axes.size();++j)if(!axes[j]){auto a="attn."+std::to_string(j)+".";
    require(bitwise(f.f(b+a+"v"),f.f(b+a+"sdpa")),"pinned R=1 reference bitwise invariant");
   }
   auto whole=encoder.run_layer(l,x,R,S,mask,pos,{},vm);compare(f,b+"out",whole.at("out"));++layers;
   if(R==1)for(size_t j=0;j<axes.size();++j)if(!axes[j]){auto a="attn."+std::to_string(j)+".";
    require(bitwise(whole.at(a+"v"),whole.at(a+"sdpa")),"native R=1 full path invariant");
   }
   require(whole==encoder.run_layer(l,x,R,S,mask,pos,{},vm),"graph reuse");
  }
 });std::cerr<<"KIND layers="<<layers<<" tensors="<<checks<<" max_norm_error="<<maxerr<<" max_error/tolerance="<<maxratio<<'\n';return result;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return std::string(e.what())=="HOST_SKIP"?77:1;}}
