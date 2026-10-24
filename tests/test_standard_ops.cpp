// NORM gates for the standard-ggml-op forecast graph (TIMESTRAL_STANDARD_OPS).
//
// The standard-op path is not bit-exact with the CPU custom-op path. It is held
// to the NORM gate of docs/PLAN.md D.1: err <= max(2 * floor, 16u). This test
// runs the ladder components (patch, layer incl. RoPE, encoder, head,
// predict_block, forecast) with the standard-op graph and prints one GATE line
// per comparison. It never stops at the first failure, so the whole table is
// visible. Exit code: 0 all gates pass, 1 any gate fails or errors, 77 skip.
#include "layer_test_helpers.hpp"
#include "head.hpp"
#include "forecaster.hpp"
#include "predict_block.hpp"
#include "predict_metric.hpp"
#include "predict_companions.hpp"
#include "refine.hpp"
#include <cstring>
#include <iostream>
#include <set>
namespace {
size_t gates=0,failed=0,errors=0;double worst_ratio=0;std::string worst_name;
void report(const std::string& comp,const std::string& name,const tstest::Result& r){
 double ratio=r.tol>0?r.err/r.tol:INFINITY;++gates;if(!r.ok)++failed;
 if(ratio>worst_ratio){worst_ratio=ratio;worst_name=comp+" "+name;}
 std::printf("GATE %s %s n=%zu err=%.3e tol=%.3e ratio=%.3f %s\n",comp.c_str(),name.c_str(),r.n,r.err,r.tol,ratio,r.ok?"PASS":"FAIL");std::fflush(stdout);
}
void error(const std::string& comp,const std::string& name,const std::string& what){++errors;std::printf("GATE %s %s ERROR %s\n",comp.c_str(),name.c_str(),what.c_str());std::fflush(stdout);}
bool has(Fixture& f,const std::string& n){return ggml_get_tensor(f.b.ctx,n.c_str());}
// base is "<case>/<pass>"; n already starts with the pass prefix.
std::string cname_label(const std::string& base,const std::string& n){auto slash=base.find('/');return base.substr(0,slash)+"/"+n;}
// Teacher-forced (floor_tf.) or free-running (floor_fr.) NORM gate on a dumped tensor.
void norm_gate(Fixture& f,const std::string& comp,const std::string& base,const std::string& n,const std::vector<float>& x,const char* floor="floor_tf."){
 try{
  auto ref=f.f(n);
  auto r=tstest::compare_norm(x,ref,f.number(std::string(floor)+n),f.number("cap."+n,true),n.c_str());
  report(comp,cname_label(base,n)+(std::string(floor)=="floor_fr."?" [fr]":" [tf]"),r);
 }catch(const std::exception& e){error(comp,base+n,e.what());}
}
ForecastRequest request(Fixture& f,size_t H){
 ForecastRequest r;auto t=f.tensor("input.context");r.batch=t->ne[2];r.targets=t->ne[1];r.length=t->ne[0];r.horizon=H;r.context=f.f("input.context");
 for(auto name:{"past","known","cat_past","cat_known"}){std::string n=std::string("input.")+name;if(!has(f,n))continue;size_t count=f.tensor(n)->ne[1];
  if(n=="input.past"){r.past=f.f(n);r.past_count=count;}if(n=="input.known"){r.known=f.f(n);r.known_count=count;}
  if(n=="input.cat_past"){r.categorical_past=f.get<double>(n,GGML_TYPE_F64);r.categorical_past_count=count;}
  if(n=="input.cat_known"){r.categorical_known=f.get<double>(n,GGML_TYPE_F64);r.categorical_known_count=count;}
 }return r;
}
void predict_gate(const std::string& comp,const std::string& label,const std::vector<float>& actual,Fixture& f,Fixture& f64,Fixture& math,const std::string& gold,const std::string& companion,const std::vector<float>& loc,const std::vector<float>& scale){
 try{
  auto ref=f.f(gold);auto d=f64.get<double>(companion,GGML_TYPE_F64);auto m=math.f(companion);std::vector<double> md(m.begin(),m.end());
  report(comp,label,tstest::compare_predict_normalized_v1(actual,ref,d,md,loc,scale));
 }catch(const std::exception& e){error(comp,label,e.what());}
}
bool want(const std::set<std::string>& sel,const char* c){return sel.empty()||sel.count(c);}

void layer_and_encoder(const Model& m,Fixture& f,const std::string& cname,const std::string& p,const std::set<std::string>& sel){
 const auto& c=m.config();size_t D=c.d_model;
 auto t=f.tensor(p+"tf.in");size_t R=t->ne[2],S=t->ne[1];
 auto ip=f.get<int32_t>(p+"tf.positions",GGML_TYPE_I32);std::vector<int64_t> pos(ip.begin(),ip.end());TimeMask mask;
 if(has(f,p+"mask.time"))mask.allowed=f.mask(p+"mask.time");
 if(has(f,p+"tf.enc_len"))mask.enc_len=f.get<int32_t>(p+"tf.enc_len",GGML_TYPE_I32).at(0);
 std::vector<int8_t> vm;if(has(f,p+"mask.variate"))vm=f.mask(p+"mask.variate");
 std::vector<size_t> selected;
 for(size_t l=0;l<c.n_layers;++l){auto b=p+"layer."+std::to_string(l)+".";if(has(f,b+"in")&&has(f,b+"out"))selected.push_back(l);}
 Encoder e(m);
 if(want(sel,"layer"))for(size_t l:selected){
  auto b=p+"layer."+std::to_string(l)+".";
  try{
   std::vector<std::string> names={"mlp_norm","mlp"};LayerValues teacher;auto axes=layer_axes(c.block_schedule[l]);
   for(size_t j=0;j<axes.size();++j){auto a="attn."+std::to_string(j)+".";bool time=axes[j];
    for(auto n:{"norm","q","k","v","sdpa","gated","out"})names.push_back(a+n);
    if(time)for(auto n:{"q_rope","k_rope"})names.push_back(a+n);
    if((time?c.time_gate:c.variate_gate)==Gate::SdpaSigmoid)names.push_back(a+"gate");
    if(time)for(auto n:{"cos","sin"})teacher[a+n]=f.f(b+a+n);
   }
   for(const auto& n:names)teacher[n]=f.f(b+n);
   auto x=f.f(b+"in");auto tf=e.run_layer(l,x,R,S,mask,pos,teacher,vm);
   for(const auto& n:names)norm_gate(f,"layer",cname+"/"+p,b+n,tf.at(n));
   auto whole=e.run_layer(l,x,R,S,mask,pos,{},vm);norm_gate(f,"layer",cname+"/"+p,b+"out",whole.at("out"));
  }catch(const std::exception& ex){error("layer",cname+"/"+b,ex.what());}
 }
 if(want(sel,"encoder")){
  try{
   auto full=e.run(f.f(p+"tf.in"),R,S,mask,pos,selected,vm);
   for(size_t l:selected){auto b=p+"layer."+std::to_string(l)+".";
    norm_gate(f,"encoder",cname+"/"+p,b+"out",full.at("layer."+std::to_string(l)+".out"),"floor_fr.");}
   norm_gate(f,"encoder",cname+"/"+p,p+"out_norm",full.at("out_norm"),"floor_fr.");
  }catch(const std::exception& ex){error("encoder",cname+"/"+p,ex.what());}
 }
 (void)D;
}
void head_gate(const Model& m,Fixture& f,const std::string& cname,const std::string& p){
 try{
  size_t D=m.config().d_model;auto nt=f.tensor(p+"out_norm"),ht=f.tensor(p+"head.in");size_t R=ht->ne[2],h=ht->ne[1],S=nt->ne[1],L=S-h;
  auto norm=f.f(p+"out_norm");std::vector<float> dec;
  for(size_t r=0;r<R;++r)dec.insert(dec.end(),norm.begin()+(r*S+L)*D,norm.begin()+(r+1)*S*D);
  Head head(m);norm_gate(f,"head",cname+"/"+p,p+"head.raw",head.project(f.f(p+"head.in"),R,h));
 }catch(const std::exception& e){error("head",cname+"/"+p,e.what());}
}
}
int main(){try{
 const char* root=getenv("TIMESTRAL_TEST_REAL_BASELINES");const char* rm=getenv("TIMESTRAL_TEST_GGUF");
 if(!root||!rm){std::cerr<<"SKIP: baseline root or model unset\n";return 77;}
 std::set<std::string> sel;if(const char* c=getenv("TS_GATE_COMPONENTS")){std::istringstream in(c);std::string s;while(std::getline(in,s,','))sel.insert(s);}
 std::string subset=getenv("TIMESTRAL_TEST_REAL_CASES")?getenv("TIMESTRAL_TEST_REAL_CASES"):"all";
 auto specs=manifest(true);std::set<std::string> chosen;
 if(subset=="all")for(auto& s:specs)chosen.insert(s.name);else{std::istringstream in(subset);std::string n;while(std::getline(in,n,','))chosen.insert(n);}
 set_standard_ops(true);
 std::printf("STANDARD_OPS=%d components=%s cases=%s\n",standard_ops()?1:0,getenv("TS_GATE_COMPONENTS")?getenv("TS_GATE_COMPONENTS"):"all",subset.c_str());
 auto m=Model::load(rm);const auto& c=m.config();
 for(auto& spec:specs){if(!chosen.count(spec.name))continue;
  std::string base=std::string(root)+"/real/baseline-"+spec.name+"-";
  if(!std::filesystem::exists(base+"f32.gguf")){std::printf("CASE %s missing baseline\n",spec.name.c_str());++errors;continue;}
  if(!tstest::host_gates_or_skip(base+"f32.gguf")){std::printf("CASE %s host gates skip\n",spec.name.c_str());++errors;continue;}
  std::printf("CASE %s\n",spec.name.c_str());std::fflush(stdout);
  Fixture f(base+"f32.gguf");
  for(auto pass:{"pos","neg"}){
   int64_t patches=(spec.horizon+c.patch_size-1)/c.patch_size;
   int last=int((patches+c.block_size-1)/c.block_size)-1;std::vector<int> ids{0};if(last>0)ids.push_back(last);
   for(int id:ids){std::string p=std::string(pass)+".b"+std::to_string(id)+".";
    if(want(sel,"patch"))try{Encoder e(m);auto t=f.tensor(p+"patch.in");auto a=e.project(f.f(p+"patch.in"),t->ne[2],t->ne[1]);
     norm_gate(f,"patch",spec.name+"/"+p,p+"patch.hidden_out",a.hidden);norm_gate(f,"patch",spec.name+"/"+p,p+"patch.out",a.output);
    }catch(const std::exception& e){error("patch",spec.name+"/"+p,e.what());}
    if(want(sel,"layer")||want(sel,"encoder"))layer_and_encoder(m,f,spec.name,p,sel);
    if(want(sel,"head")&&has(f,p+"head.in"))head_gate(m,f,spec.name,p);
   }
  }
  bool do_pb=want(sel,"predict_block"),do_fc=want(sel,"forecast");
  if(!do_pb&&!do_fc)continue;
  if(!std::filesystem::exists(base+"f64.gguf")){std::printf("CASE %s predict/forecast skipped: no f64 companion\n",spec.name.c_str());continue;}
  try{
   Fixture truth(base+"f64.gguf"),math(base+"f32-math.gguf"),tf64(base+"f64-rerun.gguf"),tfmath(base+"f32-math-rerun.gguf");
   auto r=request(f,spec.horizon);Forecaster fc(m);ForecastTrace trace;
   Forecast out;if(do_fc)out=fc.forecast(r,&trace);
   size_t ps=c.patch_size,Q=c.num_quantiles();
   size_t count=((r.horizon+ps-1)/ps+std::min(c.per_shot_horizon_patches,c.block_size)-1)/std::min(c.per_shot_horizon_patches,c.block_size);
   for(size_t pass=0;pass<2;++pass)for(size_t k=0;k<count;++k){
    std::string p=std::string(pass?"neg":"pos")+".b"+std::to_string(k)+".";
    auto t=f.tensor(p+"args.inputs");size_t R=t->ne[1],T=t->ne[0],h=f.tensor(p+"predict")->ne[1]/ps;
    auto ints=[&](const std::string& n){std::vector<int64_t> v;if(has(f,n)){auto g=f.get<int32_t>(n,GGML_TYPE_I32);v.assign(g.begin(),g.end());}return v;};
    auto group=ints(p+"args.group_ids"),roles=ints(p+"args.role_ids");
    auto loc=f.f(p+"anchor.loc"),scale=f.f(p+"anchor.scale");
    if(do_pb)try{
     auto x=f.f(p+"args.inputs");auto mask=f.mask(p+"args.observed");
     auto pred=PredictBlock(m).predict(x,mask,R,T,h,group,roles,has(f,p+"args.known_future")?f.f(p+"args.known_future"):std::vector<float>{},has(f,p+"args.known_future_mask")?f.mask(p+"args.known_future_mask"):std::vector<int8_t>{});
     bool target_present=has(f,p+"refine.target_rows");auto targets=target_present?f.mask(p+"refine.target_rows"):std::vector<int8_t>{};
     auto refined=refine_block(c,pred.output,x,mask,R,T,h,targets);
     predict_gate("predict_block",spec.name+"/"+p+"predict [tf]",pred.output,f,tf64,tfmath,p+"predict",p+"predict",loc,scale);
     predict_gate("predict_block",spec.name+"/"+p+"refined [tf]",refined,f,tf64,tfmath,p+"refined",p+"block_refined",loc,scale);
    }catch(const std::exception& e){error("predict_block",spec.name+"/"+p,e.what());}
    if(do_fc&&k<(pass?trace.negative:trace.positive).size()){
     auto& block=(pass?trace.negative:trace.positive)[k];
     predict_gate("forecast",spec.name+"/"+p+"predict [fr]",block.predicted,f,truth,math,p+"predict",p+"predict",loc,scale);
     predict_gate("forecast",spec.name+"/"+p+"refined [fr]",block.refined,f,truth,math,p+"refined",p+"refined",loc,scale);
    }
   }
   if(do_fc){
    auto loc=f.f("pos.b0.anchor.loc"),scale=f.f("pos.b0.anchor.scale");size_t all=loc.size()/r.batch;std::vector<float> l,s;
    for(size_t b=0;b<r.batch;++b)for(size_t v=0;v<r.targets;++v){l.push_back(loc[b*all+v]);s.push_back(scale[b*all+v]);}
    predict_gate("forecast",spec.name+"/final.quantiles [fr]",out.quantiles,f,truth,math,"final.quantiles","final.quantiles",l,s);
   }
  }catch(const std::exception& e){error("forecast",spec.name,e.what());}
 }
 std::printf("SUMMARY gates=%zu failed=%zu errors=%zu worst_ratio=%.3f (%s)\n",gates,failed,errors,worst_ratio,worst_name.c_str());
 return (failed||errors||!gates)?1:0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
