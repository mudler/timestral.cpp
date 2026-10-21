#include "predict_block.hpp"
#include "fixture.hpp"
#include "predict_metric.hpp"
#include "predict_companions.hpp"
#include <chrono>
template<class F> void rejects(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}require(bad,"invalid prediction accepted");}
void units(){auto m=Model::load((getenv("TIMESTRAL_TEST_MODEL_ROOT")?std::string(getenv("TIMESTRAL_TEST_MODEL_ROOT")):std::string(TS_SOURCE_DIR)+"/tests/fixtures/tiny")+"/timestral-tiny-f32.gguf");PredictBlock b(m);size_t ps=m.config().patch_size;
 rejects([&]{b.predict(std::vector<float>(ps,NAN),std::vector<int8_t>(ps,1),1,ps,1,{0});});
 rejects([&]{b.predict({}, {},0,ps,1,{});});rejects([&]{b.predict({}, {},SIZE_MAX,ps,1,{});});
 rejects([&]{b.predict(std::vector<float>(ps),std::vector<int8_t>(ps,1),1,ps,1,{0},{7});});
 rejects([&]{b.predict(std::vector<float>(ps),std::vector<int8_t>(ps,1),1,ps,1,{0},{2},{1},{1});});
}
size_t selections=0,cov_selections=0;double max_ratio=0;

void check(const Model& m,Fixture& f,Fixture& truth,Fixture& math,const std::string& p){
 auto t=f.tensor(p+"args.inputs");size_t R=t->ne[1],T=t->ne[0],ps=m.config().patch_size,Q=m.config().num_quantiles();
 auto has=[&](const std::string& n){return ggml_get_tensor(f.b.ctx,(p+n).c_str())!=nullptr;};
 auto ints=[&](const std::string& n){std::vector<int64_t> out;if(has(n)){auto v=f.get<int32_t>(p+n,GGML_TYPE_I32);out.assign(v.begin(),v.end());}return out;};
 size_t h=f.tensor(p+"predict")->ne[1]/ps;
 auto out=PredictBlock(m).predict(f.f(p+"args.inputs"),f.mask(p+"args.observed"),R,T,h,ints("args.group_ids"),ints("args.role_ids"),has("args.known_future")?f.f(p+"args.known_future"):std::vector<float>{},has("args.known_future_mask")?f.mask(p+"args.known_future_mask"):std::vector<int8_t>{});
 // Explicit auditable F32 frame, including post-replacement known row frames.
 auto loc=f.f(p+"anchor.loc"),scale=f.f(p+"anchor.scale");
 require(loc.size()==R&&scale.size()==R,"anchor shape");
 // Known statistics retain the approved G1.3 reduction allowance, not exactness.
 std::cerr.precision(9);for(size_t r=0;r<R;++r)std::cerr<<"FRAME row="<<r<<" f32_loc="<<loc[r]<<" f32_scale="<<scale[r]<<" native_loc="<<out.loc[r]<<" native_scale="<<out.scale[r]<<'\n';
 auto y=predict_companion_values(f,p+"predict",GGML_TYPE_F32,R,h*ps,Q);
 auto y64=predict_companion_values(truth,p+"predict",GGML_TYPE_F64,R,h*ps,Q),ym=predict_companion_values(math,p+"predict",GGML_TYPE_F32,R,h*ps,Q);
 std::cerr<<"G5_NATIVE_COMPARISON "<<p<<"\n";
 auto result=tstest::compare_predict_normalized_v1(out.output,f.f(p+"predict"),y64,ym,loc,scale);
 require(result.ok,"G5.4/G4.3 native prediction");
 auto corrupt=out.output;corrupt[0]+=float(100*scale[0]);
 std::cerr<<"G5_NEGATIVE_CONTROL_EXPECT_FAIL "<<p<<"\n";
 require(!tstest::compare_predict_normalized_v1(corrupt,f.f(p+"predict"),y64,ym,loc,scale).ok,"negative prediction comparator");
 std::cerr<<"G5_NEGATIVE_CONTROL_REJECTED "<<p<<"\n";
 max_ratio=std::max(max_ratio,result.err/result.tol);++selections;if(has("args.role_ids"))++cov_selections;
}
int main(int argc,char**argv){try{
 units();if(argc>1&&std::string(argv[1])=="--unit")return 0;
 bool real=argc>1&&std::string(argv[1])=="--real";
 const char* root=getenv(real?"TIMESTRAL_TEST_REAL_BASELINES":"TIMESTRAL_TEST_TINY");if(!root){std::cerr<<"SKIP: baseline root unset\n";return 77;}
 const char* mr=getenv("TIMESTRAL_TEST_MODEL_ROOT");std::string models=mr?mr:TS_SOURCE_DIR "/tests/fixtures/tiny";
 std::string subset=getenv("TIMESTRAL_TEST_REAL_CASES")?getenv("TIMESTRAL_TEST_REAL_CASES"):"all";
 auto specs=manifest(real);std::set<std::string> selected;
 if(!real||subset=="all")for(auto& spec:specs)selected.insert(spec.name);
 else {require(!subset.empty()&&subset.back()!=',',"empty selection");std::istringstream in(subset);std::string n;while(std::getline(in,n,',')){require(std::any_of(specs.begin(),specs.end(),[&](const Case& c){return c.name==n;}),"unknown case "+n);require(selected.insert(n).second,"duplicate case");}}
 size_t cases=0,failed=0;
 for(auto variant:real?std::vector<std::string>{"real"}:std::vector<std::string>{"tiny","tiny-bias"}){
  const char* rm=getenv("TIMESTRAL_TEST_GGUF");require(!real||rm,"real model unset");auto m=Model::load(real?rm:models+"/timestral-"+variant+"-f32.gguf");
  for(auto spec:specs){if(!selected.count(spec.name))continue;
   std::string base=std::string(root)+"/"+variant+"/baseline-"+spec.name+"-";
   require(tstest::host_gates_or_skip(base+"f32.gguf"),"host mismatch");Fixture f(base+"f32.gguf"),truth(base+"f64.gguf"),math(base+"f32-math.gguf");
   validate_predict_companion(f,truth,"f64",spec.name);validate_predict_companion(f,math,"f32-math",spec.name);
   // Require all five established modes; rerun files are retained evidence,
   // not substitutes for the free-running normalized-v1 companions.
   Fixture rerun64(base+"f64-rerun.gguf"),rerunmath(base+"f32-math-rerun.gguf");
   validate_predict_companion(f,rerun64,"f64-rerun",spec.name);validate_predict_companion(f,rerunmath,"f32-math-rerun",spec.name);
   for(auto p:{"pos.b0.","neg.b0."}){auto started=std::chrono::steady_clock::now();std::cerr<<"PREDICT "<<variant<<"/"<<spec.name<<" "<<p<<'\n';try{check(m,f,truth,math,p);}catch(const std::exception&e){++failed;std::cerr<<"FAILED "<<variant<<"/"<<spec.name<<" "<<p<<" "<<e.what()<<'\n';}std::cerr<<"ELAPSED "<<variant<<"/"<<spec.name<<" "<<p<<" seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<'\n';}
   ++cases;
  }
 }
 std::cerr<<"PREDICT cases="<<cases<<" selections="<<selections<<" cov_selections="<<cov_selections<<" max_ratio="<<max_ratio<<" failed="<<failed<<" selection="<<subset<<'\n';return failed?1:0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
