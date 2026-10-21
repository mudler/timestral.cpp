#include "forecaster.hpp"
#include "decode_state.hpp"
#include "refine.hpp"
#include "predict_companions.hpp"
#include "predict_metric.hpp"
#include <cstring>
#include <chrono>
#include <filesystem>
namespace {
size_t comparisons=0,teacher_blocks=0,free_blocks=0,finals=0,invariants=0;double worst=0;
bool has(Fixture& f,const std::string& n){return ggml_get_tensor(f.b.ctx,n.c_str());}
std::vector<int64_t> ints(Fixture& f,const std::string& n){if(!has(f,n))return {};auto v=f.get<int32_t>(n,GGML_TYPE_I32);return {v.begin(),v.end()};}
ForecastRequest request(Fixture& f,size_t H){
 ForecastRequest r;auto t=f.tensor("input.context");r.batch=t->ne[2];r.targets=t->ne[1];r.length=t->ne[0];r.horizon=H;r.context=f.f("input.context");
 for(auto name:{"past","known","cat_past","cat_known"}){std::string n=std::string("input.")+name;if(!has(f,n))continue;size_t count=f.tensor(n)->ne[1];
  if(n=="input.past"){r.past=f.f(n);r.past_count=count;}if(n=="input.known"){r.known=f.f(n);r.known_count=count;}
  if(n=="input.cat_past"){r.categorical_past=f.get<double>(n,GGML_TYPE_F64);r.categorical_past_count=count;}
  if(n=="input.cat_known"){r.categorical_known=f.get<double>(n,GGML_TYPE_F64);r.categorical_known_count=count;}
 }return r;
}
void bits(const std::vector<float>& a,const std::vector<float>& b,const std::string& n){require(a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0,n);}
void shape(const Forecast& f,const ForecastRequest& r){size_t Q=f.quantile_levels.size();require(f.batch==r.batch&&f.targets==r.targets&&f.horizon==r.horizon&&f.quantiles.size()==r.batch*r.targets*r.horizon*Q,"I6.6 shape");for(size_t i=0;i<f.quantiles.size();++i){require(std::isfinite(f.quantiles[i]),"I6.6 finite");if(i%Q)require(f.quantiles[i]>=f.quantiles[i-1],"I6.6 quantile order");}}
void compare(const std::string& label,const std::vector<float>& actual,Fixture& f,Fixture& f64,Fixture& math,const std::string& gold,const std::string& companion,const std::vector<float>& loc,const std::vector<float>& scale){
 auto ref=f.f(gold);auto d=f64.get<double>(companion,GGML_TYPE_F64);auto m=math.f(companion);std::vector<double> md(m.begin(),m.end());
 require(ref.size()==actual.size(),label+" size");
 auto gold_shape=f.tensor(gold);for(auto other:{&f64,&math}){auto companion_shape=other->tensor(companion);for(size_t axis=0;axis<4;++axis)require(gold_shape->ne[axis]==companion_shape->ne[axis],label+" companion shape");}
 // normalized-v1 floors are recomputed from corresponding companions, not the
 // stored raw-space floor_tf/floor_fr keys. The fixed f32 frame is printed.
 std::cerr<<"G6_COMPARE "<<label<<" gold="<<gold<<" companion="<<companion<<" frame=fixed-f32-block-anchor normalized-v1\n";
 for(size_t i=0;i<loc.size();++i)std::cerr<<"G6_FRAME row="<<i<<" loc="<<loc[i]<<" scale="<<scale[i]<<'\n';
 auto result=tstest::compare_predict_normalized_v1(actual,ref,d,md,loc,scale);require(result.ok,label);worst=std::max(worst,result.err/result.tol);++comparisons;
 // Nonvacuous independently corrupt one element; same companions and frame.
 std::cerr<<"G6_NEGATIVE_EXPECT_FAIL "<<label<<"\n";
 auto bad=ref;bad[0]+=100*scale[0];require(!tstest::compare_predict_normalized_v1(bad,ref,d,md,loc,scale).ok,"negative G6 comparator accepted");std::cerr<<"G6_NEGATIVE_REJECTED "<<label<<"\n";
}
void units(){
 for(size_t n=1;n<100;++n)for(size_t b=1;b<20;++b){DecodeState s=DecodePending{n,0};size_t steps=0;while(auto p=std::get_if<DecodePending>(&s)){require(p->remaining+p->offset==n,"scheduler conservation");size_t k=std::min(b,p->remaining);auto t=reduce_decode(s,{k});require(std::string(t.rule)==(k==p->remaining?"D2-done":"D1-feedback"),"scheduler trace");s=t.state;require(++steps<=n,"scheduler liveness");}require(std::get<DecodeDone>(s).offset==n,"scheduler terminal");bool bad=false;try{reduce_decode(s,{1});}catch(const std::exception&){bad=true;}require(bad,"scheduler rejects duplicate completion");}
}
void invariant_check(const Model& m){
 std::cerr<<"INVARIANTS BEGIN model="<<m.config().name<<"\n";
 Forecaster fc(m);ForecastRequest r;r.batch=2;r.targets=2;r.length=11;r.horizon=m.config().patch_size+1;r.per_shot_horizon_patches=1;r.clip_nonnegative=false;r.context.resize(44);
 for(size_t i=0;i<r.context.size();++i)r.context[i]=float(int(i%13)-6);r.context[0]=NAN;r.context[3]=NAN;
 std::cerr<<"INVARIANT_CALL first threads=1 horizon="<<r.horizon<<" shot="<<r.per_shot_horizon_patches<<"\n";
 set_cpu_threads(1);auto a=fc.forecast(r);shape(a,r);std::cerr<<"INVARIANT_CALL repeat threads=1\n";auto again=fc.forecast(r);bits(a.quantiles,again.quantiles,"I6.5 repeated process run");set_cpu_threads(2);std::cerr<<"INVARIANT_CALL threads=2\n";auto threaded=fc.forecast(r);bits(a.quantiles,threaded.quantiles,"I6.5 threads 1 vs 2");set_cpu_threads(1);
 for(auto& x:r.context)x=-x;std::cerr<<"INVARIANT_CALL sign threads=1\n";auto neg=fc.forecast(r);size_t Q=m.config().num_quantiles();auto reflected=neg.quantiles;
 for(size_t i=0;i<reflected.size();++i)reflected[i]=-neg.quantiles[(i/Q)*Q+Q-1-i%Q];bits(a.quantiles,reflected,"I6.4 bitwise sign symmetry clip off");
 r.clip_nonnegative=true;for(auto& x:r.context)x=std::abs(x);std::cerr<<"INVARIANT_CALL clip threads=1\n";auto clipped=fc.forecast(r);for(float x:clipped.quantiles)require(x>=0,"nonnegative clipping");
 r.horizon=0;bool rejected=false;try{fc.forecast(r);}catch(const std::exception&){rejected=true;}require(rejected,"zero horizon accepted");
 // Tiny refusal/invalid input coverage uses the same native preparation as M5.
 r.horizon=1;r.known_count=1;r.known.assign(r.batch*(r.length+r.horizon),0);r.context.assign(r.batch*r.targets*r.length,1);
 if(!supports_known_covariates(m.config())){bool refused=false;try{fc.forecast(r);}catch(const std::exception& e){refused=std::string(e.what())=="this checkpoint cannot condition on known-future covariates";}require(refused,"reference known-covariate refusal");}
 r.known.clear();r.known_count=0;r.observation_mask.assign(r.context.size(),2);rejected=false;try{fc.forecast(r);}catch(const std::exception&){rejected=true;}require(rejected,"invalid forecast mask accepted");
 ++invariants;std::cerr<<"INVARIANTS I6.4=PASS I6.5=PASS I6.6=PASS threads=1,2 repeated=2 multiblock=1\n";
}
void feedback_contract(const Config& c,const std::vector<DecodeBlock>& blocks,const std::vector<int64_t>& roles,size_t R,size_t cap){
 size_t Q=c.num_quantiles(),ps=c.patch_size;
 for(size_t k=1;k<blocks.size();++k){const auto& before=blocks[k-1];const auto& after=blocks[k];size_t n=before.patches*ps,joined=before.length+n;
  size_t keep=joined>cap?std::max(ps,(cap/ps)*ps):joined,start=joined-keep;
  require(after.length==keep&&after.context.size()==R*keep&&after.mask.size()==R*keep,"feedback context cap/shape");
  for(size_t row=0;row<R;++row)for(size_t t=0;t<keep;++t){size_t from=start+t;float value;int8_t observed;
   if(from<before.length){value=before.context[row*before.length+from];observed=before.mask[row*before.length+from];}
   else {size_t j=from-before.length;value=before.refined[(row*n+j)*Q+c.median_index()];observed=1;
    if(!std::isfinite(value)||std::abs(value)>=c.extreme_value_threshold)value=0;
    if(!roles.empty()&&roles[row]==1){value=0;observed=0;}
    if(!roles.empty()&&roles[row]==2){require(!before.future.empty(),"known feedback missing");value=before.future[row*n+j];observed=before.future_mask[row*n+j];}
   }
   require(std::memcmp(&value,&after.context[row*keep+t],sizeof(float))==0&&observed==after.mask[row*keep+t],"feedback values/masks");
  }
 }
}
void check(const Model& model,const Case& spec,const std::string& base){
 Fixture f(base+"f32.gguf"),truth(base+"f64.gguf"),math(base+"f32-math.gguf"),tf64(base+"f64-rerun.gguf"),tfmath(base+"f32-math-rerun.gguf");
 validate_predict_companion(f,truth,"f64",spec.name);validate_predict_companion(f,math,"f32-math",spec.name);validate_predict_companion(f,tf64,"f64-rerun",spec.name);validate_predict_companion(f,tfmath,"f32-math-rerun",spec.name);
 auto r=request(f,spec.horizon);Forecaster fc(model);ForecastTrace trace;trace.before_block=[&](size_t offset,size_t patches,size_t length){std::cerr<<"FREE_BLOCK_BEGIN "<<spec.name<<" offset="<<offset<<" patches="<<patches<<" context="<<length<<"\n";};std::cerr<<"FREE_RUNNING_BEGIN case="<<spec.name<<" horizon="<<r.horizon<<"\n";auto out=fc.forecast(r,&trace);std::cerr<<"FREE_RUNNING_END case="<<spec.name<<"\n";shape(out,r);
 const auto& c=model.config();size_t ps=c.patch_size,Q=c.num_quantiles();size_t count=((r.horizon+ps-1)/ps+std::min(c.per_shot_horizon_patches,c.block_size)-1)/std::min(c.per_shot_horizon_patches,c.block_size);
 require(trace.positive.size()==count&&trace.negative.size()==count,"block count");
 auto role_contract=ints(f,"pos.b0.args.role_ids");size_t rows=f.tensor("pos.b0.args.inputs")->ne[1];
 feedback_contract(c,trace.positive,role_contract,rows,c.context_length);feedback_contract(c,trace.negative,role_contract,rows,c.context_length);
 auto meta=gguf_find_key(f.b.g,"case.n_blocks");require(meta>=0&&gguf_get_val_u32(f.b.g,meta)==count,"dumped block count");
 size_t dumped=0;for(int64_t i=0;i<gguf_get_n_tensors(f.b.g);++i){std::string n=gguf_get_tensor_name(f.b.g,i);if(n.size()>=8&&n.substr(n.size()-8)==".predict")++dumped;}require(dumped==count*2,"unexpected/missing prediction blocks");
 for(size_t pass=0;pass<2;++pass)for(size_t k=0;k<count;++k){std::string p=std::string(pass?"neg":"pos")+".b"+std::to_string(k)+".";auto& block=(pass?trace.negative:trace.positive)[k];auto t=f.tensor(p+"args.inputs");size_t R=t->ne[1],T=t->ne[0],h=f.tensor(p+"predict")->ne[1]/ps;
  require(block.patches==h&&block.offset==k*std::min(c.block_size,c.per_shot_horizon_patches)*ps,"decode schedule");
  auto group=ints(f,p+"args.group_ids"),roles=ints(f,p+"args.role_ids");auto x=f.f(p+"args.inputs");auto mask=f.mask(p+"args.observed");
  bits(x,f.f(p+"refine.context"),"teacher prediction/refinement context differs");require(mask==f.mask(p+"refine.mask"),"teacher prediction/refinement mask differs");
  bool target_present=has(f,p+"refine.target_rows");require(target_present==spec.cov,"target-row contract");auto targets=target_present?f.mask(p+"refine.target_rows"):std::vector<int8_t>{};
  std::cerr<<"TEACHER_BEGIN "<<spec.name<<" "<<p<<" R="<<R<<" T="<<T<<" patches="<<h<<"\n";auto begin=std::chrono::steady_clock::now();
  auto pred=PredictBlock(model).predict(x,mask,R,T,h,group,roles,has(f,p+"args.known_future")?f.f(p+"args.known_future"):std::vector<float>{},has(f,p+"args.known_future_mask")?f.mask(p+"args.known_future_mask"):std::vector<int8_t>{});
  std::cerr<<"TEACHER_PREDICT_END "<<spec.name<<" "<<p<<" seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()<<"\n";
  auto refined=refine_block(c,pred.output,x,mask,R,T,h,targets);auto loc=f.f(p+"anchor.loc"),scale=f.f(p+"anchor.scale");
  compare("G6.2 floor_tf predict "+p,pred.output,f,tf64,tfmath,p+"predict",p+"predict",loc,scale);
  compare("G6.2 floor_tf predict+refine "+p,refined,f,tf64,tfmath,p+"refined",p+"block_refined",loc,scale);++teacher_blocks;
  compare("G6.3 floor_fr predict "+p,block.predicted,f,truth,math,p+"predict",p+"predict",loc,scale);
  compare("G6.3 floor_fr refined "+p,block.refined,f,truth,math,p+"refined",p+"refined",loc,scale);++free_blocks;
 }
 // Final (B,V,H,Q) uses the original positive-pass target anchor, explicitly
 // fixed across the horizon. No post-feedback or mirrored frame substitution.
 auto loc=f.f("pos.b0.anchor.loc"),scale=f.f("pos.b0.anchor.scale");size_t all=loc.size()/r.batch;std::vector<float> l,s;
 for(size_t b=0;b<r.batch;++b)for(size_t v=0;v<r.targets;++v){l.push_back(loc[b*all+v]);s.push_back(scale[b*all+v]);}
 auto ft=f.tensor("final.quantiles");require(ft->ne[0]==int64_t(Q)&&ft->ne[1]==int64_t(r.horizon)&&ft->ne[2]==int64_t(r.targets)&&ft->ne[3]==int64_t(r.batch),"dumped final shape");
 compare("G6.3 final floor_fr original-positive-target-anchor",out.quantiles,f,truth,math,"final.quantiles","final.quantiles",l,s);++finals;
}
}
int main(int argc,char** argv){try{
 units();bool real=argc>1&&std::string(argv[1])=="--real",unit=argc>1&&std::string(argv[1])=="--unit";
 std::string models=getenv("TIMESTRAL_TEST_MODEL_ROOT")?getenv("TIMESTRAL_TEST_MODEL_ROOT"):TS_SOURCE_DIR "/tests/fixtures/tiny";
 if(unit){auto model=Model::load(models+"/timestral-tiny-f32.gguf");invariant_check(model);return 0;}
 auto specs=manifest(real);std::string subset=getenv("TIMESTRAL_TEST_REAL_CASES")?getenv("TIMESTRAL_TEST_REAL_CASES"):"all";std::set<std::string> selected;
 if(subset=="all")for(auto& s:specs)selected.insert(s.name);else{require(!subset.empty()&&subset.back()!=',',"empty selection");std::istringstream in(subset);std::string name;while(std::getline(in,name,',')){require(std::any_of(specs.begin(),specs.end(),[&](const Case& c){return c.name==name;}),"unknown case "+name);require(selected.insert(name).second,"duplicate case");}}
 const char* root=getenv(real?"TIMESTRAL_TEST_REAL_BASELINES":"TIMESTRAL_TEST_TINY");if(!root){std::cerr<<"SKIP: baseline root unset\n";return 77;}size_t cases=0,failed=0;
 for(auto variant:real?std::vector<std::string>{"real"}:std::vector<std::string>{"tiny","tiny-bias"}){
  const char* rm=getenv("TIMESTRAL_TEST_GGUF");require(!real||rm,"real model unset");auto model=Model::load(real?rm:models+"/timestral-"+variant+"-f32.gguf");invariant_check(model);
  for(auto& spec:specs)if(selected.count(spec.name)){auto started=std::chrono::steady_clock::now();std::string base=std::string(root)+"/"+variant+"/baseline-"+spec.name+"-";std::cerr<<"FORECAST "<<variant<<"/"<<spec.name<<'\n';try{check(model,spec,base);++cases;}catch(const std::exception& e){++failed;std::cerr<<"FAILED "<<variant<<"/"<<spec.name<<" "<<e.what()<<'\n';}std::cerr<<"ELAPSED "<<variant<<"/"<<spec.name<<" seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<'\n';}
 }
 std::cerr<<"FORECAST cases="<<cases<<" teacher_blocks="<<teacher_blocks<<" free_blocks="<<free_blocks<<" finals="<<finals<<" comparisons="<<comparisons<<" invariant_models="<<invariants<<" max_ratio="<<worst<<" failed="<<failed<<" selection="<<subset<<'\n';return failed?1:0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
