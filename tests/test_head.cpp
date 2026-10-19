#include "layer_test_helpers.hpp"
#include "head.hpp"
#include "predict_block.hpp"
#include "predict_metric.hpp"
#include "predict_companions.hpp"
void monotonic(const std::vector<float>& x,size_t Q) {
 require(!x.empty()&&x.size()%Q==0,"quantile shape");
 for(size_t i=0;i<x.size();++i)require(std::isfinite(x[i])&&(i%Q==0||x[i]>=x[i-1]),"nondecreasing quantiles");
}
void units(const Model& m) {
 const auto& c=m.config();size_t Q=c.num_quantiles(),D=c.d_model;
 auto special=std::vector<float>(Q,0);special[c.median_index()]=NAN;rejects([&]{monotone_quantiles(c,special);});
 special.assign(Q,INFINITY);special[c.median_index()]=0;auto extreme=monotone_quantiles(c,special);for(float v:extreme)require(std::isfinite(v),"infinite raw clamped");
 special.assign(Q,-INFINITY);special[c.median_index()]=0;extreme=monotone_quantiles(c,special);for(float v:extreme)require(std::isfinite(v),"negative infinite raw clamped");
 special.assign(Q,1000);monotonic(monotone_quantiles(c,special),Q);special.assign(Q,-1000);monotonic(monotone_quantiles(c,special),Q);
 auto raw=std::vector<float>(3*Q,0);raw[Q+c.median_index()]=-100;
 auto q=monotone_quantiles(c,raw);monotonic(q,Q);
 require(q[c.median_index()]==0&&q[Q+c.median_index()]==-100,"median unchanged");
 rejects([&]{monotone_quantiles(c,{1});});rejects([&]{monotone_quantiles(c,{});});
 Head head(m);std::vector<float> x(6*D,0.25f);
 auto a=head.project(x,2,3);require(a.size()==6*c.patch_size*Q,"head layout");
 head.project(std::vector<float>(D,1),1,1);require(a==head.project(x,2,3),"head allocator reuse");
 rejects([&]{head.project(x,0,3);});rejects([&]{head.project({1},2,3);});
 rejects([&]{PredictBlock(m).context_only({}, {},1,c.patch_size,0,{},{});});
 // Native scaler frames are expanded to time steps, not packed patches.
 // Distinct rows and patch means expose both the row stride and anchor index.
 size_t T=3*c.patch_size,L=3,S=L+1;
 std::vector<float> context(2*T);
 for(size_t r=0;r<2;++r)for(size_t t=0;t<T;++t)context[r*T+t]=float(100*r+t);
 std::vector<int8_t> observed(context.size(),1);
 auto frames=scale_rows(c,context,observed,2,T);
 std::vector<int64_t> positions(2*S);
 for(size_t r=0;r<2;++r)for(size_t j=0;j<S;++j)positions[r*S+j]=j;
 TimeMask mask{std::vector<int8_t>(2*S*S,0),L};
 for(size_t r=0;r<2;++r)for(size_t t=0;t<S;++t)for(size_t j=0;j<=t;++j)mask.allowed[(r*S+t)*S+j]=1;
 auto prediction=PredictBlock(m).context_only(context,observed,2,T,1,mask,positions);
 for(size_t r=0;r<2;++r){
  require(prediction.loc[r]==frames.loc[(r+1)*T-1],"native last-step anchor loc");
  require(prediction.scale[r]==frames.scale[(r+1)*T-1],"native last-step anchor scale");
 }
 auto corrupt=q;corrupt[0]+=100;
 require(!tstest::compare_mag(corrupt,q,1.,8,"negative monotone comparator").ok,"nonvacuous MAG");
}
void check(const Model& m,Fixture& f,Fixture& f64,Fixture& math,const std::string& p,bool head_only) {
 const auto& c=m.config();size_t D=c.d_model,Q=c.num_quantiles(),ps=c.patch_size;
 auto nt=f.tensor(p+"out_norm"),ht=f.tensor(p+"head.in");size_t R=ht->ne[2],h=ht->ne[1],S=nt->ne[1],L=S-h;
 require(ht->ne[0]==D&&nt->ne[0]==D&&nt->ne[2]==int64_t(R)&&L>0,"head dimensions");
 auto norm=f.f(p+"out_norm"),hin=f.f(p+"head.in");std::vector<float> dec;
 for(size_t r=0;r<R;++r)dec.insert(dec.end(),norm.begin()+(r*S+L)*D,norm.begin()+(r+1)*S*D);
 require(dec==hin,"decoder token selection from out_norm");
 compare(f,p+"head.raw",Head(m).project(dec,R,h));
 auto mq=monotone_quantiles(c,f.f(p+"head.raw")),ref=f.f(p+"head.monotone");
 std::vector<double> mag(ref.size());
 for(size_t i=0;i<ref.size();i+=Q){double z=0;for(size_t j=0;j<Q;++j)z=std::max(z,std::abs(double(ref[i+j])));for(size_t j=0;j<Q;++j)mag[i+j]=z;}
 require(tstest::compare_mag(mq,ref,mag,8,(p+"G4.2").c_str()).ok,"G4.2");++checks;monotonic(mq,Q);
 if(head_only)return;
 auto in=f.tensor(p+"args.inputs");require(in->ne[0]==int64_t(L*ps)&&in->ne[1]==int64_t(R),"context-only inputs");
 auto ip=f.get<int32_t>(p+"tf.positions",GGML_TYPE_I32);std::vector<int64_t> pos(ip.begin(),ip.end());
 TimeMask mask{f.mask(p+"mask.time"),L};
 std::vector<int8_t> vm;
 if(ggml_get_tensor(f.b.ctx,(p+"mask.variate").c_str()))vm=f.mask(p+"mask.variate");
 auto result=PredictBlock(m).context_only(f.f(p+"args.inputs"),f.mask(p+"args.observed"),R,L*ps,h,mask,pos,vm);
 require(result.loc==f.f(p+"anchor.loc")&&result.scale==f.f(p+"anchor.scale"),"last context anchor");
 predict_companion_values(f,p+"predict",GGML_TYPE_F32,R,h*ps,Q);
 auto y=f.f(p+"predict");require(result.output.size()==y.size(),"predict shape");monotonic(result.output,Q);
 auto truth=predict_companion_values(f64,p+"predict",GGML_TYPE_F64,R,h*ps,Q);
 auto math_values=predict_companion_values(math,p+"predict",GGML_TYPE_F32,R,h*ps,Q);
 require(tstest::compare_predict_normalized_v1(result.output,y,truth,math_values,result.loc,result.scale).ok,"G4.3");++checks;
}
int main(int argc,char**argv){try{
 const char* mr=getenv("TIMESTRAL_TEST_MODEL_ROOT");std::string models=mr?mr:TS_SOURCE_DIR "/tests/fixtures/tiny";
 for(auto v:{"tiny","tiny-bias"})units(Model::load(models+"/timestral-"+v+"-f32.gguf"));
 if(argc>1&&std::string(argv[1])=="--unit")return 0;
 bool real=argc>1&&std::string(argv[1])=="--real",head_only=argc>2&&std::string(argv[2])=="--head-only";
 const char* root=getenv(real?"TIMESTRAL_TEST_REAL_BASELINES":"TIMESTRAL_TEST_TINY");require(root,"baseline root unset");size_t cases=0,blocks=0,failed=0;
 std::string subset=getenv("TIMESTRAL_TEST_REAL_CASES")?getenv("TIMESTRAL_TEST_REAL_CASES"):"all";
 auto specs=manifest(real);auto selected=predict_case_selection(real?subset:"all",specs);
 for(auto variant:real?std::vector<std::string>{"real"}:std::vector<std::string>{"tiny","tiny-bias"}){
  const char* rm=getenv("TIMESTRAL_TEST_GGUF");require(!real||rm,"real model unset");auto m=Model::load(real?rm:models+"/timestral-"+variant+"-f32.gguf");
  for(auto spec:specs) {if(!selected.count(spec.name))continue;
   auto path=std::string(root)+"/"+variant+"/baseline-"+spec.name+"-f32.gguf";require(tstest::host_gates_or_skip(path),"host gate failed");Fixture f(path);auto base=path.substr(0,path.size()-8);Fixture f64(base+"f64.gguf"),math(base+"f32-math.gguf");
   validate_predict_companion(f,f64,"f64",spec.name);validate_predict_companion(f,math,"f32-math",spec.name);
   for(auto pass:{"pos.b0.","neg.b0."}){std::cerr<<"HEAD "<<variant<<"/"<<spec.name<<" "<<pass<<'\n';try{check(m,f,f64,math,pass,head_only);}catch(const std::exception& e){++failed;std::cerr<<"FAILED "<<variant<<"/"<<spec.name<<" "<<pass<<" "<<e.what()<<"\n";}++blocks;}++cases;
  }
 }
 require(cases>0,"no cases");std::cerr<<"HEAD cases="<<cases<<" blocks="<<blocks<<" comparisons="<<checks<<" max_G4.1_ratio="<<maxratio<<(head_only?" HEAD-ONLY NOT M4":"")<<" failed="<<failed<<" selection="<<subset<<'\n';return failed?1:0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
