#include "refine.hpp"
#include "fixture.hpp"
#include <limits>
#include <cstring>
template<class F> void rejects(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}require(bad,"invalid refine input accepted");}
void target_contract(bool cov, bool present) { require(present==cov,"refine target_rows conditional contract"); }
void units(){
 target_contract(false,false); target_contract(true,true);
 rejects([]{target_contract(true,false);});
 rejects([]{target_contract(false,true);});
 Config c{};c.patch_size=2;c.minimum_scale=1;c.quantile_knots={0.1,0.5,0.9};
 auto frames=advancing_block_frames(c,{0,0},{1,1},{2,2,0,0},1,2,2);
 require(frames.loc==std::vector<double>({0,1}),"causal folding mean");
 require(std::abs(frames.scale[1]-std::sqrt(4./3))<1e-15,"Bessel scale");
 Config pair=c;pair.patch_size=8;
 auto cancel=advancing_block_frames(pair,{0x1p60f,1,-0x1p60f,1,1,1,1,1},std::vector<int8_t>(8,1),std::vector<double>(16),1,8,2);
 require(cancel.loc[0]==0.5,"NumPy pairwise cancellation, not sequential reduction");
 auto level=advancing_block_frames(c,{1e30f,1e30f},{1,1},{1e300,-1e300,0,0},1,2,2);
 require(level.loc[1]==double(1e30f)&&std::isfinite(level.scale[1]),"clip relative to offset preserves large level");
 auto empty=advancing_block_frames(c,{99,99},{0,0},{0,0,0,0},1,2,2);
 require(empty.loc==std::vector<double>({0,0})&&empty.scale==std::vector<double>({1,1}),"zero weight identity");
 std::vector<float> q(12,2);auto r=refine_block(c,q,{0,0},{1,1},1,2,2);
 require(r[0]==2&&r[6]==float(2*float(std::sqrt(4./3)))+1,"float32 affine stages");
 float max=std::numeric_limits<float>::max();
 std::vector<float> overflow={max,-max,-max,max,-max,-max,max,-max,-max,max,-max,-max};
 auto saturated=refine_block(c,overflow,{-max,-max},{1,1},1,2,2);
 require(std::all_of(saturated.begin(),saturated.end(),[&](float v){return v==-max;}),"float32 subtract overflow clamps to finite quantile");
 auto nanq=std::vector<float>(12,0);nanq[0]=NAN;
 require(std::isnan(refine_block(c,nanq,{0,0},{1,1},1,2,2)[0]),"refine keeps NaN");
 rejects([&]{refine_block(c,q,{0},{1},1,1,2);});
 rejects([&]{refine_block(c,q,{0,0},{1,1},1,2,2,{1,0});});
 for(double bad:{0.,-1.,double(NAN),double(INFINITY)}){auto invalid=c;invalid.minimum_scale=bad;rejects([&]{advancing_block_frames(invalid,{0,0},{1,1},{0,0},1,2,1);});}
 auto global=c;global.stabilize_with_global=true;
 rejects([&]{refine_block(global,q,{0,0},{1,1},1,2,2);});
 q[0]=-0.f;q[1]=INFINITY;q[2]=NAN;
 auto unchanged=refine_block(c,q,{0,0},{1,1},1,2,2,{0});
 require(std::memcmp(q.data(),unchanged.data(),q.size()*sizeof(float))==0,"covariates preserve bits including nonfinite and signed zero");
 std::vector<float> single={-0.f,NAN,INFINITY,1,2,3};
 auto bypass=refine_block(c,single,{}, {},1,0,1);
 require(std::memcmp(single.data(),bypass.data(),single.size()*sizeof(float))==0,"single patch reference early return");
 rejects([&]{advancing_block_frames(c,{0,0},{1,2},{0,0},1,2,1);});
 rejects([&]{advancing_block_frames(c,{}, {},{},SIZE_MAX,2,2);});
 rejects([&]{refine_block(c,q,{0,0},{1,1},1,2,SIZE_MAX);});
 rejects([&]{refine_block(c,q,{0,0},{1,1},1,2,2,{2});});
 auto extreme=advancing_block_frames(c,{0,0},{1,1},{INFINITY,-INFINITY,0,0},1,2,2);
 require(std::isfinite(extreme.scale[1])&&extreme.scale[1]>1e15,"fold infinities at clip before moments");
 auto nan=advancing_block_frames(c,{0,0},{1,1},{NAN,0,0,0},1,2,2);
 require(std::isnan(nan.loc[1])&&std::isnan(nan.scale[1]),"NaN must surface");
 require(!tstest::compare_ulp({1},{2},1,"negative frames").ok,"ULP nonvacuous");
 require(!tstest::compare_mag({1},{2},{1.},8,"negative refine").ok,"MAG nonvacuous");
}
size_t frames_count=0,refines=0,bypass_count=0;
void check(const Model& m,Fixture& f,const std::string& p){
 const auto& c=m.config();auto qt=f.tensor(p+"predict");size_t R=qt->ne[2],H=qt->ne[1]/c.patch_size;
 auto x=f.f(p+"refine.context"),q=f.f(p+"predict");auto mask=f.mask(p+"refine.mask");size_t T=f.tensor(p+"refine.context")->ne[0];
 auto id=gguf_find_key(f.b.g,"case.name");require(id>=0,"missing case.name");
 auto specs=manifest(true);auto name=std::string(gguf_get_val_str(f.b.g,id));
 auto spec=std::find_if(specs.begin(),specs.end(),[&](const Case& s){return s.name==name;});
 require(spec!=specs.end(),"unknown case.name");
 bool present=ggml_get_tensor(f.b.ctx,(p+"refine.target_rows").c_str())!=nullptr;
 target_contract(spec->cov,present);
 if(p=="pos.b0.") {
  auto nb=gguf_find_key(f.b.g,"case.n_blocks");require(nb>=0,"missing block count");
  auto tiny=gguf_find_key(f.b.g,"case.tiny");require(tiny>=0,"missing case.tiny");
  size_t horizon=gguf_get_val_bool(f.b.g,tiny)?std::max<int64_t>(1,spec->horizon/4):spec->horizon;
  size_t patches=(horizon+c.patch_size-1)/c.patch_size,blocks=(patches+c.block_size-1)/c.block_size;
  require(gguf_get_val_u32(f.b.g,nb)==blocks,"manifest block count mismatch");
  size_t dumped=0;
  for(int64_t i=0;i<gguf_get_n_tensors(f.b.g);++i){std::string n=gguf_get_tensor_name(f.b.g,i);if(n.size()>=8&&n.substr(n.size()-8)==".predict")++dumped;}
  require(dumped==2*blocks,"unexpected or missing dumped prediction blocks");
 }
 auto targets=present?f.mask(p+"refine.target_rows"):std::vector<int8_t>{};
 auto out=refine_block(c,q,x,mask,R,T,H,targets);auto gold=f.f(p+"refined");
 if(H<=1){require(tstest::compare_exact_f32(out,gold,(p+"bypass").c_str()).ok,"single patch");++bypass_count;return;}
 auto sm=f.get<double>(p+"frames.in.sinh_median",GGML_TYPE_F64);
 auto fr=advancing_block_frames(c,x,mask,sm,R,T,H);
 std::vector<float> loc(fr.loc.begin(),fr.loc.end()),scale(fr.scale.begin(),fr.scale.end());
 for(auto name:{"loc","scale"}){auto d=f.get<double>(p+"frames."+name,GGML_TYPE_F64);std::vector<float> ref(d.begin(),d.end());require(tstest::compare_ulp(std::string(name)=="loc"?loc:scale,ref,1,(p+"frames."+name).c_str()).ok,"G6.1 frames");++frames_count;}
 // Use reference input terms, not the implementation's frames, for the exact PLAN bound.
 auto dl=f.get<double>(p+"frames.loc",GGML_TYPE_F64),ds=f.get<double>(p+"frames.scale",GGML_TYPE_F64);
 auto l0=f.f(p+"refine.loc"),s0=f.f(p+"refine.scale");
 std::vector<double> mag(q.size());size_t Q=c.num_quantiles(),steps=H*c.patch_size;
 for(size_t i=0;i<q.size();++i){size_t row=i/(steps*Q),j=(i/Q%steps)/c.patch_size;float a=l0[row*T+T-1],b=s0[row*T+T-1];float lj=j?float(dl[row*H+j]):a,sj=j?float(ds[row*H+j]):b;
 mag[i]=(std::abs(double(q[i]))+std::abs(double(a)))*double(sj)/b+std::abs(double(lj));
 if(!targets.empty()&&!targets[row])require(std::memcmp(&out[i],&q[i],sizeof(float))==0,"covariate quantile changed");}
 require(tstest::compare_mag(out,gold,mag,8,(p+"refined").c_str()).ok,"G6.1 refined");++refines;
}
int main(int argc,char**argv){try{units();if(argc>1&&std::string(argv[1])=="--unit")return 0;auto status=run_component(argc,argv,check,true);std::cerr<<"REFINE frames="<<frames_count<<" refined="<<refines<<" bypass="<<bypass_count<<'\n';return status;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
