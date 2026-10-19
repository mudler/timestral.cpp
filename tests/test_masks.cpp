#include "masks.hpp"
#include "fixture.hpp"
#include <limits>
template<class F> void rejects(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}require(bad,"invalid mask input accepted");}
void units(){
 Config c{};c.patch_size=2;c.block_size=2;c.attend_missing_patches=true;
 auto a=build_masks(c,{0,0,1,0,0,0,0,0,0,0,0,0},2,6,1,{9,9});
 require(a.obs==std::vector<int8_t>({0,1,0,0,0,0}),"patch observation");
 require(a.key_valid==std::vector<int8_t>({0,1,1,0,0,0}),"leading padding versus internal gaps");
 require(a.first_group==std::vector<int64_t>({1})&&a.inverse==std::vector<int64_t>({0,0}),"group first observed");
 c.attend_missing_patches=false;auto no_missing=build_masks(c,{0,0,1,0,0,0,0,0,0,0,0,0},2,6,1,{9,9});require(no_missing.key_valid==no_missing.obs,"no missing attention");c.attend_missing_patches=true;
 require(a.positions==std::vector<int64_t>({-1,0,1,2,-1,0,1,2}),"shared negative positions");
 c.covariate_enabled=true;auto b=build_masks(c,{1,1,0,0,1,1,1,1},2,4,1,{5,5},{0,2});
 require(b.time.enc_len==2&&b.time.allowed.size()==32,"covariate layout");
 require(!b.variate[(2*2+1)*2]&&b.variate[(2*2)*2+1],"known firewall");
 require(b.anchor==std::vector<int64_t>({0,1,0,1})&&b.active==std::vector<int8_t>({0,1,0,1}),"block layout");
 auto empty=build_masks(c,{0,0},1,2,1,{0});require(empty.first_group==std::vector<int64_t>({0})&&empty.time.allowed==std::vector<int8_t>({1,0,0,1}),"all missing self diagonals");
 auto multi=build_masks(c,std::vector<int8_t>(12,1),1,12,4,{0},{1});require(multi.anchor==std::vector<int64_t>({0,0,2,2,4,4}),"multiple active blocks");
 require(multi.time.allowed[(6+4)*12+2]&&multi.time.allowed[(6+4)*12+3]&&!multi.time.allowed[(6+4)*12+4],"block decoder anchor boundary");
 rejects([&]{build_masks(c,{},0,2,1,{});});
 rejects([&]{build_masks(c,{2,0},1,2,1,{0});});
 rejects([&]{build_masks(c,{1,0},1,2,1,{0},{3});});
 rejects([&]{build_masks(c,{},std::numeric_limits<size_t>::max(),4,1,{});});
}
size_t comparisons=0;
void check(const Model& m,Fixture& f,const std::string& p){
 auto t=f.tensor(p+"args.inputs");size_t R=t->ne[1],T=t->ne[0];auto obs=f.mask(p+"scale.observed");
 auto groups=f.get<int32_t>(p+"args.group_ids",GGML_TYPE_I32);std::vector<int64_t> g(groups.begin(),groups.end()),roles;
 if(ggml_get_tensor(f.b.ctx,(p+"args.role_ids").c_str())){auto v=f.get<int32_t>(p+"args.role_ids",GGML_TYPE_I32);roles.assign(v.begin(),v.end());}
 size_t h=f.tensor(p+"predict")->ne[1]/m.config().patch_size;
 size_t length=obs.size()/R;
 auto a=build_masks(m.config(),obs,R,length,h,g,roles);
 require(a.obs==f.mask(p+"mask.obs_per_patch"),p+"obs");require(a.key_valid==f.mask(p+"mask.key_valid"),p+"key_valid");
 require(a.time.allowed==f.mask(p+"mask.time"),p+"time");
 auto ints=[&](const std::string& n,const std::vector<int64_t>& v){auto x=f.get<int32_t>(p+n,GGML_TYPE_I32);require(std::vector<int64_t>(x.begin(),x.end())==v,p+n);};
 ints("first_obs.group",a.first_group);ints("group.inverse",a.inverse);ints("tf.positions",a.positions);ints("positions",a.positions);
 if(R>1)require(a.variate==f.mask(p+"mask.variate"),p+"variate");
 if(!a.anchor.empty()){ints("mask.block_anchor",a.anchor);require(a.active==f.mask(p+"mask.block_active"),p+"active");}
 comparisons+=7+(R>1?1:0)+(!a.anchor.empty()?2:0);
 require(T>0,"context length");
}
int main(int argc,char**argv){try{units();if(argc>1&&std::string(argv[1])=="--unit")return 0;int status=run_component(argc,argv,check,true);std::cerr<<"MASK comparisons="<<comparisons<<" all-blocks\n";return status;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
