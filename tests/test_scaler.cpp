#include "scaler.hpp"
#include "model_loader.hpp"
#include "parity.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace ts;
void require(bool ok, const std::string& s) { if (!ok) throw std::runtime_error(s); }
void units(const Config& source) {
    Config c = source; c.patch_size = 2; c.minimum_scale = 0.00001; c.use_asinh = true;
    std::vector<float> x{0,-0.0f,0,0, 5,5,5,5, 0,1,0,1, -2,0,2,4};
    std::vector<int8_t> m(16,1); std::fill(m.begin(),m.begin()+4,0);
    auto a = scale_rows(c,x,m,4,4);
    require(a.loc[0]==0 && a.scale[0]==float(c.minimum_scale) && a.scaled[1]==0 && !std::signbit(a.scaled[1]),"no observations and masked signed zero");
    require(a.loc[4]==5 && a.scale[4]==float(c.minimum_scale),"constant floor");
    require(a.loc[12]==-1 && a.scale[12]==std::sqrt(2.0f) && a.loc[15]==1,"causal patch last sample statistics");
    auto st=instance_norm_stats(c,x,m,4,4);
    require(st.is_binary==std::vector<int8_t>({0,0,1,0}) && st.loc[2]==0 && st.scale[2]==1,"indicator exact");
    require(st.loc[3]==1 && st.scale[3]==std::sqrt(5.0f),"population variance");
    auto before=a; overwrite_known_rows(c,x,m,{0,0,1,1},4,a);
    require(std::equal(a.scaled.begin(),a.scaled.begin()+8,before.scaled.begin()) && a.scaled[9]==1 && a.loc[12]==1,"known/nonknown overwrite");
    x[0]=std::numeric_limits<float>::quiet_NaN();
    auto bad=scale_rows(c,x,m,4,4);
    require(std::isnan(bad.loc[0]) && bad.scaled[0]==0,"unprepared NaN propagates stats, masked output zero");
    auto cl=clamp_nonfinite({-INFINITY,2,INFINITY,NAN, INFINITY,-INFINITY,NAN,INFINITY},4);
    require(cl[0]==2 && cl[2]==2 && std::isnan(cl[3]) && cl[4]==0 && cl[5]==0 && std::isnan(cl[6]),"finite extrema and all nonfinite");
    auto d=denormalize_quantiles({0,1,2},{3},{2},3,false);
    require(d==std::vector<float>({3,5,7}),"inverse affine no asinh");
    bool rejected=false; try { scale_rows(c,{1},{1},1,1); } catch(const std::exception&) { rejected=true; }
    require(rejected,"reject incomplete patch");
    auto negative = [&](const auto& run, const char* label) {
        bool threw=false; try { run(); } catch(const std::exception&) { threw=true; }
        require(threw,label);
    };
    negative([&] { scale_rows(c,{1,2},{1},1,2); },"mask size mismatch");
    negative([&] { scale_rows(c,{1,2},{1,2},1,2); },"nonboolean mask");
    negative([&] { clamp_nonfinite({1,2},3); },"invalid clamp axis");
    negative([&] { denormalize_quantiles({1,2},{1},{1},3,true); },"invalid quantile shape");
    auto small=c; small.minimum_scale=1e-12;
    auto epsilon=scale_rows(small,{0,1e-10f},{1,1},1,2);
    require(std::abs(epsilon.scaled[0])<0.001f,"float32 epsilon affine clamp");
    auto overflow=denormalize_quantiles({-1000,0,1000,NAN},{2},{3},4,true);
    require(overflow[0]==2 && overflow[1]==2 && overflow[2]==2 && std::isnan(overflow[3]),"sinh overflow clamp preserves NaN");
    auto signed_zero=scale_rows(c,{-0.0f,0.0f},{1,1},1,2);
    require(std::signbit(signed_zero.scaled[0]),"observed negative zero preserved");
    c.stabilize_with_global=true; rejected=false;
    try { scale_rows(c,{1,2},{1,1},1,2); } catch(const std::exception&) { rejected=true; }
    require(rejected,"unsupported stabilization must not silently run");
}

// One open per fixture: large real baselines must not be reloaded per tensor.
struct Fixture {
    tstest::Baseline b;
    explicit Fixture(const std::string& path) { require(b.open(path,true),path); }
    bool has(const std::string& n) { return ggml_get_tensor(b.ctx,n.c_str()); }
    ggml_tensor* tensor(const std::string& n) { auto t=ggml_get_tensor(b.ctx,n.c_str()); require(t!=nullptr,"missing "+n); return t; }
    template<class T> std::vector<T> get(const std::string& n, ggml_type ty) {
        auto t=tensor(n); require(t->type==ty,"wrong type "+n);
        auto p=static_cast<const T*>(t->data); return {p,p+ggml_nelements(t)};
    }
    std::vector<float> f(const std::string& n) { return get<float>(n,GGML_TYPE_F32); }
    std::vector<int8_t> mask(const std::string& n) { return get<int8_t>(n,GGML_TYPE_I8); }
};
void check_fixture(const std::string& path,const Config& c, size_t& blocks) {
    require(tstest::host_gates_or_skip(path),"HOST_SKIP");
    Fixture f(path);
    for (auto pass : {"pos","neg"}) {
        std::vector<int> ids;
        for(int i=0; f.has(std::string(pass)+".b"+std::to_string(i)+".scale.in"); ++i) ids.push_back(i);
        require(!ids.empty(),"no blocks");
        if(ids.size()>1) ids={ids.front(),ids.back()};
        for(int id:ids) {
            std::string p=std::string(pass)+".b"+std::to_string(id)+".";
            auto x=f.f(p+"scale.in"); auto m=f.mask(p+"scale.observed");
            size_t len=f.tensor(p+"scale.in")->ne[0], rows=x.size()/len;
            auto a=scale_rows(c,x,m,rows,len);
            require(tstest::compare_ulp(a.loc,f.f(p+"loc"),0,(p+"G1.1.loc").c_str()).ok,"G1.1 loc");
            require(tstest::compare_ulp(a.scale,f.f(p+"scale"),0,(p+"G1.1.scale").c_str()).ok,"G1.1 scale");
            require(tstest::compare_ulp(a.scaled,f.f(p+"scaled"),2,(p+"G1.2").c_str()).ok,"G1.2");
            if(f.has(p+"known.rows")) {
                auto kr=f.mask(p+"known.rows");
                bool any_known=std::any_of(kr.begin(),kr.end(),[](int8_t v) { return v!=0; });
                require(!any_known || f.has(p+"known.loc"),"missing known statistics");
                if(any_known) {
                    auto st=instance_norm_stats(c,x,m,rows,len);
                    auto rl=f.f(p+"known.loc"), rs=f.f(p+"known.scale");
                    std::vector<double> lm(rows),sm(rows);
                    for(size_t r=0;r<rows;++r) for(size_t t=0;t<len;++t) if(m[r*len+t]) {
                        double v=x[r*len+t]; lm[r]=std::max(lm[r],std::abs(v));
                        sm[r]=std::max(sm[r],(v-rl[r])*(v-rl[r])/rs[r]);
                    }
                    require(tstest::compare_mag(st.loc,rl,lm,32,(p+"G1.3.loc").c_str()).ok,"G1.3 loc");
                    require(tstest::compare_mag(st.scale,rs,sm,32,(p+"G1.3.scale").c_str()).ok,"G1.3 scale");
                    require(tstest::compare_exact_i8(st.is_binary,f.mask(p+"known.is_binary"),(p+"G1.3.binary").c_str()).ok,"G1.3 binary");
                }
                overwrite_known_rows(c,x,m,kr,len,a);
                auto ref=f.f(p+"known.scaled");
                auto original=scale_rows(c,x,m,rows,len);
                auto ref_scale=any_known?f.f(p+"known.scale"):std::vector<float>();
                std::vector<float> got, want; std::vector<double> mag;
                for(size_t r=0;r<rows;++r) {
                    double mx=0; for(size_t t=0;t<len;++t) if(m[r*len+t]) mx=std::max(mx,std::abs(double(x[r*len+t])));
                    for(size_t t=0;t<len;++t) {
                        size_t i=r*len+t;
                        if(kr[r]) { got.push_back(a.scaled[i]); want.push_back(ref[i]); mag.push_back(mx/ref_scale[r]+1); }
                        else require(tstest::detail::bits_of(a.scaled[i])==tstest::detail::bits_of(original.scaled[i]) &&
                                     a.loc[i]==original.loc[i] && a.scale[i]==original.scale[i],"nonknown overwrite");
                    }
                }
                if(!got.empty()) require(tstest::compare_mag(got,want,mag,4,(p+"G1.2.known").c_str()).ok,"known scaled");
            }
            auto inp=f.f(p+"denorm.in"), loc=f.f(p+"anchor.loc"), sc=f.f(p+"anchor.scale");
            auto out=denormalize_quantiles(inp,loc,sc,c.num_quantiles(),c.use_asinh);
            std::vector<double> mag(inp.size()); size_t stride=inp.size()/loc.size();
            for(size_t i=0;i<inp.size();++i) mag[i]=std::abs((c.use_asinh?std::sinh(double(inp[i])):double(inp[i]))*sc[i/stride])+std::abs(double(loc[i/stride]));
            require(tstest::compare_mag(out,f.f(p+"predict"),mag,6,(p+"G1.4").c_str()).ok,"G1.4 denorm");
            if(f.has(p+"refine.clamp.in")) {
                auto z=f.f(p+"refine.clamp.in"); auto last=f.tensor(p+"refine.clamp.in")->ne[0];
                require(tstest::compare_exact_f32(clamp_nonfinite(z,last),f.f(p+"refine.clamp.out"),(p+"clamp").c_str()).ok,"clamp");
            }
            ++blocks;
        }
    }
}
int main(int argc,char** argv) {
    try {
        const char* mr=std::getenv("TIMESTRAL_TEST_MODEL_ROOT");
        std::string models=mr?mr:std::string(TS_SOURCE_DIR)+"/tests/fixtures/tiny";
        auto model=Model::load(models+"/timestral-tiny-f32.gguf"); units(model.config());
        if(argc>1 && std::string(argv[1])=="--unit") return 0;
        const bool real=argc>1 && std::string(argv[1])=="--real";
        const char* root=std::getenv(real?"TIMESTRAL_TEST_REAL_BASELINES":"TIMESTRAL_TEST_TINY");
        if(!root || !*root) { std::cerr<<"SKIP: baseline root unset (unit tests passed)\n"; return 77; }
        std::vector<std::string> cases={"uni_short","uni_one_patch","multi","past_cov","known_cov","cat_uni","cat_multi","long_h","long_h_known","nans","nonneg","const","all_missing","ctx_cap"};
        size_t count=0, blocks=0, missing=0;
        for(auto variant: real?std::vector<std::string>{"real"}:std::vector<std::string>{"tiny","tiny-bias"}) {
            const char* realmodel=std::getenv("TIMESTRAL_TEST_GGUF");
            if(real && !realmodel) { std::cerr<<"SKIP: TIMESTRAL_TEST_GGUF unset\n"; return 77; }
            auto mdl=Model::load(real?realmodel:models+"/timestral-"+variant+"-f32.gguf");
            for(auto name:cases) {
                std::string path=std::string(root)+"/"+variant+"/baseline-"+name+"-f32.gguf";
                if(!std::filesystem::exists(path)) { ++missing; std::cerr<<"MISSING "<<variant<<"/"<<name<<'\n'; continue; }
                std::cerr<<"FIXTURE "<<variant<<"/"<<name<<'\n';
                check_fixture(path,mdl.config(),blocks); ++count;
            }
        }
        std::cerr<<"SCALER cases="<<count<<" pass/blocks="<<blocks<<" missing="<<missing<<'\n';
        return missing?77:0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return std::string(e.what())=="HOST_SKIP"?77:1; }
}
