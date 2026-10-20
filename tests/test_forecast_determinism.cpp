#include "forecaster.hpp"
#include "fixture.hpp"
#include <cstring>

static ForecastRequest request(Fixture& f,size_t horizon) {
    ForecastRequest r;
    auto t=f.tensor("input.context");
    r.batch=t->ne[2];r.targets=t->ne[1];r.length=t->ne[0];r.horizon=horizon;
    r.context=f.f("input.context");
    for(auto name:{"past","known","cat_past","cat_known"}) {
        std::string n=std::string("input.")+name;
        if(!ggml_get_tensor(f.b.ctx,n.c_str()))continue;
        size_t count=f.tensor(n)->ne[1];
        if(n=="input.past"){r.past=f.f(n);r.past_count=count;}
        if(n=="input.known"){r.known=f.f(n);r.known_count=count;}
        if(n=="input.cat_past"){r.categorical_past=f.get<double>(n,GGML_TYPE_F64);r.categorical_past_count=count;}
        if(n=="input.cat_known"){r.categorical_known=f.get<double>(n,GGML_TYPE_F64);r.categorical_known_count=count;}
    }
    return r;
}
static void check(const Model& m,const ForecastRequest& r) {
    Forecaster fc(m);
    set_cpu_threads(1);auto expected=fc.forecast(r);
    for(int threads:{1,6,6,1}) {
        set_cpu_threads(threads);auto out=fc.forecast(r);
        require(out.quantiles.size()==expected.quantiles.size(),"forecast size");
        require(std::memcmp(out.quantiles.data(),expected.quantiles.data(),
                            out.quantiles.size()*sizeof(float))==0,"forecast bitwise determinism");
    }
    set_cpu_threads(1);
}
int main(int argc,char** argv) {try {
    if(argc>1 && std::string(argv[1])=="--real-512") {
        const char* path=getenv("TIMESTRAL_TEST_GGUF");require(path,"real model unset");
        auto m=Model::load(path);ForecastRequest r;
        r.batch=r.targets=1;r.length=512;r.horizon=32;r.context.resize(r.length);
        for(size_t i=0;i<r.length;++i)r.context[i]=float(int((17*i)%257)-128)/64.f;
        check(m,r);std::cout<<"PASS real 512/32 bitwise repeats threads=1,6,6,1\n";return 0;
    }
    const char* root=getenv("TIMESTRAL_TEST_TINY");
    if(!root){std::cerr<<"SKIP baseline root unset\n";return 77;}
    const char* models=getenv("TIMESTRAL_TEST_MODEL_ROOT");
    std::string mr=models?models:TS_SOURCE_DIR "/tests/fixtures/tiny";
    size_t count=0;
    for(auto variant:{"tiny","tiny-bias"}) {
        auto m=Model::load(mr+"/timestral-"+variant+"-f32.gguf");
        for(const auto& spec:manifest(false)) {
            Fixture f(std::string(root)+"/"+variant+"/baseline-"+spec.name+"-f32.gguf");
            check(m,request(f,spec.horizon));++count;
        }
    }
    std::cout<<"PASS "<<count<<" tiny forecasts bitwise repeats threads=1,6,6,1\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
