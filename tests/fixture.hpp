#pragma once
#include "model_loader.hpp"
#include "parity.hpp"
#include "json_min.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
using namespace ts;
inline void require(bool ok, const std::string& s) { if (!ok) throw std::runtime_error(s); }
// One open per fixture: large real baselines must not be reloaded per tensor.
struct Fixture {
    tstest::Baseline b;
    std::vector<size_t> expected_layers;
    explicit Fixture(const std::string& path) { require(b.open(path,true),path); }
    ggml_tensor* tensor(const std::string& n) { auto t=ggml_get_tensor(b.ctx,n.c_str()); require(t!=nullptr,"missing "+n); return t; }
    template<class T> std::vector<T> get(const std::string& n, ggml_type ty) {
        auto t=tensor(n); require(t->type==ty,"wrong type "+n);
        auto p=static_cast<const T*>(t->data); return {p,p+ggml_nelements(t)};
    }
    double number(const std::string& key, bool optional=false) {
        auto id=gguf_find_key(b.g,key.c_str());
        if(id<0 && optional) return std::nan("");
        require(id>=0,"missing metadata "+key);
        auto type=gguf_get_kv_type(b.g,id);
        require(type==GGUF_TYPE_FLOAT64 || type==GGUF_TYPE_FLOAT32,"nonfloat metadata "+key);
        return type==GGUF_TYPE_FLOAT64?gguf_get_val_f64(b.g,id):gguf_get_val_f32(b.g,id);
    }
    std::vector<float> f(const std::string& n) { return get<float>(n,GGML_TYPE_F32); }
    std::vector<int8_t> mask(const std::string& n) { return get<int8_t>(n,GGML_TYPE_I8); }
};
struct Case {
    std::string name;
    int64_t horizon;
    bool cov, known;
    std::vector<size_t> layers; // empty means all, per cases.json
};
inline std::vector<Case> manifest(bool real) {
    std::ifstream in(std::string(TS_SOURCE_DIR)+"/tests/cases.json");
    require(bool(in),"cannot open cases.json");
    std::ostringstream text; text << in.rdbuf();
    auto source=text.str();
    auto doc=jsonmin::Parser(source).object();
    std::vector<Case> result;
    for(const auto& value:doc.at("cases").arr) {
        const auto& o=value.obj;
        if(!real && !o.at("tiny").b) continue;
        auto count=[&](const char* group,const char* field) -> int64_t {
            auto it=o.find(group); if(it==o.end()) return 0;
            auto f=it->second.obj.find(field);
            return f==it->second.obj.end()?0:f->second.integer();
        };
        bool known=count("known","numeric") || count("known","indicator") || count("cat_known","rows");
        bool cov=known || count("past","numeric") || count("cat_past","rows");
        auto h=o.at("H").integer();
        std::vector<size_t> layers;
        if(real) for(const auto& l:o.at("layers").arr) layers.push_back(size_t(l.integer()));
        result.push_back({o.at("name").text,real?h:std::max<int64_t>(1,h/4),cov,known,layers});
    }
    require(!result.empty(),"empty manifest");
    return result;
}

#include <functional>
#include <filesystem>
#include <iostream>
// Coverage is manifest-driven. A real subset must be an explicit comma-separated
// list, never inferred from files (which may still be written by another process).
inline int run_component(int argc,char** argv, const std::function<void(const Model&,Fixture&,const std::string&)>& check, bool all_blocks=false) {
    bool real=argc>1 && std::string(argv[1])=="--real";
    auto specs=manifest(real);
    if(real) {
        const char* subset=getenv("TIMESTRAL_TEST_REAL_CASES");
        if(!subset || !*subset) subset="all";
        std::vector<Case> selected; std::istringstream list(subset); std::string name;
        if(std::string(subset)=="all") selected=specs;
        else while(std::getline(list,name,',')) {
            auto it=std::find_if(specs.begin(),specs.end(),[&](const Case& c){return c.name==name;});
            require(it!=specs.end(),"unknown real case "+name);
            require(std::none_of(selected.begin(),selected.end(),[&](const Case& c){return c.name==name;}),"duplicate real case"); selected.push_back(*it);
        }
        require(!selected.empty(),"empty real case selection");
        specs=selected;
    }
    const char* root=getenv(real?"TIMESTRAL_TEST_REAL_BASELINES":"TIMESTRAL_TEST_TINY");
    if(!root) {std::cerr<<"SKIP baseline root unset\n";return 77;}
    const char* mr=getenv("TIMESTRAL_TEST_MODEL_ROOT");
    std::string models=mr?mr:TS_SOURCE_DIR "/tests/fixtures/tiny";
    for(const auto& variant:real?std::vector<std::string>{"real"}:std::vector<std::string>{"tiny","tiny-bias"})
        for(const auto& spec:specs) require(std::filesystem::exists(std::string(root)+"/"+variant+"/baseline-"+spec.name+"-f32.gguf"),"MISSING "+variant+"/"+spec.name);
    size_t cases=0,blocks=0;
    for(const auto& variant:real?std::vector<std::string>{"real"}:std::vector<std::string>{"tiny","tiny-bias"}) {
        const char* rm=getenv("TIMESTRAL_TEST_GGUF");require(!real || rm,"real model unset");
        auto m=Model::load(real?rm:models+"/timestral-"+variant+"-f32.gguf");
        for(const auto& spec:specs) {
            auto path=std::string(root)+"/"+variant+"/baseline-"+spec.name+"-f32.gguf";
            require(std::filesystem::exists(path),"MISSING "+path);
            require(tstest::host_gates_or_skip(path),"HOST_SKIP"); Fixture f(path);
            f.expected_layers=spec.layers;
            if(f.expected_layers.empty())for(size_t l=0;l<m.config().n_layers;++l)f.expected_layers.push_back(l);
            else {f.expected_layers.push_back(m.config().n_layers-1);std::sort(f.expected_layers.begin(),f.expected_layers.end());f.expected_layers.erase(std::unique(f.expected_layers.begin(),f.expected_layers.end()),f.expected_layers.end());}
            for(auto pass:{"pos","neg"}) {
                int64_t patches=(spec.horizon+m.config().patch_size-1)/m.config().patch_size;
                int last=int((patches+m.config().block_size-1)/m.config().block_size)-1;
                std::vector<int> ids{0};if(all_blocks){for(int k=1;k<=last;++k)ids.push_back(k);}else if(last>0)ids.push_back(last);
                for(int id:ids) {check(m,f,std::string(pass)+".b"+std::to_string(id)+".");++blocks;}
            }
            ++cases;
        }
    }
    std::cerr<<"COMPONENT cases="<<cases<<" pass/blocks="<<blocks<<" missing=0 "<<(real?(specs.size()==manifest(true).size()?"REAL COMPLETE":"REAL EXPLICIT SUBSET"):"TINY COMPLETE")<<'\n';return 0;
}
