#include "forecaster.hpp"
#include "fixture.hpp"
#include <cstring>
#include <iostream>
size_t checks=0,cases=0;
void check(bool b,const char* msg){++checks;require(b,msg);}
void bits(const std::vector<float>& a,const std::vector<float>& b,const char* msg){check(a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*sizeof(float)),msg);}
void blocks(const std::vector<ts::DecodeBlock>& a,const std::vector<ts::DecodeBlock>& b){check(a.size()==b.size(),"block counts");for(size_t i=0;i<a.size();++i){check(a[i].offset==b[i].offset&&a[i].patches==b[i].patches&&a[i].length==b[i].length&&a[i].mask==b[i].mask&&a[i].future_mask==b[i].future_mask,"block metadata");bits(a[i].context,b[i].context,"context bytes");bits(a[i].future,b[i].future,"future bytes");bits(a[i].predicted,b[i].predicted,"prediction bytes");bits(a[i].refined,b[i].refined,"refined bytes");}}
void run(const ts::Model& m,const ts::ForecastRequest& r) {
    ts::Forecaster fc(m);auto plain=fc.forecast(r);ts::ForecastTrace original;size_t progress=0;original.before_block=[&](size_t,size_t,size_t){++progress;};
    auto full=fc.forecast(r,&original);bits(plain.quantiles,full.quantiles,"default trace output");check(!original.initial_anchors,"default has no anchors");
    ts::ForecastTrace captured;captured.capture_initial_anchors=true;auto with=fc.forecast(r,&captured);bits(plain.quantiles,with.quantiles,"capture output");blocks(original.positive,captured.positive);blocks(original.negative,captured.negative);
    ts::ForecastTrace only;only.capture_initial_anchors=true;only.capture_blocks=false;size_t count=0;only.before_block=[&](size_t,size_t,size_t){++count;};auto light=fc.forecast(r,&only);
    bits(plain.quantiles,light.quantiles,"anchor-only output");check(only.positive.empty()&&only.negative.empty()&&count==progress,"anchor-only block/progress");check(bool(only.initial_anchors)&&bool(captured.initial_anchors),"anchor presence");
    auto& anchor=*only.initial_anchors;check(anchor.batch==r.batch&&anchor.targets==r.targets,"anchor dimensions");bits(anchor.loc,captured.initial_anchors->loc,"loc modes");bits(anchor.scale,captured.initial_anchors->scale,"scale modes");
    auto a=ts::assemble_rows(m.config(),r.context,r.observation_mask,r.batch,r.targets,r.length,r.horizon,r.past,r.past_count,r.known,r.known_count,r.max_context,r.categorical_past,r.categorical_past_count,r.categorical_known,r.categorical_known_count);
    check(!captured.positive.empty()&&captured.positive[0].offset==0,"first positive block");auto& first=captured.positive[0];bits(first.context,a.rows,"assembled context");check(first.mask==a.mask,"assembled mask");
    if(!a.future.empty()) {
        size_t width=a.horizon,n=first.patches*m.config().patch_size;
        std::vector<float> future;std::vector<int8_t> fm;
        for(size_t row=0;row<a.rows_count;++row) {
            future.insert(future.end(),a.future.begin()+row*width,a.future.begin()+row*width+n);
            fm.insert(fm.end(),a.future_mask.begin()+row*width,a.future_mask.begin()+row*width+n);
        }
        bits(first.future,future,"assembled first future");check(first.future_mask==fm,"assembled first future mask");
    }
    auto bp=ts::PredictBlock(m).predict(first.context,first.mask,a.rows_count,first.length,first.patches,a.groups,a.roles,first.future,first.future_mask);
    size_t all=a.rows_count/r.batch;std::vector<float> loc,scale;
    for(size_t b=0;b<r.batch;++b)for(size_t v=0;v<r.targets;++v){loc.push_back(bp.loc[b*all+v]);scale.push_back(bp.scale[b*all+v]);}
    bits(anchor.loc,loc,"actual BP location bytes");bits(anchor.scale,scale,"actual BP scale bytes");
    auto copied=anchor;only.capture_initial_anchors=false;fc.forecast(r,&only);check(!only.initial_anchors,"disabled reuse clears");
    only.capture_initial_anchors=true;auto bad=r;bad.context.clear();bool failed=false;try{fc.forecast(bad,&only);}catch(const std::exception&){failed=true;}check(failed&&!only.initial_anchors,"assembly failure clears");
    fc.forecast(r,&only);check(bool(only.initial_anchors),"successful reuse");bad=r;bad.quantile_real_cap_k=NAN;failed=false;try{fc.forecast(bad,&only);}catch(const std::exception&){failed=true;}check(failed&&!only.initial_anchors,"post-capture failure transactional");
    bits(copied.loc,loc,"owned anchors survive reuse");
    if(!captured.negative.empty()) {
        auto positive_request=r;positive_request.symmetric_averaging=false;
        ts::ForecastTrace positive;positive.capture_initial_anchors=true;positive.capture_blocks=false;
        fc.forecast(positive_request,&positive);
        bits(positive.initial_anchors->loc,loc,"symmetry leaves initial location");
        bits(positive.initial_anchors->scale,scale,"symmetry leaves initial scale");
    }
    ++cases;
}
ts::ForecastRequest fixture(Fixture& f,size_t H){ts::ForecastRequest r;auto t=f.tensor("input.context");r.batch=t->ne[2];r.targets=t->ne[1];r.length=t->ne[0];r.horizon=H;r.context=f.f("input.context");
 for(auto name:{"past","known","cat_past","cat_known"}){std::string n=std::string("input.")+name;if(!ggml_get_tensor(f.b.ctx,n.c_str()))continue;size_t C=f.tensor(n)->ne[1];if(n=="input.past"){r.past=f.f(n);r.past_count=C;}if(n=="input.known"){r.known=f.f(n);r.known_count=C;}if(n=="input.cat_past"){r.categorical_past=f.get<double>(n,GGML_TYPE_F64);r.categorical_past_count=C;}if(n=="input.cat_known"){r.categorical_known=f.get<double>(n,GGML_TYPE_F64);r.categorical_known_count=C;}}return r;}
int main(int argc,char** argv){try{bool real=argc>1&&std::string(argv[1])=="--real";const char* suite=getenv(real?"TIMESTRAL_TEST_REAL_BASELINES":"TIMESTRAL_TEST_TINY");const char* model=getenv("TIMESTRAL_TEST_GGUF");if(!suite||(real&&!model))return 77;
 auto m=ts::Model::load(real?model:TS_SOURCE_DIR "/tests/fixtures/tiny/timestral-tiny-f32.gguf");
 for(auto name:real?std::vector<std::string>{"uni_short","known_cov"}:std::vector<std::string>{"uni_short","multi","known_cov","cat_uni","long_h"}) {Fixture f(std::string(suite)+(real?"/real/":"/tiny/")+"baseline-"+name+"-f32.gguf");size_t H=0;for(auto& c:manifest(real))if(c.name==name)H=c.horizon;run(m,fixture(f,H));std::cout<<"ANCHOR "<<name<<" PASS\n";}
 if(!real){ts::ForecastRequest r;r.batch=2;r.targets=2;r.length=19;r.horizon=20;r.max_context=13;r.per_shot_horizon_patches=1;r.context.resize(76);for(size_t row=0;row<4;++row)for(size_t t=0;t<19;++t)r.context[row*19+t]=float(10*row+t*(row+1));r.past_count=r.known_count=1;r.past.resize(38,1000);r.known.resize(78,2000);run(m,r);}
 std::cout<<"PASS anchor cases="<<cases<<" checks="<<checks<<'\n';return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
