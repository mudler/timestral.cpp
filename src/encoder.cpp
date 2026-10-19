#include "encoder.hpp"
#include <limits>
#include <algorithm>
namespace ts {
static size_t encoder_capacity(const Model& model) {
    // Bound graph metadata before ggml construction (including views and leaves).
    const size_t n=model.config().n_layers;
    if(!n || n>(1000000-4096)/512) throw Error("encoder graph capacity exceeded");
    return 4096+512*n;
}
Encoder::Encoder(const Model& model) : model_(model), runner_(encoder_capacity(model)) {}
Embedding Encoder::embed(const std::vector<float>& x,const std::vector<int8_t>& observed,size_t rows,size_t length) {
    const size_t ps=model_.config().patch_size;
    if(!rows || !length || length%ps || rows>std::numeric_limits<size_t>::max()/length || x.size()!=rows*length || observed.size()!=x.size()) throw Error("invalid patch input dimensions");
    if(x.size()>std::numeric_limits<size_t>::max()/2) throw Error("patch size overflow");
    std::vector<float> features(2*x.size());
    for(size_t r=0;r<rows;++r) for(size_t t=0;t<length;++t) {
        size_t i=r*length+t, j=(r*(length/ps)+t/ps)*2*ps+t%ps;
        if(observed[i]!=0 && observed[i]!=1) throw Error("nonboolean observed mask");
        features[j]=x[i]; features[j+ps]=1-observed[i];
    }
    return project(features,rows,length/ps);
}
Embedding Encoder::project(const std::vector<float>& features,size_t rows,size_t patches) {
    const size_t width=2*model_.config().patch_size;
    if(!rows || !patches || rows>std::numeric_limits<size_t>::max()/patches/width || features.size()!=rows*patches*width) throw Error("invalid patch feature dimensions");
    auto ctx=runner_.begin(); auto x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,width,patches,rows); ggml_set_input(x);
    auto y=residual_block(ctx,model_,x); runner_.allocate({y.hidden,y.output}); runner_.upload(x,features); runner_.compute();
    return {features,runner_.read(y.hidden),runner_.read(y.output)};
}
}

#include "rope.hpp"
#include <cmath>
namespace ts {
LayerValues Encoder::run_layer(size_t layer,const std::vector<float>& values,size_t R,size_t S,const TimeMask& mask,const std::vector<int64_t>& positions,const LayerValues& teacher,const std::vector<int8_t>& variate_mask) {
    return execute(layer,values,R,S,mask,positions,teacher,variate_mask,false,{});
}
LayerValues Encoder::run(const std::vector<float>& values,size_t R,size_t S,const TimeMask& mask,const std::vector<int64_t>& positions,const std::vector<size_t>& selected,const std::vector<int8_t>& variate_mask) {
    for(size_t i=0;i<selected.size();++i) {
        if(selected[i]>=model_.config().n_layers || std::find(selected.begin(),selected.begin()+i,selected[i])!=selected.begin()+i) throw Error("invalid selected encoder layer");
    }
    return execute(0,values,R,S,mask,positions,{},variate_mask,true,selected);
}
LayerValues Encoder::execute(size_t layer,const std::vector<float>& values,size_t R,size_t S,const TimeMask& mask,const std::vector<int64_t>& positions,const LayerValues& teacher,const std::vector<int8_t>& variate_mask,bool full,const std::vector<size_t>& selected) {
    const auto& m=model_;
    const auto& c=m.config();size_t D=c.d_model,H=c.num_heads,hd=c.head_dim();
    if(layer>=c.n_layers)throw Error("invalid layer index");
    auto axes=layer_axes(c.block_schedule[layer]);
    if(!R || !S || R>std::numeric_limits<size_t>::max()/S/D || values.size()!=R*S*D || positions.size()!=R*S)throw Error("invalid time layer shape");
    if(S>std::numeric_limits<size_t>::max()/S/R || (!mask.allowed.empty() && mask.allowed.size()!=R*S*S) || mask.enc_len>S || (mask.enc_len && mask.allowed.empty()))throw Error("invalid time mask shape");
    std::vector<float> additive(R*S*S);
    for(size_t r=0;r<R;++r)for(size_t t=0;t<S;++t){bool any=false;for(size_t j=0;j<S;++j){size_t i=(r*S+t)*S+j;int a=mask.allowed.empty()?int(j<=t):mask.allowed[i];if(a!=0 && a!=1)throw Error("nonboolean time mask");if(mask.enc_len && t<mask.enc_len && j>=mask.enc_len && a)throw Error("open two-stream quadrant");any|=a;additive[i]=a?0:-INFINITY;}if(!any)throw Error("fully masked query");}
    auto tables=Rope(m).gather(positions);
    auto& runner=runner_;auto ctx=runner.begin();std::vector<std::pair<ggml_tensor*,const std::vector<float>*>> uploads;
    auto input=[&](const std::vector<float>& v,size_t a,size_t b,size_t d,size_t e=1){auto t=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,a,b,d,e);if(size_t(ggml_nelements(t))!=v.size())throw Error("teacher layout mismatch");ggml_set_input(t);uploads.push_back({t,&v});return t;};
    std::map<std::string,ggml_tensor*> out;
    auto boundary=[&](const std::string& n,ggml_tensor* t){if(full)return t;out[n]=t;auto it=teacher.find(n);return it==teacher.end()?t:input(it->second,t->ne[0],t->ne[1],t->ne[2],t->ne[3]);};
    auto x=input(values,D,S,R);
    if(R>std::numeric_limits<size_t>::max()/R/S || (!variate_mask.empty() && variate_mask.size()!=S*R*R))throw Error("invalid variate mask shape");
    std::vector<float> va(S*R*R);
    for(size_t t=0;t<S;++t)for(size_t r=0;r<R;++r){bool any=false;
        for(size_t k=0;k<R;++k){size_t i=(t*R+r)*R+k;int a=variate_mask.empty()?1:variate_mask[i];
            if(a!=0 && a!=1)throw Error("nonboolean variate mask");any|=a;va[i]=a?0:-INFINITY;}
        if(!any)throw Error("fully masked variate query");
    }
    bool has_time=false,has_variate=false;
    for(size_t li=layer;li<(full?c.n_layers:layer+1);++li)for(bool time:layer_axes(c.block_schedule[li])) {has_time|=time;has_variate|=!time;}
    auto am=has_time?input(additive,S,S,R):nullptr;
    auto vm=has_variate?input(va,R,R,S):nullptr;
    auto cos_input=has_time?input(tables.cos,hd,1,S,R):nullptr;
    auto sin_input=has_time?input(tables.sin,hd,1,S,R):nullptr;
    for(size_t li=layer;li<(full?c.n_layers:layer+1);++li) {
    axes=layer_axes(c.block_schedule[li]);
    std::string b="blk."+std::to_string(li)+".";
    auto residual=x;
    for(size_t j=0;j<axes.size();++j){bool time=axes[j];size_t seq=time?S:R,batch=time?R:S;
        std::string n="attn."+std::to_string(j)+".",a=b+n;
        auto norm=boundary(n+"norm",rms_norm(ctx,residual,m.tensor(b+"attn_norm."+std::to_string(j)+".weight"),c.norm_eps));
        auto xn=time?norm:ggml_cont(ctx,ggml_permute(ctx,norm,0,2,1,3));
        auto fused=affine(ctx,m,a+"qkv",xn,c.attn_bias);if(!full)out[n+"qkv"]=fused;
        auto part=[&](size_t i,const char* name){auto v=ggml_view_4d(ctx,fused,hd,H,seq,batch,hd*4,fused->nb[1],fused->nb[2],i*D*4);return boundary(n+name,ggml_cont(ctx,v));};
        auto q=part(0,"q"),k=part(1,"k"),v=part(2,"v");
        if(time){auto cos=boundary(n+"cos",cos_input);auto sin=boundary(n+"sin",sin_input);
            q=boundary(n+"q_rope",apply_rope(ctx,q,cos,sin));k=boundary(n+"k_rope",apply_rope(ctx,k,cos,sin));}
        auto sdpa=boundary(n+"sdpa",time?attention_time(ctx,q,k,v,am):attention_variate(ctx,q,k,v,vm));auto gated=sdpa;
        if((time?c.time_gate:c.variate_gate)==Gate::SdpaSigmoid){auto gate=boundary(n+"gate",affine(ctx,m,a+"gate",xn,c.attn_bias));gated=ggml_mul(ctx,sdpa,ggml_sigmoid(ctx,gate));}
        if(!time)gated=ggml_cont(ctx,ggml_permute(ctx,gated,0,2,1,3));
        gated=boundary(n+"gated",gated);
        auto att=boundary(n+"out",affine(ctx,m,a+"out",gated,c.attn_bias));
        residual=ggml_add(ctx,residual,att);
    }
    auto mn=boundary("mlp_norm",rms_norm(ctx,residual,m.tensor(b+"ffn_norm.weight"),c.norm_eps));
    auto mlp=boundary("mlp",swiglu(ctx,m,b,mn));x=ggml_add(ctx,residual,mlp);
    if(!full)out["out"]=x;
    else if(std::find(selected.begin(),selected.end(),li)!=selected.end())out["layer."+std::to_string(li)+".out"]=x;
    }
    if(full)out["out_norm"]=rms_norm(ctx,x,m.tensor("out_norm.weight"),c.norm_eps);
    for(const auto& it:teacher)if(!out.count(it.first))throw Error("unknown teacher boundary");
    std::vector<ggml_tensor*> outputs;for(auto& it:out)outputs.push_back(it.second);runner.allocate(outputs);
    for(auto& u:uploads)runner.upload(u.first,*u.second);runner.compute();LayerValues result;for(auto& it:out)result[it.first]=runner.read(it.second);
    // Diagnostic fused projection follows the reference's original interleaved order.
    if(!full)for(size_t j=0;j<axes.size();++j){auto& z=result.at("attn."+std::to_string(j)+".qkv");auto copy=z;for(size_t token=0;token<R*S;++token)for(size_t i=0;i<3;++i)for(size_t h=0;h<H;++h)for(size_t d=0;d<hd;++d)z[token*3*D+i*D+d*H+h]=copy[token*3*D+i*D+h*hd+d];}
    return result;
}
LayerValues replay_time_layer(const Model& m,size_t layer,const std::vector<float>& x,size_t rows,size_t sequence,const TimeMask& mask,const std::vector<int64_t>& positions,const LayerValues& teacher) {
    if(layer>=m.config().n_layers || m.config().block_schedule[layer]!=LayerKind::Time)throw Error("not a time layer");
    return Encoder(m).run_layer(layer,x,rows,sequence,mask,positions,teacher);
}

}
