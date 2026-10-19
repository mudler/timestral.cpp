#include "encoder.hpp"
#include <limits>
namespace ts {
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
LayerValues Encoder::run_layer(size_t layer,const std::vector<float>& values,size_t R,size_t S,const TimeMask& mask,const std::vector<int64_t>& positions,const LayerValues& teacher) {
    const auto& m=model_;
    const auto& c=m.config();size_t D=c.d_model,H=c.num_heads,hd=c.head_dim();
    if(layer>=c.n_layers || c.block_schedule[layer]!=LayerKind::Time)throw Error("not a time layer");
    if(!R || !S || R>std::numeric_limits<size_t>::max()/S/D || values.size()!=R*S*D || positions.size()!=R*S)throw Error("invalid time layer shape");
    if(S>std::numeric_limits<size_t>::max()/S/R || (!mask.allowed.empty() && mask.allowed.size()!=R*S*S) || mask.enc_len>S || (mask.enc_len && mask.allowed.empty()))throw Error("invalid time mask shape");
    std::vector<float> additive(R*S*S);
    for(size_t r=0;r<R;++r)for(size_t t=0;t<S;++t){bool any=false;for(size_t j=0;j<S;++j){size_t i=(r*S+t)*S+j;int a=mask.allowed.empty()?int(j<=t):mask.allowed[i];if(a!=0 && a!=1)throw Error("nonboolean time mask");if(mask.enc_len && t<mask.enc_len && j>=mask.enc_len && a)throw Error("open two-stream quadrant");any|=a;additive[i]=a?0:-INFINITY;}if(!any)throw Error("fully masked query");}
    auto tables=Rope(m).gather(positions);
    auto& runner=runner_;auto ctx=runner.begin();std::vector<std::pair<ggml_tensor*,const std::vector<float>*>> uploads;
    auto input=[&](const std::vector<float>& v,size_t a,size_t b,size_t d,size_t e=1){auto t=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,a,b,d,e);if(size_t(ggml_nelements(t))!=v.size())throw Error("teacher layout mismatch");ggml_set_input(t);uploads.push_back({t,&v});return t;};
    std::map<std::string,ggml_tensor*> out;
    auto boundary=[&](const std::string& n,ggml_tensor* t){out[n]=t;auto it=teacher.find(n);return it==teacher.end()?t:input(it->second,t->ne[0],t->ne[1],t->ne[2],t->ne[3]);};
    auto x=input(values,D,S,R);
    auto cos=boundary("attn.0.cos",input(tables.cos,hd,1,S,R));
    auto sin=boundary("attn.0.sin",input(tables.sin,hd,1,S,R));
    auto am=input(additive,S,S,R);
    std::string b="blk."+std::to_string(layer)+".",a=b+"attn.0.";
    auto norm=boundary("attn.0.norm",rms_norm(ctx,x,m.tensor(b+"attn_norm.0.weight"),c.norm_eps));
    auto fused=affine(ctx,m,a+"qkv",norm,c.attn_bias);out["attn.0.qkv"]=fused;
    auto part=[&](size_t i,const char* name){auto v=ggml_view_4d(ctx,fused,hd,H,S,R,hd*4,fused->nb[1],fused->nb[2],i*D*4);return boundary(std::string("attn.0.")+name,ggml_cont(ctx,v));};
    auto q=part(0,"q"),k=part(1,"k"),v=part(2,"v");
    q=boundary("attn.0.q_rope",apply_rope(ctx,q,cos,sin));k=boundary("attn.0.k_rope",apply_rope(ctx,k,cos,sin));
    auto sdpa=boundary("attn.0.sdpa",attention_time(ctx,q,k,v,am));auto gated=sdpa;
    if(c.time_gate==Gate::SdpaSigmoid){auto gate=boundary("attn.0.gate",affine(ctx,m,a+"gate",norm,c.attn_bias));gated=ggml_mul(ctx,sdpa,ggml_sigmoid(ctx,gate));}
    gated=boundary("attn.0.gated",gated);
    auto att=boundary("attn.0.out",affine(ctx,m,a+"out",gated,c.attn_bias));
    auto residual=ggml_add(ctx,x,att);
    auto mn=boundary("mlp_norm",rms_norm(ctx,residual,m.tensor(b+"ffn_norm.weight"),c.norm_eps));
    auto mlp=boundary("mlp",swiglu(ctx,m,b,mn));out["out"]=ggml_add(ctx,residual,mlp);
    for(const auto& it:teacher)if(!out.count(it.first))throw Error("unknown teacher boundary");
    std::vector<ggml_tensor*> outputs;for(auto& it:out)outputs.push_back(it.second);runner.allocate(outputs);
    for(auto& u:uploads)runner.upload(u.first,*u.second);runner.compute();LayerValues result;for(auto& it:out)result[it.first]=runner.read(it.second);
    // Diagnostic fused projection follows the reference's original interleaved order.
    auto& z=result.at("attn.0.qkv");auto copy=z;for(size_t token=0;token<R*S;++token)for(size_t i=0;i<3;++i)for(size_t h=0;h<H;++h)for(size_t d=0;d<hd;++d)z[token*3*D+i*D+d*H+h]=copy[token*3*D+i*D+h*hd+d];
    return result;
}
LayerValues replay_time_layer(const Model& m,size_t layer,const std::vector<float>& x,size_t rows,size_t sequence,const TimeMask& mask,const std::vector<int64_t>& positions,const LayerValues& teacher) {
    return Encoder(m).run_layer(layer,x,rows,sequence,mask,positions,teacher);
}

}
