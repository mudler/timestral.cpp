#include "rope.hpp"
#include <cstring>
#include <limits>
namespace ts {
std::vector<float> Rope::inv_freq() const {
    auto t=model_.tensor("rope.inv_freq"); auto p=static_cast<const float*>(t->data); return {p,p+ggml_nelements(t)};
}
RopeTables Rope::gather(const std::vector<int64_t>& positions) const {
    RopeTables result; size_t hd=model_.config().head_dim();
    if(positions.size()>std::numeric_limits<size_t>::max()/hd) throw Error("RoPE size overflow");
    result.cos.resize(positions.size()*hd); result.sin.resize(positions.size()*hd);
    for(size_t i=0;i<positions.size();++i) {
        const int64_t minimum=model_.config().rope_table_pos_min;
        if(positions[i]<minimum || positions[i]>minimum+model_.config().rope_table_len()-1) throw Error("position outside stored RoPE table");
        auto row=model_.rope_index(positions[i]);
        for(auto name:{"rope.cos","rope.sin"}) {
            auto t=model_.tensor(name); auto& v=std::string(name)=="rope.cos"?result.cos:result.sin;
            std::memcpy(v.data()+i*hd, static_cast<const char*>(t->data)+row*t->nb[1],hd*sizeof(float));
        }
    }
    return result;
}
std::vector<float> Rope::rotate(const std::vector<float>& q,const RopeTables& t,size_t rows,size_t sequence,size_t heads) {
    size_t hd=model_.config().head_dim();
    if(!rows || !sequence || !heads || rows>std::numeric_limits<size_t>::max()/sequence/hd/heads || q.size()!=rows*sequence*heads*hd || t.cos.size()!=rows*sequence*hd || t.sin.size()!=t.cos.size()) throw Error("invalid RoPE input dimensions");
    auto ctx=runner_.begin(); auto x=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,hd,heads,sequence,rows);
    auto c=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,hd,1,sequence,rows); auto s=ggml_dup_tensor(ctx,c);
    ggml_set_input(x);ggml_set_input(c);ggml_set_input(s);
    auto y=apply_rope(ctx,x,c,s);runner_.allocate({y});runner_.upload(x,q);runner_.upload(c,t.cos);runner_.upload(s,t.sin);runner_.compute();return runner_.read(y);
}
}
