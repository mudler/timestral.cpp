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
