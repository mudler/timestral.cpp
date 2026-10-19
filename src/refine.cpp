#include "refine.hpp"
#include "common.hpp"
#include "numpy_sum.hpp"
// Reuse the existing NumPy pairwise helper; provenance: third_party/notices/NumPy.txt.
// Algorithms: pinned Timestral scaler.py / forecaster.py (revision in NOTICE).
#include <algorithm>
#include <cmath>
#include <limits>

namespace ts {
namespace {
size_t product(size_t a,size_t b) {
    if (!a || !b || a > SIZE_MAX/b) throw Error("refine: invalid or overflowing shape");
    return a*b;
}
void validate(const Config& c,const std::vector<float>& x,const std::vector<int8_t>& mask,
              size_t rows,size_t length) {
    if (x.size()!=product(rows,length) || mask.size()!=x.size() || !c.patch_size || length%c.patch_size)
        throw Error("refine: invalid context shape");
    if (!(c.minimum_scale>0) || !std::isfinite(c.minimum_scale)) throw Error("refine: invalid minimum scale");
    for(auto b:mask) if(b!=0 && b!=1) throw Error("refine: mask must be boolean");
}
struct State { double w=0,mean=0,m2=0; };
State combine(State a,State b) {
    double w=a.w+b.w, safe=std::max(w,std::numeric_limits<double>::min());
    double delta=b.mean-a.mean;
    double mean=a.mean+delta*b.w/safe;
    double m2=a.m2+b.m2+delta*delta*a.w*b.w/safe;
    return {w,w==0?0:mean,w==0?0:m2};
}
}
BlockFrames advancing_block_frames(const Config& c,const std::vector<float>& x,
 const std::vector<int8_t>& mask,const std::vector<double>& median,size_t rows,size_t length,size_t patches) {
    validate(c,x,mask,rows,length);
    size_t ps=c.patch_size,n=product(rows,patches);
    if(median.size()!=product(n,ps)) throw Error("refine: invalid median shape");
    size_t context_patches=length/ps;
    if(patches>SIZE_MAX-context_patches) throw Error("refine: overflowing scan shape");
    BlockFrames out;out.loc.resize(n);out.scale.resize(n);
    const double clip=std::pow(double(std::numeric_limits<float>::max()),0.5)/1e3;
    std::vector<double> terms(ps),folded(ps);
    for(size_t r=0;r<rows;++r) {
        std::vector<State> states(context_patches+patches);
        for(size_t p=0;p<context_patches;++p) {
            size_t start=r*length+p*ps;
            for(size_t k=0;k<ps;++k) terms[k]=mask[start+k];
            double w=numpy_sum(terms);
            for(size_t k=0;k<ps;++k) terms[k]=double(x[start+k])*mask[start+k];
            double mean=numpy_sum(terms)/std::max(w,1.0);
            for(size_t k=0;k<ps;++k){double d=double(x[start+k])-mean;terms[k]=mask[start+k]*(d*d);}
            double m2=numpy_sum(terms);
            states[p]={w,w==0?0:mean,w==0?0:m2};
        }
        // Literal inclusive Hillis-Steele rounds, including zero-padded horizon.
        for(size_t d=1;d<states.size();) {
            auto previous=states;
            for(size_t i=0;i<states.size();++i) states[i]=combine(i>=d?previous[i-d]:State{},previous[i]);
            if(d>states.size()/2) break;
            d*=2;
        }
        State state=states[context_patches-1];double offset=state.mean;
        for(size_t j=0;j<patches;++j) {
            double var=std::max(state.m2/std::max(state.w-1,1.0),0.0);
            double scale=std::max(std::sqrt(var),c.minimum_scale);
            out.loc[r*patches+j]=state.mean;out.scale[r*patches+j]=scale;
            for(size_t k=0;k<ps;++k) {
                double v=median[(r*patches+j)*ps+k]*scale+state.mean-offset;
                folded[k]=std::clamp(v,-clip,clip)+offset;
            }
            double mean=numpy_sum(folded)/double(ps);
            for(size_t k=0;k<ps;++k){double d=folded[k]-mean;terms[k]=d*d;}
            state=combine(state,{double(ps),mean,numpy_sum(terms)});
        }
    }
    return out;
}
std::vector<float> refine_block(const Config& c,const std::vector<float>& q,const std::vector<float>& x,
 const std::vector<int8_t>& mask,size_t rows,size_t length,size_t patches,const std::vector<int8_t>& targets) {
    size_t steps=product(patches,c.patch_size),Q=c.num_quantiles();
    if(q.size()!=product(product(rows,steps),Q)) throw Error("refine: invalid quantile shape");
    // The reference bypasses context scaling and target selection for one patch.
    if(patches<=1) return q;
    validate(c,x,mask,rows,length);
    if(!targets.empty() && targets.size()!=rows) throw Error("refine: invalid target rows");
    for(auto b:targets) if(b!=0 && b!=1) throw Error("refine: target rows must be boolean");
    if(std::find(c.quantile_knots.begin(),c.quantile_knots.end(),0.5)==c.quantile_knots.end())
        throw Error("refine: missing median knot");
    auto anchor=scale_rows(c,x,mask,rows,length);
    std::vector<double> median(product(rows,steps));
    for(size_t r=0;r<rows;++r) for(size_t t=0;t<steps;++t)
        median[r*steps+t]=(double(q[(r*steps+t)*Q+c.median_index()])-anchor.loc[r*length+length-1])/anchor.scale[r*length+length-1];
    auto frames=advancing_block_frames(c,x,mask,median,rows,length,patches);
    std::vector<float> out(q.size());
    for(size_t r=0;r<rows;++r) {
        float loc0=anchor.loc[r*length+length-1],scale0=anchor.scale[r*length+length-1];
        for(size_t t=0;t<steps;++t) {
            size_t j=t/c.patch_size;
            float loc=j?float(frames.loc[r*patches+j]):loc0,scale=j?float(frames.scale[r*patches+j]):scale0;
            for(size_t k=0;k<Q;++k){size_t i=(r*steps+t)*Q+k;float norm=(q[i]-loc0)/scale0;float v=norm*scale;out[i]=v+loc;}
        }
    }
    out=clamp_nonfinite(out,Q);
    for(size_t r=0;r<rows;++r) if(!targets.empty()&&!targets[r])
        std::copy_n(q.begin()+r*steps*Q,steps*Q,out.begin()+r*steps*Q);
    return out;
}
}
