#pragma once
#include <algorithm>
#include <cstddef>
namespace ts::detail {
// Adapted from PyTorch SumKernel.cpp at 5c4886908584029761b579af026dcfb627c84070:
// multi_row_sum, row_sum, vectorized_inner_sum. Scalar lanes retain AVX2 F32
// contiguous-row operation order. Full copyright/licence: third_party/notices/PyTorch.txt.
// Scope and modifications: third_party/notices/README.md.
inline float torch_sum_f32(const float* x,size_t length){
    // For short contiguous inputs the kernel uses four scalar ILP partial sums.
    if(length<8){float partial[4]={};size_t i=0;for(;i+4<=length;i+=4)for(int j=0;j<4;++j)partial[j]+=x[i+j];for(;i<length;++i)partial[0]+=x[i];for(int j=1;j<4;++j)partial[0]+=partial[j];return partial[0];}
    float acc[4][4][8]={};
    size_t groups=length/32,log=0;for(size_t n=groups?groups-1:0;n;n>>=1)++log;
    size_t power=std::max(size_t(4),log/4),step=size_t(1)<<power,mask=step-1,i=0;
    auto add=[&](size_t group){for(int k=0;k<4;++k)for(int lane=0;lane<8;++lane)acc[0][k][lane]+=x[group*32+k*8+lane];};
    for(;i+step<=groups;){for(size_t j=0;j<step;++j,++i)add(i);
        for(size_t level=1;level<4;++level){for(int k=0;k<4;++k)for(int lane=0;lane<8;++lane){acc[level][k][lane]+=acc[level-1][k][lane];acc[level-1][k][lane]=0;}if(i&(mask<<(level*power)))break;}}
    for(;i<groups;++i)add(i);
    for(int level=1;level<4;++level)for(int k=0;k<4;++k)for(int lane=0;lane<8;++lane)acc[0][k][lane]+=acc[level][k][lane];
    size_t vector=groups*4;for(;vector<length/8;++vector)for(int lane=0;lane<8;++lane)acc[0][0][lane]+=x[vector*8+lane];
    for(int k=1;k<4;++k)for(int lane=0;lane<8;++lane)acc[0][0][lane]+=acc[0][k][lane];
    float sum=0;for(size_t d=vector*8;d<length;++d)sum+=x[d];for(int lane=0;lane<8;++lane)sum+=acc[0][0][lane];return sum;
}
}
