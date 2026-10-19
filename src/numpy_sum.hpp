#pragma once
#include <cstddef>
#include <vector>
namespace ts {
// NumPy pairwise_sum, contiguous float64: eight lanes, 128-element leaves.
// Adapted from NumPy 2.5.4; see third_party/notices/NumPy.txt.
inline double numpy_sum(const double* x,size_t n){
 if(n<8){double r=-0.0;for(size_t i=0;i<n;++i)r+=x[i];return r;}
 if(n<=128){double r[8];for(size_t j=0;j<8;++j)r[j]=x[j];size_t i=8;
  for(;i<n-(n%8);i+=8)for(size_t j=0;j<8;++j)r[j]+=x[i+j];
  double out=((r[0]+r[1])+(r[2]+r[3]))+((r[4]+r[5])+(r[6]+r[7]));
  for(;i<n;++i)out+=x[i];return out;
 }
 size_t half=n/2;half-=half%8;return numpy_sum(x,half)+numpy_sum(x+half,n-half);
}
inline double numpy_sum(const std::vector<double>& x){return numpy_sum(x.data(),x.size());}
}
