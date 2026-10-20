#pragma once
// Private diagnostic adaptation of public PyTorch CPU flash loop structure at
// 5c4886908584029761b579af026dcfb627c84070. Keep project notices/PyTorch.txt.
// exp polynomial follows the existing attributed native attention_exp helper.
#include <mkl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
namespace ts::standalone_algorithm {
inline size_t multiply(size_t a,size_t b){if(b&&a>SIZE_MAX/b)throw std::runtime_error("scratch size overflow");return a*b;}
struct Aligned {std::unique_ptr<void,decltype(&free)> raw{nullptr,free};float* data;size_t count;explicit Aligned(size_t n):count(n){if(n>(SIZE_MAX-128)/4)throw std::runtime_error("scratch overflow");void*p=nullptr;if(posix_memalign(&p,64,n*4+128))throw std::bad_alloc();raw.reset(p);std::memset(p,0xa5,n*4+128);data=(float*)((unsigned char*)p+64);}void check()const{auto p=(const unsigned char*)raw.get();for(size_t i=0;i<64;++i)if(p[i]!=0xa5||p[64+4*count+i]!=0xa5)throw std::runtime_error("scratch guard");}};
inline float exp20(float x){if(x < -87.3365478515625f)return 0;if(!std::isfinite(x)||x>88)return std::exp(x);float n=std::floor(std::fma(x,1.4426950216293335f,.5f)),r=std::fma(-n,.6931471824645996f,x),y=std::fma(r,.00828929059f,.0418978221f);for(float a:{.166676521f,.499991506f,.999999701f,1.f})y=std::fma(r,y,a);return(y*std::ldexp(1.f,int(n)-1))*2;}
inline int query_tile(int count){return count>=768?256:count>=192?64:32;}
inline void sdpa(ggml_tensor* dst,int ith,int nth,void* split_data){
 if(!dst||ith<0||nth<=0||ith>=nth)throw std::runtime_error("partition");for(int i=0;i<4;++i)if(!dst->src[i]||!dst->src[i]->data)throw std::runtime_error("input pointer");if(!dst->data)throw std::runtime_error("output pointer");
 auto q=dst->src[0];auto Q=(const float*)q->data,K=(const float*)dst->src[1]->data,V=(const float*)dst->src[2]->data,M=(const float*)dst->src[3]->data;auto Y=(float*)dst->data;
 int64_t hd64=q->ne[0],heads64=q->ne[1],seq64=q->ne[2],rows64=q->ne[3],split64=0;if(reinterpret_cast<uintptr_t>(split_data)>uint64_t(seq64>0?seq64:0))throw std::runtime_error("split bounds");split64=int64_t(reinterpret_cast<uintptr_t>(split_data));if(hd64<=0||heads64<=0||seq64<=0||rows64<=0||rows64>INT32_MAX||split64>seq64||hd64>INT32_MAX||seq64>INT32_MAX||heads64>INT32_MAX/hd64)throw std::runtime_error("LP64 dimensions");int hd=int(hd64),heads=int(heads64),S=int(seq64),R=int(rows64),split=int(split64),stride=heads*hd;size_t elements=multiply(multiply(size_t(R),size_t(S)),size_t(stride));size_t masks=multiply(multiply(size_t(R),size_t(S)),size_t(S));size_t tasks=multiply(multiply(size_t(R),size_t(heads)),size_t(S));if(elements>size_t(PTRDIFF_MAX)/sizeof(float)||masks>size_t(PTRDIFF_MAX)/sizeof(float)||tasks>size_t(INT64_MAX))throw std::runtime_error("index/byte bounds");multiply(elements,sizeof(float));multiply(masks,sizeof(float));
 float scale=float(1/std::sqrt(double(hd)));int64_t task=0;
 for(int row=0;row<R;++row)for(int head=0;head<heads;++head)for(int region=0;region<(split&&split<S?2:1);++region){int begin=region?split:0,count=region?S-split:(split?split:S),keys=region?S:(split?split:S);if(!count)continue;int tile=std::min(query_tile(count),count),keytile=std::min(512,keys);
  for(int first=0;first<count;first+=std::min(tile,count-first),++task){if(task%nth!=ith)continue;int queries=std::min(tile,count-first);Aligned scores(multiply(size_t(tile),size_t(keytile))),output(multiply(size_t(tile),size_t(hd)));std::vector<float> maxima(queries,-INFINITY),sums(queries,0);for(int keyfirst=0;keyfirst<keys;keyfirst+=std::min(keytile,keys-keyfirst)){int width=std::min(keytile,keys-keyfirst);const float* qp=Q+(int64_t(row)*S+begin+first)*stride+head*hd;const float* kp=K+(int64_t(row)*S+keyfirst)*stride+head*hd;
   cblas_sgemm(CblasColMajor,CblasTrans,CblasNoTrans,width,queries,hd,1,kp,stride,qp,queries==1?hd:stride,0,scores.data,width);
   for(int t=0;t<queries;++t){float maximum=-INFINITY;float* line=scores.data+t*width;for(int j=0;j<width;++j){line[j]=line[j]*scale+M[(int64_t(row)*S+begin+first+t)*S+keyfirst+j];maximum=std::max(maximum,line[j]);}maximum=std::max(maxima[t],maximum);if(maximum==-INFINITY){std::fill_n(line,width,0.f);continue;}float lanes[8]={};int end=width/8*8;for(int j=0;j<width;++j){float shifted=line[j]-maximum;line[j]=j<end?exp20(shifted):float(std::exp(double(shifted)));if(j<end)lanes[j%8]+=line[j];}float total=((lanes[0]+lanes[4])+(lanes[2]+lanes[6]))+((lanes[1]+lanes[5])+(lanes[3]+lanes[7]));for(int j=end;j<width;++j)total+=line[j];float factor=std::exp(maxima[t]-maximum);sums[t]=total+factor*sums[t];maxima[t]=maximum;if(keyfirst)for(int d=0;d<hd;++d)output.data[size_t(t)*size_t(hd)+size_t(d)]*=factor;
   }
   const float* vp=V+(int64_t(row)*S+keyfirst)*stride+head*hd;cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,hd,queries,width,1,vp,stride,scores.data,width,keyfirst?1.f:0.f,output.data,hd);
  }
  for(int t=0;t<queries;++t){float reciprocal=1.f/(sums[t]==0?1.f:sums[t]);for(int d=0;d<hd;++d)Y[(int64_t(row)*S+begin+first+t)*stride+head*hd+d]=output.data[size_t(t)*size_t(hd)+size_t(d)]*reciprocal;}scores.check();output.check();
  }
 }
}
}
