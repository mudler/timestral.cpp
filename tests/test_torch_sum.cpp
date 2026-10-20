#include "scaler.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <limits>
#ifdef TS_HAVE_TORCH_SUM
#include "torch_sum.hpp"
#endif
using namespace ts;
static void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
static bool bits(float a,float b){return !std::memcmp(&a,&b,4);}
int main(int argc,char**argv){try{
 // Standalone regression first: masked mirrored 0/-1 row, length 864.
 Config c;c.minimum_scale=1e-6;c.patch_size=32;
 std::vector<float> x(864);std::vector<int8_t> mask(864,1);for(size_t i=0;i<x.size();++i)x[i]=((i/5)%2==0)?-1.f:0.f;
 auto stats=instance_norm_stats(c,x,mask,1,x.size());
 check(bits(stats.loc[0],-0.5023148059844970703125f),"mirrored indicator loc");
 check(bits(stats.scale[0],0.499994695186614990234375f),"mirrored indicator scale must follow pinned F32 reduction");
#ifdef TS_HAVE_TORCH_SUM
 check(detail::torch_sum_f32(nullptr,0)==0.f,"empty sum");
 if(argc==2){std::ifstream f(argv[1],std::ios::binary);uint32_t count=0;f.read((char*)&count,4);size_t values=0;for(uint32_t case_id=0;case_id<count;++case_id){uint32_t n;float expected;f.read((char*)&n,4);f.read((char*)&expected,4);std::vector<float> a(n);f.read((char*)a.data(),n*4);check(bool(f),"truncated sum witness");auto copy=a;float actual=detail::torch_sum_f32(a.data(),a.size());check(bits(actual,expected),"actual Torch sum witness mismatch");check(!std::memcmp(a.data(),copy.data(),n*4),"sum input mutated");values+=n;}check(f.peek()==EOF,"trailing sum witness");std::cout<<"PASS Torch sum witnesses="<<count<<" values="<<values<<'\n';}
#endif
 std::cout<<"PASS independent known-row length864 regression\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
