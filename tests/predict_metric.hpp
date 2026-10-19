#pragma once
#include "parity.hpp"
#include <stdexcept>
namespace tstest {
// G4.3 normalized-v1: fixed f32 anchor, F64 companion derivative and RMS.
// Both companion modes are mandatory. No raw-space floor metadata is used.
inline Result compare_predict_normalized_v1(const std::vector<float>& actual,
 const std::vector<float>& ref32,const std::vector<double>& ref64,
 const std::vector<double>& math32,const std::vector<float>& loc,
 const std::vector<float>& scale) {
 const size_t n=ref32.size(),R=loc.size();
 if(!n||!R||n%R||actual.size()!=n||ref64.size()!=n||math32.size()!=n||scale.size()!=R)
  throw std::runtime_error("G4.3 normalized-v1 missing companion or invalid shape");
 double sq=0,fl=0,err=0;Result out;out.n=n;
 for(size_t i=0;i<n;++i){size_t r=i/(n/R);double l=loc[r],s=scale[r];
  if(!std::isfinite(l)||!std::isfinite(s)||s<=0||!std::isfinite(ref32[i])||!std::isfinite(ref64[i])||!std::isfinite(math32[i])||!std::isfinite(actual[i]))
   throw std::runtime_error("G4.3 normalized-v1 invalid frame or value");
  double z=std::asinh((ref64[i]-l)/s),w=s*std::cosh(z);
  if(!std::isfinite(w)||w<=0)throw std::runtime_error("G4.3 normalized-v1 invalid derivative");
  sq+=z*z;
  fl=std::max(fl,std::max(std::abs(double(ref32[i])-ref64[i]),std::abs(double(ref32[i])-math32[i]))/w);
  double e=std::abs(double(actual[i])-double(ref32[i]))/w;
  if(e>err){err=e;out.worst=i;}
 }
 double rms=std::sqrt(sq/n);
 if(!std::isfinite(rms)||rms==0)throw std::runtime_error("G4.3 normalized-v1 zero or invalid RMS");
 out.err=err/rms;out.tol=std::max(2*(fl/rms),16*kU);out.ok=out.err<=out.tol;
 std::fprintf(stderr,"G4.3 normalized-v1 fixed-f32-anchor floor=%.9g err=%.9g tol=%.9g ratio=%.9g -> %s\n",fl/rms,out.err,out.tol,out.err/out.tol,out.ok?"PASS":"FAIL");
 return out;
}
}
