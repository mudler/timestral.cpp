#pragma once
#include "fixture.hpp"
#include "encoder.hpp"
#include <set>
template<class F> void rejects(F f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}require(rejected,"expected rejection");}
size_t checks=0,layers=0;double maxerr=0,maxratio=0;
double stat(Fixture& f,const std::string& n,const std::string& key){
 auto name="stat."+n+"."+key;auto id=gguf_find_key(f.b.g,name.c_str());require(id>=0,"missing "+name);
 auto ty=gguf_get_kv_type(f.b.g,id);double v;
 if(ty==GGUF_TYPE_UINT32)v=gguf_get_val_u32(f.b.g,id);
 else if(ty==GGUF_TYPE_UINT64)v=double(gguf_get_val_u64(f.b.g,id));
 else v=f.number(name);
 require(std::isfinite(v)&&v>=0,"invalid "+name);return v;
}
void compare(Fixture& f,const std::string& n,const std::vector<float>& x,const char* floor="floor_tf."){
 auto ref=f.f(n);require(stat(f,n,"numel")==ref.size(),"numel "+n);
 double rms=stat(f,n,"rms");double sqref=0;for(float v:ref){require(std::isfinite(v),"nonfinite reference");sqref+=double(v)*v;}
 require(std::abs(rms-std::sqrt(sqref/ref.size()))<=1e-12*std::max(1.,rms),"rms statistic mismatch");
 bool cap_required=n.find(".norm")==std::string::npos && n.find(".mlp_norm")==std::string::npos && n.find(".gated")==std::string::npos;
 if(cap_required){double cap=f.number("cap."+n);require(std::isfinite(cap)&&cap>=0,"invalid cap");}
 double fl=f.number(std::string(floor)+n);require(std::isfinite(fl)&&fl>=0,"invalid floor");
 auto report=tstest::compare_norm(x,ref,fl,f.number("cap."+n,true),n.c_str());require(report.ok,n);
 double e=0,sq=0;for(size_t i=0;i<ref.size();++i){e=std::max(e,std::abs(double(x[i])-ref[i]));sq+=double(ref[i])*ref[i];}
 double err=e/std::max(std::sqrt(sq/ref.size()),double(std::numeric_limits<float>::min()));
 double tol=std::max(2*fl,16*std::ldexp(1.,-24));maxerr=std::max(maxerr,err);maxratio=std::max(maxratio,err/tol);++checks;
}
