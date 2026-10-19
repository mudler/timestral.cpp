#pragma once
#include "fixture.hpp"
#include <set>
inline std::string companion_string(gguf_context* g,const std::string& key){
 auto id=gguf_find_key(g,key.c_str());require(id>=0&&gguf_get_kv_type(g,id)==GGUF_TYPE_STRING,"missing/string metadata "+key);return gguf_get_val_str(g,id);
}
inline std::string companion_scalar(gguf_context* g,int64_t id){
 std::ostringstream out;out.precision(17);auto type=gguf_get_kv_type(g,id);out<<int(type)<<":";
 switch(type){
 case GGUF_TYPE_STRING:out<<gguf_get_val_str(g,id);break;
 case GGUF_TYPE_BOOL:out<<gguf_get_val_bool(g,id);break;
 case GGUF_TYPE_UINT32:out<<gguf_get_val_u32(g,id);break;
 case GGUF_TYPE_INT32:out<<gguf_get_val_i32(g,id);break;
 case GGUF_TYPE_UINT64:out<<gguf_get_val_u64(g,id);break;
 case GGUF_TYPE_INT64:out<<gguf_get_val_i64(g,id);break;
 case GGUF_TYPE_FLOAT32:out<<gguf_get_val_f32(g,id);break;
 case GGUF_TYPE_FLOAT64:out<<gguf_get_val_f64(g,id);break;
 default:throw std::runtime_error("unsupported companion provenance metadata type");
 }return out.str();
}
inline void validate_predict_companion(Fixture& reference,Fixture& other,const std::string& mode,const std::string& case_name){
 require(companion_string(reference.b.g,"mode")=="f32","reference mode must be f32");
 require(companion_string(other.b.g,"mode")==mode,"wrong companion mode "+mode);
 require(companion_string(reference.b.g,"case.name")==case_name,"wrong requested case");
 for(auto key:{"case.json","model.config_json","model.id","env.timestral_sha","env.timestral_version","env.hf_revision","env.torch","env.numpy","env.python","env.cpu_capability","env.glibc"}){
  require(companion_string(reference.b.g,key)==companion_string(other.b.g,key),std::string("companion provenance mismatch ")+key);
 }
 std::set<std::string> keys;
 for(auto g:{reference.b.g,other.b.g})for(int64_t id=0;id<gguf_get_n_kv(g);++id){std::string k=gguf_get_key(g,id);if(k.rfind("model.",0)==0||k.rfind("case.",0)==0||k.rfind("source.",0)==0||k.rfind("env.",0)==0){if(k!="env.sdpa_kernel")keys.insert(k);}}
 for(auto& k:keys){auto a=gguf_find_key(reference.b.g,k.c_str()),b=gguf_find_key(other.b.g,k.c_str());require(a>=0&&b>=0,"missing companion provenance "+k);require(companion_scalar(reference.b.g,a)==companion_scalar(other.b.g,b),"companion provenance mismatch "+k);}
 require(tstest::host_matches(companion_string(other.b.g,"env.cpu_capability"),companion_string(other.b.g,"env.glibc"),tstest::running_cpu_capability(),tstest::running_glibc()),"companion host mismatch");
 require(!companion_string(other.b.g,"env.sdpa_kernel").empty(),"missing companion kernel");
 if(mode=="f64"){
  auto id=gguf_find_key(other.b.g,"truth.patches");require(id>=0&&gguf_get_kv_type(other.b.g,id)==GGUF_TYPE_ARRAY&&gguf_get_arr_type(other.b.g,id)==GGUF_TYPE_STRING&&gguf_get_arr_n(other.b.g,id)>0,"missing F64 truth provenance");
  for(size_t i=0;i<gguf_get_arr_n(other.b.g,id);++i)require(std::string(gguf_get_arr_str(other.b.g,id,i)).size()>0,"empty truth patch");
 }
}
inline std::vector<double> predict_companion_values(Fixture& f,const std::string& name,ggml_type type,size_t R,size_t steps,size_t Q){
 auto t=f.tensor(name);require(t->type==type,"wrong prediction dtype "+name);
 require(t->ne[0]==int64_t(Q)&&t->ne[1]==int64_t(steps)&&t->ne[2]==int64_t(R)&&t->ne[3]==1,"wrong padded prediction shape "+name);
 if(type==GGML_TYPE_F64)return f.get<double>(name,type);
 auto v=f.f(name);return {v.begin(),v.end()};
}
inline std::set<std::string> predict_case_selection(const std::string& subset,const std::vector<Case>& specs){
 std::set<std::string> names;if(subset=="all"){for(auto& s:specs)if(!s.cov)names.insert(s.name);return names;}
 std::istringstream in(subset);std::string name;require(!subset.empty()&&subset.back()!=',',"empty case selection");
 while(std::getline(in,name,',')){auto it=std::find_if(specs.begin(),specs.end(),[&](const Case& s){return s.name==name&&!s.cov;});require(it!=specs.end(),"unknown or non-context case "+name);require(names.insert(name).second,"duplicate case "+name);}return names;
}
