#include "predict_companions.hpp"
#include <functional>
#include <filesystem>
template<class F>void rejects_boundary(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}require(bad,"invalid companion accepted");}
void make_file(const std::string& path,const std::string& mode,ggml_type type,bool wrong_shape=false,const std::string& corrupt_key=""){
 auto g=gguf_init_empty();
 for(auto key:{"case.json","model.config_json","model.id","env.timestral_sha","env.timestral_version","env.hf_revision","env.torch","env.numpy","env.python"})gguf_set_val_str(g,key,"fixture");
 gguf_set_val_str(g,"case.name","uni_short");gguf_set_val_str(g,"mode",mode.c_str());
 gguf_set_val_str(g,"env.cpu_capability",tstest::running_cpu_capability().c_str());gguf_set_val_str(g,"env.glibc",tstest::running_glibc().c_str());
 gguf_set_val_u32(g,"model.patch_size",2);gguf_set_val_u32(g,"env.threads",1);
 const char* truth[]={"fixture truth"};gguf_set_arr_str(g,"truth.patches",truth,1);gguf_set_val_str(g,"env.sdpa_kernel",mode=="f32-math"?"MATH (forced)":"fixture kernel");
 if(!corrupt_key.empty())gguf_set_val_str(g,corrupt_key.c_str(),"corrupted provenance");
 auto ctx=ggml_init({4096,nullptr,false});auto t=ggml_new_tensor_4d(ctx,type,wrong_shape?4:2,wrong_shape?2:4,2,1);ggml_set_name(t,"pos.b0.predict");memset(t->data,0,ggml_nbytes(t));gguf_add_tensor(g,t);
 require(gguf_write_to_file(g,path.c_str(),false),"write scratch boundary fixture");gguf_free(g);ggml_free(ctx);
}
int main(){try{
 auto dir=std::filesystem::path(TS_TEST_TMP_DIR)/"predict-companion-boundaries";std::filesystem::create_directories(dir);
 auto a=(dir/"reference.gguf").string(),b=(dir/"companion.gguf").string();make_file(a,"f32",GGML_TYPE_F32);Fixture ref(a);
 auto attempt=[&](const std::string& mode,ggml_type type,bool shape=false,const std::string& key=""){
  make_file(b,mode,type,shape,key);Fixture other(b);validate_predict_companion(ref,other,"f64","uni_short");predict_companion_values(other,"pos.b0.predict",GGML_TYPE_F64,2,4,2);
 };
 attempt("f64",GGML_TYPE_F64);
 rejects_boundary([&]{attempt("f32-math",GGML_TYPE_F32);});
 rejects_boundary([&]{attempt("f64",GGML_TYPE_F32);});
 rejects_boundary([&]{attempt("f64",GGML_TYPE_F64,true);});
 for(auto key:{"case.json","case.name","model.config_json","model.id","env.timestral_sha","env.hf_revision","env.torch","env.cpu_capability","env.glibc","env.threads"})rejects_boundary([&]{attempt("f64",GGML_TYPE_F64,false,key);});
 make_file(b,"f32-math",GGML_TYPE_F32);{Fixture math(b);validate_predict_companion(ref,math,"f32-math","uni_short");predict_companion_values(math,"pos.b0.predict",GGML_TYPE_F32,2,4,2);rejects_boundary([&]{validate_predict_companion(ref,math,"f64","uni_short");});}
 make_file(b,"f32-math",GGML_TYPE_F64);{Fixture math(b);validate_predict_companion(ref,math,"f32-math","uni_short");rejects_boundary([&]{predict_companion_values(math,"pos.b0.predict",GGML_TYPE_F32,2,4,2);});}
 auto specs=manifest(true);require(predict_case_selection("uni_short,multi",specs).size()==2,"valid subset");
 for(auto s:{"bogus","uni_short,bogus","uni_short,uni_short","uni_short,",",uni_short",""})rejects_boundary([&]{predict_case_selection(s,specs);});
 std::filesystem::remove_all(dir);return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
