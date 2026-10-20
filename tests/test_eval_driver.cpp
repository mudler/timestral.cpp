#include "protocol.hpp"
#include "weight_policy.hpp"
#include "fixture.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <unistd.h>
size_t checks=0;
void check(bool x,const char* s){++checks;require(x,s);}
void put(std::vector<unsigned char>& b,size_t p,uint64_t v,size_t n){for(size_t i=0;i<n;++i)b.at(p+i)=unsigned(v>>(8*i))&255;}
std::vector<unsigned char> encode(const ts::ForecastRequest& r,const tsq::Digest& model) {
 std::string name="native-test";size_t header=288+name.size()+192;std::vector<unsigned char>b(header);std::memcpy(b.data(),"TSQREQ01",8);put(b,8,1,4);put(b,12,header,4);
 size_t dims[]={r.batch,r.targets,r.length,r.horizon,r.past_count,r.known_count,r.categorical_past_count,r.categorical_known_count};for(int i=0;i<8;++i)put(b,24+i*8,dims[i],8);
 put(b,88,32768,8);put(b,96,32,8);put(b,104,32,8);put(b,112,3,4);
 double options[]={10000,1e30,1e-6};std::memcpy(b.data()+120,options,24);put(b,144,1,4);put(b,152,name.size(),4);std::copy(model.begin(),model.end(),b.begin()+160);
 std::string hf="9874d252ab0aae16f7f7c999f98f98d6ad7d8365",ref="a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff";auto h=tsq::hash_bytes(hf.data(),hf.size()),c=tsq::hash_bytes(ref.data(),ref.size());std::copy(h.begin(),h.end(),b.begin()+192);std::copy(c.begin(),c.end(),b.begin()+224);std::copy(name.begin(),name.end(),b.begin()+288);
 const void* ptrs[]={r.context.data(),nullptr,r.past.data(),r.known.data(),r.categorical_past.data(),r.categorical_known.data()};size_t counts[]={r.context.size(),0,r.past.size(),r.known.size(),r.categorical_past.size(),r.categorical_known.size()},widths[]={4,1,4,4,8,8};int types[]={1,2,1,1,3,3};
 for(int i=0;i<6;++i){size_t p=288+name.size()+i*32,n=counts[i]*widths[i];put(b,p,types[i],4);put(b,p+8,counts[i],8);put(b,p+16,n,8);put(b,p+24,n?b.size():0,8);if(n){auto bytes=static_cast<const unsigned char*>(ptrs[i]);b.insert(b.end(),bytes,bytes+n);}}
 put(b,16,b.size(),8);return b;
}
void reject(const std::vector<unsigned char>& b){bool fail=false;try{tsq::decode(b);}catch(const std::exception&){fail=true;}check(fail,"mutation accepted");}
ts::ForecastRequest request(Fixture& f,size_t H){ts::ForecastRequest r;auto t=f.tensor("input.context");r.batch=t->ne[2];r.targets=t->ne[1];r.length=t->ne[0];r.horizon=H;r.context=f.f("input.context");for(auto name:{"past","known","cat_past","cat_known"}){std::string n=std::string("input.")+name;if(!ggml_get_tensor(f.b.ctx,n.c_str()))continue;size_t C=f.tensor(n)->ne[1];if(n=="input.past"){r.past=f.f(n);r.past_count=C;}if(n=="input.known"){r.known=f.f(n);r.known_count=C;}if(n=="input.cat_past"){r.categorical_past=f.get<double>(n,GGML_TYPE_F64);r.categorical_past_count=C;}if(n=="input.cat_known"){r.categorical_known=f.get<double>(n,GGML_TYPE_F64);r.categorical_known_count=C;}}return r;}
void provenance_cases(const ts::Model& m){
 std::string path=std::string(TS_TEST_TMP_DIR)+"/driver-provenance.gguf";
 for(unsigned format=0;format<6;++format){auto g=gguf_init_empty();
  auto set=[&](const char* k,const std::string& v){gguf_set_val_str(g,k,v.c_str());};
  if(format==1||format==2){std::string name=format==1?"f16":"q8_0";set("timestral.conversion.outtype",name);set("timestral.conversion.type_policy","b4-"+name+"-v1");set("timestral.conversion.converter_sha256",std::string(64,'a'));}
  if(format>=3){const char* names[]={"q6_k","q5_k","q4_k"};set("timestral.quantization.type",names[format-3]);set("timestral.quantization.version","1");set("timestral.quantization.tool_version","1");set("timestral.quantization.policy","b4-k-v1");set("timestral.quantization.input_gguf_sha256",std::string(64,'b'));set("timestral.quantization.source_sha256",std::string(64,'c'));set("timestral.quantization.runtime_commit",std::string(40,'d'));set("timestral.quantization.ggml_commit","e705c5fed490514458bdd2eaddc43bd098fcce9b");}
  for(auto& n:m.tensor_names()){auto t=*m.tensor(n);t.buffer=nullptr;gguf_add_tensor(g,&t);}
  check(gguf_write_to_file(g,path.c_str(),false),"provenance fixture");tsq::validate_provenance(path,static_cast<tsq::Format>(format));check(true,"valid format metadata");
  set(format>=3?"timestral.quantization.policy":"timestral.conversion.outtype","wrong");check(gguf_write_to_file(g,path.c_str(),false),"bad provenance fixture");bool no=false;try{tsq::validate_provenance(path,static_cast<tsq::Format>(format));}catch(const std::exception&){no=true;}check(no,"metadata conflict accepted");gguf_free(g);
 }
 unlink(path.c_str());
}
int main(int argc,char** argv){try{
 bool real=argc>1&&std::string(argv[1])=="--real";const char* suite=getenv(real?"TIMESTRAL_TEST_REAL_BASELINES":"TIMESTRAL_TEST_TINY"),*path=getenv("TIMESTRAL_TEST_GGUF");if(!suite||(real&&!path))return 77;std::string model=real?path:TS_SOURCE_DIR "/tests/fixtures/tiny/timestral-tiny-f32.gguf";auto m=ts::Model::load(model);auto hash=tsq::hash_file(model);if(real)tsq::validate_model(m);
 if(!real)provenance_cases(m);
 size_t cases=0;
 for(auto name:{"uni_short","known_cov"}){size_t H=std::string(name)=="uni_short"?(real?96:24):(real?64:16);Fixture f(std::string(suite)+(real?"/real/":"/tiny/")+"baseline-"+name+"-f32.gguf");auto r=request(f,H);auto bytes=encode(r,hash);auto decoded=tsq::decode(bytes);check(!std::memcmp(decoded.request.context.data(),r.context.data(),r.context.size()*4),"input bytes");
  if(!real){ // tiny model has intentionally different geometry; codec options
   decoded.request.max_context=0;decoded.request.per_shot_horizon_patches=0;
  }
  r.symmetric_averaging=true;r.clip_nonnegative=true;r.quantile_real_cap_k=10000;if(real){r.max_context=32768;r.per_shot_horizon_patches=32;}
  ts::Forecaster fc(m);ts::ForecastTrace a,b;a.capture_initial_anchors=b.capture_initial_anchors=true;a.capture_blocks=b.capture_blocks=false;auto original=fc.forecast(r,&a),transport=fc.forecast(decoded.request,&b);
  check(original.quantiles==transport.quantiles&&!std::memcmp(original.quantiles.data(),transport.quantiles.data(),original.quantiles.size()*4),"forecast transport bytes");check(a.initial_anchors&&b.initial_anchors,"anchor presence");check(!std::memcmp(a.initial_anchors->loc.data(),b.initial_anchors->loc.data(),r.batch*r.targets*4)&&!std::memcmp(a.initial_anchors->scale.data(),b.initial_anchors->scale.data(),r.batch*r.targets*4),"anchor bytes");
  tsq::Identity id;id.model=hash;id.commit=std::string(40,'a');auto result=tsq::result(decoded,id,&transport,&*b.initial_anchors);check(!std::memcmp(result.data(),"TSQRES01",8),"result magic");
  if(const char* dir=getenv("TSQ_EXPORT_DIR")) {
   std::string base=std::string(dir)+"/"+name;
   std::ofstream input(base+".request",std::ios::binary);input.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());
   std::ofstream expected(base+".expected",std::ios::binary);expected.write(reinterpret_cast<const char*>(result.data()),result.size());
   check(bool(input)&&bool(expected),"export transport fixtures");
  }
  if(!real){
   for(unsigned format=0;format<6;++format){auto v2=bytes;put(v2,8,2,4);put(v2,116,format,4);put(v2,156,format==0?1:0,4);auto decoded2=tsq::decode(v2);check(decoded2.version==2&&static_cast<unsigned>(decoded2.format)==format&&decoded2.capture_anchors==(format==0),"v2 format/flags");
    auto response=tsq::result(decoded2,id,&transport,format==0?&*b.initial_anchors:nullptr);check(response.size()==result.size()-(format?8*r.batch*r.targets:0),"v2 anchor length");
    bool rejected=false;try{tsq::result(decoded2,id,&transport,format?&*b.initial_anchors:nullptr);}catch(const std::exception&){rejected=true;}check(rejected,"wrong anchor presence");
    for(auto mutation:std::vector<std::pair<size_t,unsigned>>{{116,6},{156,2},{156,format==0?0u:1u}}){auto bad=v2;put(bad,mutation.first,mutation.second,4);reject(bad);}
   }
   const ggml_type types[]={GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_Q8_0};
   // Descriptor-only checks never execute a tensor after changing its type.
   for(unsigned fmt=0;fmt<3;++fmt){for(auto& n:m.tensor_names())if(ts::reduced_weight_name(n))m.tensor(n)->type=types[fmt];tsq::validate_storage(m,static_cast<tsq::Format>(fmt));
    auto excluded=m.tensor("out_norm.weight");excluded->type=GGML_TYPE_F16;bool no=false;try{tsq::validate_storage(m,static_cast<tsq::Format>(fmt));}catch(const std::exception&){no=true;}excluded->type=GGML_TYPE_F32;check(no,"reduced exclusion accepted");}
   for(auto& n:m.tensor_names())m.tensor(n)->type=GGML_TYPE_F32;
   for(unsigned fmt=3;fmt<6;++fmt){bool no=false;try{tsq::validate_storage(m,static_cast<tsq::Format>(fmt));}catch(const std::exception&){no=true;}check(no,"K format accepts F32 fallback");
    std::vector<std::pair<ggml_tensor*,int64_t>> saved;const ggml_type kinds[]={GGML_TYPE_Q6_K,GGML_TYPE_Q5_K,GGML_TYPE_Q4_K};
    for(auto& n:m.tensor_names())if(ts::reduced_weight_name(n)){auto t=m.tensor(n);saved.push_back({t,t->ne[0]});t->ne[0]=256;t->type=kinds[fmt-3];}
    tsq::validate_storage(m,static_cast<tsq::Format>(fmt));check(true,"K complete descriptor format");
    auto t=saved.front().first;t->ne[0]=192;no=false;try{tsq::validate_storage(m,static_cast<tsq::Format>(fmt));}catch(const std::exception&){no=true;}check(no,"K invalid row accepted");
    for(auto entry:saved){entry.first->ne[0]=entry.second;entry.first->type=GGML_TYPE_F32;}
   }
   tsq::validate_storage(m,tsq::Format::F32);auto weight=m.tensor("head.out.weight");auto saved=weight->type;weight->type=GGML_TYPE_F16;bool rejected=false;try{tsq::validate_storage(m,tsq::Format::F32);}catch(const std::exception&){rejected=true;}weight->type=saved;check(rejected,"mixed storage accepted");
  }
  if(!real){for(size_t p:{size_t(0),size_t(8),size_t(12),size_t(16),size_t(24),size_t(112),size_t(116),size_t(152),size_t(156),size_t(192),size_t(288+11),size_t(288+11+8),size_t(288+11+16),size_t(288+11+24)}){auto bad=bytes;bad[p]^=255;reject(bad);}auto bad=bytes;bad.pop_back();reject(bad);bad=bytes;bad.push_back(0);reject(bad);
   std::string out=std::string(TS_TEST_TMP_DIR)+"/eval-result-"+std::to_string(getpid());unlink(out.c_str());tsq::publish(out,result);check(tsq::read_file(out)==result,"atomic result");bool exists=false;try{tsq::publish(out,result);}catch(const std::exception&){exists=true;}check(exists,"existing output overwritten");std::string alias=out+"-link";unlink(alias.c_str());check(symlink(out.c_str(),alias.c_str())==0,"symlink setup");exists=false;try{tsq::publish(alias,result);}catch(const std::exception&){exists=true;}check(exists,"symlink output accepted");unlink(alias.c_str());unlink(out.c_str());
  }
  ++cases;std::cout<<"DRIVER_TRANSPORT "<<name<<" PASS\n";
 }
 std::cout<<"PASS eval codec cases="<<cases<<" checks="<<checks<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
