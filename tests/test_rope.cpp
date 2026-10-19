#include "fixture.hpp"
#include "rope.hpp"
#include <iostream>
template<class F> void rejects(F f) {bool bad=false;try{f();}catch(const std::exception&){bad=true;}require(bad,"expected rejection");}
void units(const Model& m) {
 Rope rope(m);const size_t hd=m.config().head_dim();
 auto t=rope.gather({-1,0,1});require(t.cos.size()==3*hd,"gather shape");
 rejects([&]{rope.gather({INT64_MAX});});rejects([&]{rope.gather({INT64_MIN});});
 rejects([&]{rope.gather({int64_t(m.config().rope_table_pos_min)-1});});
 auto inv=rope.inv_freq();auto stored=static_cast<const float*>(m.tensor("rope.inv_freq")->data);
 require(std::memcmp(inv.data(),stored,inv.size()*4)==0,"stored inverse frequencies");
 require(inv.front()==1.0f,"inverse first frequency");
 for(size_t i=m.config().rope_dim/2;i<inv.size();++i)require(inv[i]==0,"partial frequency padding");
 require(rope.gather({}).cos.empty(),"empty gather");
 std::vector<float> q(3*2*hd);for(size_t i=0;i<q.size();++i) q[i]=float(i+1)/7;
 auto a=rope.rotate(q,t,1,3,2);auto b=rope.rotate(q,t,1,3,2);require(a==b,"repeat graph");
 auto one=rope.gather({1});rope.rotate(std::vector<float>(hd,1),one,1,1,1);
 require(rope.rotate(q,t,1,3,2)==a,"grow and shrink");
 for(size_t i=0;i<q.size();++i) {
   size_t d=i%hd, token=i/(2*hd), pair=d<hd/2?i+hd/2:i-hd/2;
   float rotated=d<hd/2?-q[pair]:q[pair];
   float product=q[i]*t.cos[token*hd+d];float second=rotated*t.sin[token*hd+d];
   float expected=product+second;
   require(tstest::compare_ulp({a[i]},{expected},2,"paired halves").ok,"rotation formula");
   if(d%(hd/2)>=m.config().rope_dim/2)require(a[i]==q[i],"unrotated tail");
 }
 rejects([&]{rope.rotate({}, {},0,0,0);});rejects([&]{rope.rotate(q,one,1,3,2);});
 GraphRunner r;rejects([&]{r.compute();});rejects([&]{r.read(nullptr);});
 auto ctx=r.begin();auto x=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,hd,2,3,1);
 auto wrong=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,hd,2,3,1);rejects([&]{apply_rope(ctx,x,wrong,wrong);});
 rejects([]{GraphRunner r(0);});rejects([&]{r.allocate({});});
 auto in=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,4);ggml_set_input(in);auto out=ggml_scale(ctx,in,2);
 rejects([&]{r.upload(in,{1,2,3,4});});r.allocate({out});
 rejects([&]{r.upload(in,{1});});rejects([&]{r.read(out);});r.upload(in,{1,2,3,4});r.compute();
 require(r.read(out)==std::vector<float>({2,4,6,8}),"backend copied output");
 rejects([&]{r.upload(out,{1,2,3,4});});

}
int main(int argc,char** argv) {try {
 const char* mr=getenv("TIMESTRAL_TEST_MODEL_ROOT");auto m=Model::load(std::string(mr?mr:TS_SOURCE_DIR "/tests/fixtures/tiny")+"/timestral-tiny-f32.gguf");units(m);
 if(argc>1 && std::string(argv[1])=="--unit") return 0;
 size_t rotations=0, zeros=0;
 int rc=run_component(argc,argv,[&](const Model& m,Fixture& f,const std::string& p){
   Rope rope(m);auto raw=f.get<int32_t>(p+"tf.positions",GGML_TYPE_I32);std::vector<int64_t> positions(raw.begin(),raw.end());auto tables=rope.gather(positions);
   auto id=gguf_find_key(f.b.g,"layers.dumped");require(id>=0,"layers.dumped missing");
   std::string layers=gguf_get_val_str(f.b.g,id);std::vector<size_t> expected;
   if(layers=="all")for(size_t l=0;l<m.config().n_layers;++l)expected.push_back(l);
   else {std::istringstream list(layers);std::string n;while(std::getline(list,n,','))expected.push_back(std::stoul(n));}
   require(expected==f.expected_layers,"dumped layers disagree with manifest");
   for(auto l:expected) {require(l<m.config().n_layers,"layer out of range");auto axes=layer_axes(m.config().block_schedule[l]);
    for(size_t j=0;j<axes.size();++j) if(axes[j]) {
     auto base=p+"layer."+std::to_string(l)+".attn."+std::to_string(j)+".";
     RopeTables dumped{f.f(base+"cos"),f.f(base+"sin")};
     require(tstest::compare_exact_f32(tables.cos,dumped.cos,(base+"cos").c_str()).ok,"gather cos");
     require(tstest::compare_exact_f32(tables.sin,dumped.sin,(base+"sin").c_str()).ok,"gather sin");
     for(size_t i=0;i<tables.sin.size();++i) if(tables.sin[i]==0 && std::signbit(tables.sin[i])!=std::signbit(dumped.sin[i]))++zeros;
     for(auto name:{"q","k"}) {
       auto n=base+name;auto t=f.tensor(n);auto q=f.f(n);
       auto got=rope.rotate(q,dumped,t->ne[3],t->ne[2],t->ne[1]);
       require(tstest::compare_ulp(got,f.f(n+"_rope"),2,(n+"_rope").c_str()).ok,"rotation");++rotations;
     }
    }
   }
 });std::cerr<<"ROPE rotations="<<rotations<<" signed_zero_differences="<<zeros<<'\n';return rc;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return std::string(e.what())=="HOST_SKIP"?77:1;}}
