#include "backend.hpp"
#include "common.hpp"
#include <iostream>
#include <thread>
using namespace ts;
void require(bool b){if(!b)throw Error("status test");}
template<class F>void rejects(F f){bool bad=false;try{f();}catch(...){bad=true;}require(bad);}
int main(){try{size_t count=0;for(int failure=0;failure<3;++failure){GraphRunner r;auto ctx=r.begin();auto x=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,1);ggml_set_input(x);auto y=ggml_dup(ctx,x);rejects([&]{r.read(y);});r.allocate({y});r.upload(x,{1});r.compute();require(r.read(y)==std::vector<float>{1});if(failure==0)rejects([&]{r.upload(x,{});});if(failure==1)rejects([&]{r.compute();});if(failure==2)rejects([&]{r.allocate({y});});rejects([&]{r.read(y);});ctx=r.begin();x=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,1);ggml_set_input(x);y=ggml_dup(ctx,x);r.allocate({y});r.upload(x,{2});r.compute();require(r.read(y)==std::vector<float>{2});++count;}
 for(int workers:{1,2,3,6}){CallbackStatus a,b;std::vector<std::thread>threads;for(int i=0;i<workers;++i)threads.emplace_back([&]{a.publish("worker failed");});for(auto&t:threads)t.join();require(a.failed()&&!b.failed());require(std::string(a.message())=="worker failed");a.publish("overwrite");require(std::string(a.message())=="worker failed");a.reset();require(!a.failed());a.slot.store(Publication::Reserved);require(a.failed()&&std::string(a.message())=="callback error publication incomplete");a.publish("loser");require(a.slot.load()==Publication::Reserved);++count;}
 std::cout<<"PASS graph status cases="<<count<<" stale-output/reset/publication/reserved/isolation\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
