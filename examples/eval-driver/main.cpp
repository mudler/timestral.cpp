#include "protocol.hpp"
#include "backend.hpp"
#include <charconv>
#include <iostream>
#include <map>
#include <sys/stat.h>
#include <unistd.h>
#ifndef TSQ_RUNTIME_COMMIT
#error Explicit driver build identity required
#endif
#ifndef TSQ_SOURCE_SHA256
#error Explicit source snapshot identity required
#endif
namespace {
struct Stamp {struct stat s{};explicit Stamp(const std::string& p){if(lstat(p.c_str(),&s)||!S_ISREG(s.st_mode))throw std::runtime_error("regular nonsymlink file required");}};
bool same(const Stamp& a,const Stamp& b){return a.s.st_dev==b.s.st_dev&&a.s.st_ino==b.s.st_ino&&a.s.st_size==b.s.st_size&&a.s.st_mtim.tv_sec==b.s.st_mtim.tv_sec&&a.s.st_mtim.tv_nsec==b.s.st_mtim.tv_nsec&&a.s.st_ctim.tv_sec==b.s.st_ctim.tv_sec&&a.s.st_ctim.tv_nsec==b.s.st_ctim.tv_nsec;}
}
int main(int argc,char** argv){
 std::string output;tsq::Request request;tsq::Identity identity;identity.commit=TSQ_RUNTIME_COMMIT;
 try {
  identity.source=tsq::unhex(TSQ_SOURCE_SHA256);std::map<std::string,std::string> args;
  for(int i=1;i<argc;i+=2){if(i+1>=argc)throw std::runtime_error("missing option value");std::string k=argv[i];if(k!="--input"&&k!="--output"&&k!="--model"&&k!="--format"&&k!="--threads")throw std::runtime_error("unknown option");if(!args.emplace(k,argv[i+1]).second)throw std::runtime_error("duplicate option");}
  for(auto k:{"--input","--output","--model","--format","--threads"})if(!args.count(k))throw std::runtime_error("required option missing");
  output=args.at("--output");struct stat exists{};if(lstat(output.c_str(),&exists)==0||errno!=ENOENT)throw std::runtime_error("refuse existing output");
  if(args.at("--format")!="f32")throw std::runtime_error("F32 driver only");int threads=0;auto value=args.at("--threads");auto parsed=std::from_chars(value.data(),value.data()+value.size(),threads);if(parsed.ec!=std::errc{}||parsed.ptr!=value.data()+value.size()||threads<1||threads>1024)throw std::runtime_error("invalid threads");
  char executable[4096];ssize_t n=readlink("/proc/self/exe",executable,sizeof(executable)-1);if(n<=0||n>=ssize_t(sizeof(executable)-1))throw std::runtime_error("cannot identify executable");executable[n]=0;Stamp exe(executable),input(args.at("--input")),model(args.at("--model"));
  identity.executable=tsq::hash_file(executable);identity.model=tsq::hash_file(args.at("--model"));auto bytes=tsq::read_file(args.at("--input"));identity.input=tsq::hash_bytes(bytes.data(),bytes.size());request=tsq::decode(bytes);if(request.model!=identity.model)throw std::runtime_error("model SHA256 mismatch");
  auto m=ts::Model::load(args.at("--model"));tsq::validate_model(m);ts::set_cpu_threads(threads);ts::Forecaster fc(m);ts::ForecastTrace trace;trace.capture_initial_anchors=true;trace.capture_blocks=false;auto forecast=fc.forecast(request.request,&trace);if(!trace.initial_anchors)throw std::runtime_error("missing F32 anchors");
  if(!same(model,Stamp(args.at("--model")))||identity.model!=tsq::hash_file(args.at("--model"))||!same(exe,Stamp(executable))||identity.executable!=tsq::hash_file(executable)||!same(input,Stamp(args.at("--input")))||identity.input!=tsq::hash_file(args.at("--input")))throw std::runtime_error("input/model/executable changed");
  tsq::publish(output,tsq::result(request,identity,&forecast,&*trace.initial_anchors));return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';if(!output.empty()){try{tsq::publish(output,tsq::result(request,identity,nullptr,nullptr,e.what()));}catch(const std::exception& write){std::cerr<<"diagnostic publication failed: "<<write.what()<<'\n';}}return 1;}
}
