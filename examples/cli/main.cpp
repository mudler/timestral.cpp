// M6c measurement entry point only; the general CLI follows in M8.
#include "timestral.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <sys/resource.h>

static std::string cpu_model() {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("model name", 0) == 0) return line.substr(line.find(':') + 2);
    return "unknown";
}
static std::string quoted(const std::string& s) {
    std::string out="\"";
    for (char c:s) { if (c=='"' || c=='\\') out+='\\'; if (c>=32) out+=c; }
    return out+'"';
}
int main(int argc, char** argv) {
    try {
        if (argc<2 || std::string(argv[1])!="bench")
            throw std::runtime_error("usage: timestral-cli bench --model FILE [--context-len 32768 --horizon 1024 --variates 1 --threads 1 --repeat 5 --warmup 1]");
        std::map<std::string,std::string> opts={{"--model",""},{"--context-len","32768"},{"--horizon","1024"},{"--variates","1"},{"--threads","1"},{"--repeat","5"},{"--warmup","1"}};
        for (int i=2;i<argc;i+=2) {
            if (!opts.count(argv[i]) || i+1==argc) throw std::runtime_error("unknown option or missing value");
            opts[argv[i]]=argv[i+1];
        }
        auto number=[&](const char* key, bool zero=false) {
            const auto& s=opts.at(key);
            if(s.empty() || s.find_first_not_of("0123456789")!=std::string::npos) throw std::runtime_error("invalid integer option");
            auto n=std::stoull(s);
            if ((!zero && n==0) || n>std::numeric_limits<int>::max()) throw std::runtime_error("integer option out of range");
            return size_t(n);
        };
        ts::ForecastRequest request;
        request.batch=1; request.targets=number("--variates"); request.length=number("--context-len"); request.horizon=number("--horizon");
        const auto threads=number("--threads"), repeat=number("--repeat"), warmup=number("--warmup",true);
        if(opts["--model"].empty()) throw std::runtime_error("--model is required");
        ts::set_cpu_threads(int(threads));
        auto model=ts::Model::load(opts["--model"]);
        ts::Forecaster fc(model);
        request.context.resize(request.targets*request.length);
        double input_sum=0;
        for(size_t v=0;v<request.targets;++v) for(size_t i=0;i<request.length;++i) {
            float x=float(int((i*17+v*13)%257)-128)/64.0f;
            request.context[v*request.length+i]=x; input_sum+=x;
        }
        std::vector<double> samples;
        double checksum=0;
        for(size_t i=0;i<warmup+repeat;++i) {
            auto start=std::chrono::steady_clock::now();
            auto result=fc.forecast(request);
            double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            checksum=0;
            for(float x:result.quantiles) { if(!std::isfinite(x)) throw std::runtime_error("nonfinite output"); checksum+=x; }
            if(i>=warmup) samples.push_back(seconds);
            std::cerr<<(i<warmup?"warmup":"sample")<<" "<<i<<" seconds="<<seconds<<"\n";
        }
        auto sorted=samples; std::sort(sorted.begin(),sorted.end());
        double median=(sorted[(repeat-1)/2]+sorted[repeat/2])/2;
        rusage usage{}; if(getrusage(RUSAGE_SELF,&usage)) throw std::runtime_error("getrusage failed");
#ifdef __APPLE__
        auto rss=usage.ru_maxrss;
#else
        auto rss=usage.ru_maxrss*1024LL;
#endif
        std::cout<<std::setprecision(12)<<"{\"backend\":\"native-cpu\",\"model\":"<<quoted(opts["--model"])
                 <<",\"cpu\":"<<quoted(cpu_model())<<",\"context_len\":"<<request.length<<",\"horizon\":"<<request.horizon
                 <<",\"variates\":"<<request.targets<<",\"threads\":"<<threads<<",\"repeat\":"<<repeat<<",\"warmup\":"<<warmup
                 <<",\"input\":\"dyadic-v1\",\"input_sum\":"<<input_sum<<",\"output_sum\":"<<checksum
                 <<",\"median_seconds\":"<<median<<",\"peak_rss_bytes\":"<<rss<<",\"samples_seconds\":[";
        for(size_t i=0;i<samples.size();++i) std::cout<<(i?",":"")<<samples[i];
        std::cout<<"]}\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<"\n"; return 1; }
}
