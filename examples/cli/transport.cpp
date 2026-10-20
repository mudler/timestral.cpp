#include "transport.hpp"
#include "timestral_capi.h"
#include "timestral.h"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
struct InputError:std::runtime_error {using std::runtime_error::runtime_error;};
using Options=std::map<std::string,std::vector<std::string>>;
int number(const std::string& s) {
    int n=0;auto r=std::from_chars(s.data(),s.data()+s.size(),n);
    if(r.ec!=std::errc{} || r.ptr!=s.data()+s.size() || n<=0)throw InputError("expected positive integer: "+s);
    return n;
}
size_t product(size_t a,size_t b){if(b && a>SIZE_MAX/b)throw InputError("input size overflow");return a*b;}
std::string escape(const std::string& s){std::ostringstream o;o<<'"';for(unsigned char c:s){if(c=='"'||c=='\\')o<<'\\'<<char(c);else if(c<32)o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;else o<<char(c);}o<<'"';return o.str();}
struct Csv {size_t rows=0,cols=0;std::vector<double> values;};
Csv read_csv(const std::string& path) {
    std::ifstream in(path);if(!in)throw InputError("cannot read input: "+path);
    Csv result;std::string line;
    while(std::getline(in,line)) {
        if(!line.empty() && line.back()=='\r')line.pop_back();
        size_t begin=0,col=0;std::vector<double> row;
        while(true) {
            size_t end=line.find(',',begin);std::string token=line.substr(begin,end==std::string::npos?end:end-begin);
            auto left=token.find_first_not_of(" \t");token=left==std::string::npos?"":token.substr(left,token.find_last_not_of(" \t")-left+1);
            std::string lower=token;for(char& c:lower)if(c>='A'&&c<='Z')c=char(c-'A'+'a');
            double value=std::numeric_limits<double>::quiet_NaN();
            if(!token.empty() && lower!="nan") {
                auto first=token.data();if(*first=='+')++first;
                auto parsed=std::from_chars(first,token.data()+token.size(),value,std::chars_format::general);
                if(first==token.data()+token.size() || parsed.ec!=std::errc{} || parsed.ptr!=token.data()+token.size() || !std::isfinite(value))
                    throw InputError(path+":"+std::to_string(result.rows+1)+":"+std::to_string(col+1)+": invalid finite number or missing value");
            }
            row.push_back(value);++col;if(end==std::string::npos)break;begin=end+1;
        }
        if(!result.rows)result.cols=col;
        if(col!=result.cols)throw InputError(path+":"+std::to_string(result.rows+1)+":"+std::to_string(col+1)+": ragged CSV row");
        if(result.rows>=INT32_MAX || col>INT32_MAX)throw InputError(path+": CSV dimensions exceed ABI");
        result.values.insert(result.values.end(),row.begin(),row.end());++result.rows;
    }
    if(in.bad())throw InputError("input read failure: "+path);
    if(!result.rows)throw InputError(path+": empty CSV");return result;
}
template<class T> std::vector<T> pack(const std::vector<std::string>& paths,size_t rows,size_t& cols) {
    std::vector<T> result;cols=0;
    for(const auto& path:paths) {
        auto f=read_csv(path);if(f.rows!=rows)throw InputError(path+": wrong number of time rows");
        if(!cols)cols=f.cols;if(cols!=f.cols)throw InputError(path+": inconsistent batch width");
        size_t offset=result.size();size_t n=product(rows,cols);if(n>result.max_size()-offset)throw InputError("input vector size overflow");result.resize(offset+n);
        for(size_t t=0;t<rows;++t)for(size_t v=0;v<cols;++v) {
            double x=f.values[t*cols+v];
            if(std::isfinite(x) && (x>double(std::numeric_limits<T>::max()) || x<double(std::numeric_limits<T>::lowest())))
                throw InputError(path+":"+std::to_string(t+1)+":"+std::to_string(v+1)+": number outside input type range");
            T y=static_cast<T>(x);
            if(std::isfinite(x) && !std::isfinite(y))throw InputError(path+":"+std::to_string(t+1)+":"+std::to_string(v+1)+": number outside input type range");
            result[offset+v*rows+t]=y;
        }
    }
    return result;
}
Options options(int argc,char** argv,bool info) {
    Options o;
    for(int i=2;i<argc;++i) {
        std::string k=argv[i];bool flag=k=="--json" || (!info && (k=="--no-symmetric"||k=="--no-clip"));
        bool valued=k=="--model" || (!info && (k=="--context"||k=="--past-cov"||k=="--known-cov"||k=="--cat-past"||k=="--cat-known"||k=="--horizon"||k=="--threads"));
        if(!flag&&!valued)throw InputError("unknown option: "+k);
        bool repeated=k=="--context"||k=="--past-cov"||k=="--known-cov"||k=="--cat-past"||k=="--cat-known";
        if(!repeated && o.count(k))throw InputError("duplicate option: "+k);
        if(flag)o[k].push_back("");else {if(++i>=argc || std::string(argv[i]).rfind("--",0)==0)throw InputError("missing value: "+k);o[k].push_back(argv[i]);}
    }
    if(!o.count("--model") || o["--model"][0].empty())throw InputError("--model is required");return o;
}
void levels(std::ostream& o,const std::vector<double>& q){o<<'['<<std::setprecision(std::numeric_limits<double>::max_digits10);for(size_t i=0;i<q.size();++i)o<<(i?",":"")<<q[i];o<<']';}
int forecast(const Options& o) {
    if(!o.count("--context")||!o.count("--horizon"))throw InputError("--context and --horizon are required");
    int H=number(o.at("--horizon")[0]),threads=o.count("--threads")?number(o.at("--threads")[0]):1;
    if(threads>1024)throw InputError("threads exceed supported range");
    size_t B=o.at("--context").size();if(B>INT32_MAX)throw InputError("batch exceeds ABI");
    auto first=read_csv(o.at("--context")[0]);size_t T=first.rows,V=0;
    auto context=pack<float>(o.at("--context"),T,V);
    std::vector<float> past,known;std::vector<double> cp,ck;size_t P=0,K=0,CP=0,CK=0;
    auto files=[&](const char* key){auto it=o.find(key);if(it==o.end())return std::vector<std::string>{};if(it->second.size()!=B)throw InputError(std::string(key)+": expected exactly one file per context");return it->second;};
    if(T>SIZE_MAX-size_t(H))throw InputError("time dimensions overflow");
    past=pack<float>(files("--past-cov"),T,P);known=pack<float>(files("--known-cov"),T+H,K);
    cp=pack<double>(files("--cat-past"),T,CP);ck=pack<double>(files("--cat-known"),T+H,CK);
    using Handle=std::unique_ptr<timestral_ctx,decltype(&timestral_capi_free)>;
    Handle handle(timestral_capi_load(o.at("--model")[0].c_str(),threads),timestral_capi_free);
    if(!handle)throw std::runtime_error(timestral_capi_last_error(nullptr));
    timestral_forecast_request r{};r.context=context.data();r.B=int32_t(B);r.V=int32_t(V);r.T=int32_t(T);r.prediction_length=H;
    r.past_cov=past.empty()?nullptr:past.data();r.c_past=int32_t(P);r.known_cov=known.empty()?nullptr:known.data();r.c_known=int32_t(K);
    r.cat_past=cp.empty()?nullptr:cp.data();r.cc_past=int32_t(CP);r.cat_known=ck.empty()?nullptr:ck.data();r.cc_known=int32_t(CK);
    r.symmetric_averaging=o.count("--no-symmetric")?0:-1;r.clip_nonnegative=o.count("--no-clip")?0:-1;
    int32_t Q=0;std::unique_ptr<float,decltype(&timestral_capi_free_buffer)> out(timestral_capi_forecast(handle.get(),&r,&Q),timestral_capi_free_buffer);
    if(!out)throw std::runtime_error(timestral_capi_last_error(handle.get()));
    size_t n=product(product(product(B,V),size_t(H)),size_t(Q));std::vector<double> q;for(int i=0;i<Q;++i)q.push_back(timestral_capi_quantile_level(handle.get(),i));
    for(size_t i=0;i<n;++i)if(!std::isfinite(out.get()[i]))throw std::runtime_error("nonfinite output");
    std::ostringstream text;text.imbue(std::locale::classic());
    if(o.count("--json")) {
        text<<"{\"shape\":["<<B<<','<<V<<','<<H<<','<<Q<<"],\"quantiles\":";levels(text,q);text<<",\"values\":["<<std::setprecision(std::numeric_limits<float>::max_digits10);
        for(size_t i=0;i<n;++i)text<<(i?",":"")<<out.get()[i];text<<"]}\n";
    } else {
        text<<"series,variate,step"<<std::setprecision(std::numeric_limits<double>::max_digits10);for(double level:q)text<<",q_"<<level;text<<'\n'<<std::setprecision(std::numeric_limits<float>::max_digits10);
        for(size_t b=0;b<B;++b)for(size_t v=0;v<V;++v)for(int t=0;t<H;++t){text<<b<<','<<v<<','<<t;for(int i=0;i<Q;++i)text<<','<<out.get()[((b*V+v)*H+t)*Q+i];text<<'\n';}
    }
    std::cout<<text.str();std::cout.flush();if(!std::cout)throw std::runtime_error("output write failed");return 0;
}
int info(const Options& o) {
    auto m=ts::Model::load(o.at("--model")[0]);const auto& c=m.config();
    std::ostringstream text;text.imbue(std::locale::classic());
    if(o.count("--json")){text<<"{\"version\":"<<escape(ts::version())<<",\"name\":"<<escape(c.name)<<",\"patch_size\":"<<c.patch_size<<",\"d_model\":"<<c.d_model<<",\"num_heads\":"<<c.num_heads<<",\"layers\":"<<c.n_layers<<",\"context_length\":"<<c.context_length<<",\"block_size\":"<<c.block_size<<",\"quantiles\":";levels(text,c.quantile_knots);text<<"}\n";}
    else {text<<"timestral "<<ts::version()<<"\nmodel: "<<c.name<<"\npatch_size: "<<c.patch_size<<"\nd_model: "<<c.d_model<<"\nnum_heads: "<<c.num_heads<<"\nlayers: "<<c.n_layers<<"\ncontext_length: "<<c.context_length<<"\nblock_size: "<<c.block_size<<"\nquantiles: ";levels(text,c.quantile_knots);text<<'\n';}
    std::cout<<text.str();std::cout.flush();if(!std::cout)throw std::runtime_error("output write failed");return 0;
}
}
int cli_transport(int argc,char** argv) {
    try {std::string command=argv[1];if(command!="forecast"&&command!="info")throw InputError("unsupported command: "+command);auto o=options(argc,argv,command=="info");return command=="info"?info(o):forecast(o);}
    catch(const InputError& e){std::cerr<<e.what()<<'\n';return 2;}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
