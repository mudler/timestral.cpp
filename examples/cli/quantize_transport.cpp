#include "quantize_transport.hpp"
#include "quantize.hpp"
#include <iostream>
#include <string>
int cli_quantize(int argc,char** argv){
    if(argc!=5){std::cerr<<"usage: timestral-cli quantize <in.gguf> <out.gguf> q6_k|q5_k|q4_k\n";return 2;}
    const std::string name=argv[4];ts::QuantizeType type;
    if(name=="q6_k")type=ts::QuantizeType::Q6_K;
    else if(name=="q5_k")type=ts::QuantizeType::Q5_K;
    else if(name=="q4_k")type=ts::QuantizeType::Q4_K;
    else {std::cerr<<"quantize: expected q6_k, q5_k, or q4_k\n";return 2;}
    try {
        auto r=ts::quantize_gguf(argv[2],argv[3],type);
        if(r.publication!=ts::QuantizePublication::Durable){
            std::cerr<<"quantize: output exists; published but directory durability is unconfirmed; no rollback occurred\n";
            return 1;
        }
        std::cout<<"{\"input_sha256\":\""<<r.input_sha256<<"\",\"output_sha256\":\""<<r.output_sha256
                 <<"\",\"input_bytes\":"<<r.input_bytes<<",\"output_bytes\":"<<r.output_bytes
                 <<",\"converted_tensors\":"<<r.converted_tensors<<",\"copied_tensors\":"<<r.copied_tensors
                 <<",\"type\":\""<<name<<"\"}\n";
        std::cout.flush();if(!std::cout){std::cerr<<"quantize: output file exists; writing the success report failed\n";return 1;}
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
