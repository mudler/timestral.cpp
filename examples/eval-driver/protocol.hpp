#pragma once
#include "forecaster.hpp"
#include <array>
#include <string>
#include <vector>
namespace tsq {
constexpr size_t limit=size_t(1)<<30;
using Digest=std::array<unsigned char,32>;
struct Request {ts::ForecastRequest request;std::string id;Digest model{},fixture_input{};};
struct Identity {Digest model{},executable{},source{},input{};std::string commit;};
Digest hash_bytes(const void*,size_t);
Digest hash_file(const std::string&);
Digest unhex(const std::string&);
std::vector<unsigned char> read_file(const std::string&);
Request decode(const std::vector<unsigned char>&);
std::vector<unsigned char> result(const Request&,const Identity&,const ts::Forecast*,const ts::ForecastAnchors*,const std::string& diagnostic={});
void validate_model(const ts::Model&);
void publish(const std::string&,const std::vector<unsigned char>&);
}
