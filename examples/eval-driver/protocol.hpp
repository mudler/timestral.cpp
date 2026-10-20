#pragma once
#include "forecaster.hpp"
#include <array>
#include <string>
#include <vector>
namespace tsq {
constexpr size_t limit=size_t(1)<<30;
using Digest=std::array<unsigned char,32>;
enum class Format : uint32_t { F32=0,F16=1,Q8_0=2,Q6_K=3,Q5_K=4,Q4_K=5 };
Format parse_format(const std::string&);
struct Request {ts::ForecastRequest request;std::string id;Digest model{},fixture_input{};
 uint32_t version=1;Format format=Format::F32;bool capture_anchors=true;};
struct Identity {Digest model{},executable{},source{},input{};std::string commit;};
Digest hash_bytes(const void*,size_t);
Digest hash_file(const std::string&);
Digest unhex(const std::string&);
std::vector<unsigned char> read_file(const std::string&);
Request decode(const std::vector<unsigned char>&);
std::vector<unsigned char> result(const Request&,const Identity&,const ts::Forecast*,const ts::ForecastAnchors*,const std::string& diagnostic={});
// Internal storage validator is separate so synthetic tests cannot weaken the
// production pinned-209M geometry/options check.
void validate_storage(const ts::Model&,Format);
void validate_provenance(const std::string& path,Format);
void validate_model(const ts::Model&,Format=Format::F32);
void publish(const std::string&,const std::vector<unsigned char>&);
}
