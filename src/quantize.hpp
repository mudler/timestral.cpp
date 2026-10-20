#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
namespace ts {
enum class QuantizeType { Q6_K, Q5_K, Q4_K };
enum class QuantizePublication { Durable, PublishedDurabilityUnconfirmed };
struct QuantizeResult {
    uint64_t input_bytes=0, output_bytes=0;
    size_t converted_tensors=0, copied_tensors=0;
    std::string input_sha256, output_sha256;
    QuantizePublication publication=QuantizePublication::Durable;
};
// Optional Linux/POSIX tool library, not the inference core. Throws Error before
// publication; never overwrites an existing destination. A directory fsync failure
// after publication returns PublishedDurabilityUnconfirmed rather than rollback.
QuantizeResult quantize_gguf(const std::string& input,const std::string& output,QuantizeType type);
}
