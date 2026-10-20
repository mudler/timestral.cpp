#pragma once
#include <variant>
#include <cstddef>
#include <stdexcept>
namespace ts {
// Pure block scheduler. The shell performs prediction and supplies BlockReady.
struct DecodePending { size_t remaining, offset; };
struct DecodeDone { size_t offset; };
using DecodeState = std::variant<DecodePending, DecodeDone>;
struct BlockReady { size_t patches; };
struct DecodeTransition { DecodeState state; const char* rule; };
inline DecodeTransition reduce_decode(const DecodeState& state, BlockReady event) {
 if (auto p=std::get_if<DecodePending>(&state)) {
  if (!event.patches || event.patches>p->remaining) throw std::invalid_argument("invalid decode completion");
  if(event.patches==p->remaining)return {DecodeDone{p->offset+event.patches},"D2-done"};
  return {DecodePending{p->remaining-event.patches,p->offset+event.patches},"D1-feedback"};
 }
 if(std::holds_alternative<DecodeDone>(state))throw std::invalid_argument("block after decode completion");
 throw std::logic_error("unreachable decode state");
}
}
