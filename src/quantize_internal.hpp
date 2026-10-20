#pragma once
#include "quantize.hpp"
#include <functional>
namespace ts::quantize_detail {
// Internal test seam, never a CLI option. Empty in production.
struct Hooks {
    std::function<void()> before_publish;
    bool fail_after_snapshot=false, fail_after_write=false, fail_directory_sync=false;
    bool force_link_publish=false;
};
QuantizeResult run(const std::string&,const std::string&,QuantizeType,const Hooks&);
}
