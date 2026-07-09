#pragma once
#include <string>

namespace pi::core {
struct PiResult {
  std::string digits;
  double compute_time_ms;
  int iterations;
};  // struct PiResult
}  // namespace pi::core
