#pragma once
#include "pi/core/base_method.cuh"

namespace pi::method {

class Riemann : public pi::core::BaseMethod {
 private:
  unsigned long long num_intervals_;

 public:
  explicit Riemann(unsigned long long num_intervals);

  pi::core::PiResult compute_impl() override;
};

}  // namespace pi::method
