#pragma once
#include "pi/core/base_method.cuh"

namespace pi::method {
class MonteCarlo : public pi::core::BaseMethod {
 private:
  unsigned long long total_points_;

 public:
  explicit MonteCarlo(unsigned long long total_points);

 protected:
  pi::core::PiResult compute_impl() override;
};  // class MonteCarlo

}  // namespace pi::method
