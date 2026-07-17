#pragma once
#include "pi/core/base_method.cuh"

namespace pi::method {

class GaussLegendre : public pi::core::BaseMethod {
 private:
  int iterations_;

 public:
  explicit GaussLegendre(int iterations);

 protected:
  pi::core::PiResult compute_impl() override;
};

}  // namespace pi::method
