#pragma once
#include "pi/core/base_method.cuh"

namespace pi::method {

class Nilakantha : public pi::core::BaseMethod {
 private:
  unsigned long long terms_;

 public:
  explicit Nilakantha(unsigned long long terms);

 protected:
  pi::core::PiResult compute_impl() override;
};  // class Nilakantha

}  // namespace pi::method
