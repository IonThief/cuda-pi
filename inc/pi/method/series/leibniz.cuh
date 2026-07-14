#pragma once
#include "pi/core/base_method.cuh"

namespace pi::method {
class Leibniz : public pi::core::BaseMethod {
 private:
  unsigned long long terms_;

 public:
  explicit Leibniz(unsigned long long terms);

 protected:
  pi::core::PiResult compute_impl() override;

};  // class Leibniz
}  // namespace pi::method
