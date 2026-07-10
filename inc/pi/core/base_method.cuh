#pragma once
#include "pi/core/types.cuh"

namespace pi::core {
class BaseMethod {
 public:
  virtual ~BaseMethod() = default;

  PiResult compute();

 protected:
  virtual PiResult compute_impl() = 0;
};  // class BaseMethod

}  // namespace pi::core
