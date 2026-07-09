#pragma once
#include "pi/core/types.cuh"

namespace pi::core {

class BaseMethod {
 public:
  virtual ~BaseMethod() = default;

  virtual PiResult compute() = 0;
};  // class BaseMethod

}  // namespace pi::core
