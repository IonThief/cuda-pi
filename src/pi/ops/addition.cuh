#pragma once
#include "pi/core/types.cuh"

namespace pi::ops {

void add_big_float(const core::BigFloat* a, const core::BigFloat* b,
                   core::BigFloat* result);

void sub_big_float(const core::BigFloat* a, const core::BigFloat* b,
                   core::BigFloat* result);

}  // namespace pi::ops
