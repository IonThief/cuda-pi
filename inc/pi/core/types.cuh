#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

namespace pi::core {
struct PiResult {
  double value;
  double compute_time_ms;
  int iterations;
};  // struct PiResult

struct BigFloat {
  int sign;
  long long exponent;
  size_t num_limbs;
  uint32_t* limbs;
};  // struct BigFloat

inline void allocate_big_float(BigFloat* bf, size_t num_limbs) {
  bf->sign = 1;  // 1 for +ve, -1 for -ve
  bf->exponent = 0;
  bf->num_limbs = num_limbs;

  cudaError_t err =
      cudaMallocManaged(&(bf->limbs), num_limbs * sizeof(uint32_t));
  if (err != cudaSuccess) printf("CUDA Error: %s\n", cudaGetErrorString(err));
}

inline void free_big_float(BigFloat* bf) {
  if (bf->limbs) {
    cudaFree(bf->limbs);
    bf->limbs = nullptr;
  }
}
}  // namespace pi::core
