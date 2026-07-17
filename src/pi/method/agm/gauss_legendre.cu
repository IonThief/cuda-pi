#include <cuda_runtime.h>

#include <cmath>

#include "pi/method/agm/gauss_legendre.cuh"

namespace {

__global__ void gauss_legendre_kernel(double* result, int iterations) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    double a = 1.0;
    double b = 1.0 / sqrt(2.0);
    double t = 0.25;
    double p = 1.0;

    for (int i = 0; i < iterations; ++i) {
      double a_next = (a + b) / 2.0;
      double b_next = sqrt(a * b);
      double t_next = t - p * (a - a_next) * (a - a_next);
      double p_next = 2.0 * p;

      a = a_next;
      b = b_next;
      t = t_next;
      p = p_next;
    }
    *result = ((a + b) * (a + b)) / (4.0 * t);
  }
}

}  // namespace

namespace pi::method {
GaussLegendre::GaussLegendre(int iterations) : iterations_(iterations) {}

pi::core::PiResult GaussLegendre::compute_impl() {
  double* d_result = nullptr;
  cudaMalloc(&d_result, sizeof(double));

  // 1 thread in 1 block, because the algorithm is sequential.
  // there will be another version that use parallel threads to compute more
  // digits of PI
  gauss_legendre_kernel<<<1, 1>>>(d_result, iterations_);
  cudaDeviceSynchronize();

  double final_pi = 0.0;
  cudaMemcpy(&final_pi, d_result, sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_result);

  return pi::core::PiResult{
      final_pi,
      0.0,
      iterations_,
  };
}

}  // namespace pi::method
