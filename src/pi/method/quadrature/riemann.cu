#include <cuda_runtime.h>

#include <numeric>
#include <string>
#include <vector>

#include "pi/method/quadrature/riemann.cuh"

namespace {
constexpr int THREADS = 256;

__global__ void riemann_kernel(double* block_results,
                               unsigned long long total_steps) {
  __shared__ double shared_sums[THREADS];

  unsigned int local_id = threadIdx.x;
  unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int total_threads = blockDim.x * gridDim.x;

  double step_size = 1.0 / static_cast<double>(total_steps);
  double local_sum = 0.0;

  for (unsigned long long i = global_id; i < total_steps; i += total_threads) {
    double x = (static_cast<double>(i) + 0.5) * step_size;
    local_sum += 1.0 / (1.0 + (x * x));
  }

  shared_sums[local_id] = local_sum;
  __syncthreads();

  for (unsigned int step = blockDim.x / 2; step > 0; step /= 2) {
    if (local_id < step) {
      shared_sums[local_id] += shared_sums[local_id + step];
    }
    __syncthreads();
  }
  if (local_id == 0) {
    block_results[blockIdx.x] = shared_sums[0];
  }
}

}  // namespace

namespace pi::method {
Riemann::Riemann(unsigned long long num_intervals)
    : num_intervals_(num_intervals) {}

pi::core::PiResult Riemann::compute_impl() {
  int num_blocks = 512;
  size_t alloc_size = num_blocks * sizeof(double);
  std::vector<double> h_partial_sums(num_blocks);
  double* d_partial_sums = nullptr;

  cudaMalloc(&d_partial_sums, alloc_size);
  riemann_kernel<<<num_blocks, THREADS>>>(d_partial_sums, num_intervals_);
  cudaDeviceSynchronize();

  cudaMemcpy(h_partial_sums.data(), d_partial_sums, alloc_size,
             cudaMemcpyDeviceToHost);
  cudaFree(d_partial_sums);

  double total_sum =
      std::accumulate(h_partial_sums.begin(), h_partial_sums.end(), 0.0);
  double step = 1.0 / static_cast<double>(num_intervals_);
  double final_pi = total_sum * step * 4.0;

  return pi::core::PiResult{
      std::to_string(final_pi),
      0.0,
      static_cast<int>(num_intervals_),
  };
}

}  // namespace pi::method
