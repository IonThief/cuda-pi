#include <cuda_runtime.h>

#include <numeric>
#include <vector>

#include "pi/method/series/nilakantha.cuh"

namespace {
constexpr int THREADS = 256;
constexpr int BLOCKS = 512;

__global__ void nilakantha_kernel(double* block_results,
                                  unsigned long long total_terms) {
  __shared__ double shared_sums[THREADS];

  unsigned int local_id = threadIdx.x;
  unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int total_threads = blockDim.x * gridDim.x;

  double local_sum = 0.0;

  for (unsigned long long i = global_id + 1; i <= total_terms;
       i += total_threads) {
    double n = static_cast<double>(i);
    double d1 = 2.0 * n;
    double d2 = d1 + 1.0;
    double d3 = d1 + 2.0;

    double denom = d1 * d2 * d3;

    double sign = (i & 1) == 1 ? 1.0 : -1.0;

    local_sum += sign * (4.0 / denom);
  }

  shared_sums[local_id] = local_sum;
  __syncthreads();

  for (unsigned int step = blockDim.x / 2; step > 0; step /= 2) {
    if (local_id < step) shared_sums[local_id] += shared_sums[local_id + step];
    __syncthreads();
  }

  if (local_id == 0) block_results[blockIdx.x] = shared_sums[0];
}

}  // namespace

namespace pi::method {

Nilakantha::Nilakantha(unsigned long long terms) : terms_(terms) {}

pi::core::PiResult Nilakantha::compute_impl() {
  double* d_block_results;
  cudaMalloc(&d_block_results, BLOCKS * sizeof(double));

  nilakantha_kernel<<<BLOCKS, THREADS>>>(d_block_results, terms_);
  cudaDeviceSynchronize();

  std::vector<double> h_block_results(BLOCKS);
  cudaMemcpy(h_block_results.data(), d_block_results, BLOCKS * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaFree(d_block_results);

  double total_fractions =
      std::accumulate(h_block_results.begin(), h_block_results.end(), 0.0);

  double final_pi = 3.0 + total_fractions;

  return pi::core::PiResult{
      final_pi,
      0.0,
      static_cast<int>(terms_),
  };
}

}  // namespace pi::method
