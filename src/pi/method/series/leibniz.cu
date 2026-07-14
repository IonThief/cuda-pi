#include <cuda_runtime.h>

#include <numeric>
#include <vector>

#include "pi/method/series/leibniz.cuh"

namespace {
constexpr int THREADS = 256;
constexpr int BLOCKS = 512;

__global__ void leibniz_kernel(double* block_results,
                               unsigned long long total_terms) {
  __shared__ double shared_sums[THREADS];

  unsigned int local_id = threadIdx.x;
  unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int total_threads = blockDim.x * gridDim.x;

  double local_sum = 0.0;

  for (unsigned long long i = global_id; i < total_terms; i += total_threads) {
    double denom = 2.0 * static_cast<double>(i) + 1.0;
    double sign = (i & 1) == 0 ? 1.0 : -1.0;
    local_sum += sign / denom;
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

Leibniz::Leibniz(unsigned long long terms) : terms_(terms) {}

pi::core::PiResult Leibniz::compute_impl() {
  double* d_block_results;

  cudaMalloc(&d_block_results, BLOCKS * sizeof(double));

  leibniz_kernel<<<BLOCKS, THREADS>>>(d_block_results, terms_);
  cudaDeviceSynchronize();

  std::vector<double> h_block_results(BLOCKS);

  cudaMemcpy(h_block_results.data(), d_block_results, BLOCKS * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaFree(d_block_results);

  double total_sum =
      std::accumulate(h_block_results.begin(), h_block_results.end(), 0.0);

  double final_pi = total_sum * 4.0;

  return pi::core::PiResult{
      final_pi,
      0.0,
      static_cast<int>(terms_),
  };
}

}  // namespace pi::method
