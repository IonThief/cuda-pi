#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <numeric>
#include <vector>

#include "pi/method/stochastic/monte_carlo.cuh"

namespace {
constexpr int THREADS = 256;
constexpr int BLOCKS = 512;

__global__ void monte_carlo_kernel(unsigned long long seed,
                                   unsigned long long points_per_thread,
                                   unsigned long long* block_hits) {
  __shared__ unsigned long long shared_hits[THREADS];

  unsigned int local_id = threadIdx.x;
  unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;

  curandState_t local_state;
  curand_init(seed, global_id, 0, &local_state);

  unsigned long long local_hits = 0;
  for (unsigned long long i = 0; i < points_per_thread; ++i) {
    float x = curand_uniform(&local_state);
    float y = curand_uniform(&local_state);
    if (((x * x) + (y * y)) <= 1.0f) local_hits++;
  }

  shared_hits[local_id] = local_hits;
  __syncthreads();

  // Tree reduction
  for (unsigned int step = blockDim.x / 2; step > 0; step /= 2) {
    if (local_id < step) shared_hits[local_id] += shared_hits[local_id + step];
    __syncthreads();
  }

  if (local_id == 0) block_hits[blockIdx.x] = shared_hits[0];
}

}  // namespace

namespace pi::method {

MonteCarlo::MonteCarlo(unsigned long long total_points)
    : total_points_(total_points) {}

pi::core::PiResult MonteCarlo::compute_impl() {
  int total_threads = BLOCKS * THREADS;
  unsigned long long points_per_thread = total_points_ / total_threads;
  unsigned long long actual_total = points_per_thread * total_threads;

  unsigned long long* d_block_hits;
  cudaMalloc(&d_block_hits, BLOCKS * sizeof(unsigned long long));

  monte_carlo_kernel<<<BLOCKS, THREADS>>>(1234ULL, points_per_thread,
                                          d_block_hits);
  cudaDeviceSynchronize();

  std::vector<unsigned long long> h_block_hits(BLOCKS);
  cudaMemcpy(h_block_hits.data(), d_block_hits,
             BLOCKS * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  cudaFree(d_block_hits);

  unsigned long long total_hits =
      std::accumulate(h_block_hits.begin(), h_block_hits.end(), 0ULL);
  double final_pi =
      4.0 * static_cast<double>(total_hits) / static_cast<double>(actual_total);

  return pi::core::PiResult{
      final_pi,
      0.0,
      static_cast<int>(actual_total),
  };
}
}  // namespace pi::method
