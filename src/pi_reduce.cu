#include <cuda_runtime.h>

#include <numeric>
#include <vector>

#include "pi_reduce.cuh"

namespace cuda_pi {
__global__ void reduce_pi_kernel(double* d_partial_sums,
                                 unsigned long long num_intervals) {
  __shared__ double sdata[THREADS_PER_BLOCK];

  unsigned int tid = threadIdx.x;  // local thread id
  unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int grid_size = blockDim.x * gridDim.x;

  double step = 1.0 / static_cast<double>(num_intervals);
  double local_sum = 0.0;

  for (unsigned long long i = global_tid; i < num_intervals; i += grid_size) {
    double x = (static_cast<double>(i) + 0.5) * step;
    local_sum += 1.0 / (1.0 + x * x);  // derivative value at mid-point
  }

  sdata[tid] = local_sum;
  __syncthreads();

  // Parallel Tree Reduction to make use of all threads
  // summing sequentially will take O(N), later threads will have to wait
  // for its turn to be added
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    d_partial_sums[blockIdx.x] = sdata[0];
  }
}

double calculate(unsigned long long num_intervals) {
  int num_blocks = 512;
  size_t alloc_size = num_blocks * sizeof(double);

  std::vector<double> h_partial_sums(num_blocks);
  double* d_partial_sums = nullptr;

  cudaMalloc(&d_partial_sums, alloc_size);

  reduce_pi_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_partial_sums,
                                                      num_intervals);
  cudaGetLastError();
  cudaDeviceSynchronize();
  cudaMemcpy(h_partial_sums.data(), d_partial_sums, alloc_size,
             cudaMemcpyDeviceToHost);
  cudaFree(d_partial_sums);

  double total_sum =
      std::accumulate(h_partial_sums.begin(), h_partial_sums.end(), 0.0);

  double step = 1.0 / static_cast<double>(num_intervals);
  return total_sum * step * 4.0;
}

}  // namespace cuda_pi
