#pragma once

namespace cuda_pi {

constexpr int THREADS_PER_BLOCK = 256;

__global__ void reduce_pi_kernel(double* d_partial_sums,
                                 unsigned long long num_intervals);

double calculate(unsigned long long num_intervals);

}  // namespace cuda_pi
