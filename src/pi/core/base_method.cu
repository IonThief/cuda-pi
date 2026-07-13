#include <cuda_runtime.h>

#include "pi/core/base_method.cuh"

namespace pi::core {

PiResult BaseMethod::compute() {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  PiResult result = this->compute_impl();
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  result.compute_time_ms = static_cast<double>(ms);

  return result;
}

}  // namespace pi::core
