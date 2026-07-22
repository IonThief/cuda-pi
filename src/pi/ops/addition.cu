#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include "pi/ops/addition.cuh"

namespace {

const uint8_t KILL = 0;
const uint8_t PROPAGATE = 1;
const uint8_t GENERATE = 2;

struct CarryOp {
  __device__ __forceinline__ uint8_t operator()(const uint8_t& prev,
                                                const uint8_t& current) const {
    return (current == PROPAGATE) ? prev : current;
  }
};

__global__ void block_add_kernel(const uint32_t* a, const uint32_t* b,
                                 uint32_t* result, uint8_t* block_states,
                                 uint8_t* prop_flags, size_t n) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;

  uint64_t sum = 0;
  uint8_t local_state = KILL;

  if (idx < n) {
    sum = static_cast<uint64_t>(a[idx]) + b[idx];
    local_state = (sum > 0xFFFFFFFF) ? GENERATE
                                     : ((sum == 0xFFFFFFFF) ? PROPAGATE : KILL);
  }

  typedef cub::BlockScan<uint8_t, 256> BlockScan;
  __shared__ typename BlockScan::TempStorage temp;
  uint8_t b_carry_in;

  BlockScan(temp).ExclusiveScan(local_state, b_carry_in, PROPAGATE, CarryOp());

  if (idx < n) {
    uint32_t local_carry_in = (b_carry_in == GENERATE) ? 1 : 0;
    result[idx] = static_cast<uint32_t>(sum) + local_carry_in;
    prop_flags[idx] = (b_carry_in == PROPAGATE) ? 1 : 0;
  }
  if (tid == blockDim.x - 1) {
    block_states[blockDim.x] = CarryOp()(b_carry_in, local_state);
  }
}

__global__ void scan_block_states_kernel(const uint8_t* block_states,
                                         uint32_t* block_carries_in,
                                         int num_blocks,
                                         uint32_t initial_carry_in) {
  if (threadIdx.x != 0) return;

  uint32_t carry = initial_carry_in;
  for (int i = 0; i < num_blocks; ++i) {
    block_carries_in[i] = carry;
    uint8_t state = block_states[i];
    if (state == GENERATE)
      carry = 1;
    else if (state == KILL)
      carry = 0;
  }
}

__global__ void apply_base_carry_kernel(uint32_t* result,
                                        const uint32_t* block_carries_in,
                                        const uint8_t* prop_flags, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  if (block_carries_in[blockIdx.x] == 1 && prop_flags[idx] == 1)
    result[idx] += 1;
}

__global__ void flip_bits_kernel(const uint32_t* b, uint32_t* flipped_b,
                                 size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) flipped_b[idx] = ~b[idx];
}

}  // namespace

namespace pi::ops {

void add(const uint32_t* a, const uint32_t* b, uint32_t* result, size_t n,
         uint32_t carry_in) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  uint8_t* d_block_states = nullptr;
  uint8_t* d_prop_flags = nullptr;
  uint32_t* d_block_carries_in = nullptr;

  cudaMalloc(&d_block_states, blocks * sizeof(uint8_t));
  cudaMalloc(&d_prop_flags, n * sizeof(uint8_t));
  cudaMalloc(&d_block_carries_in, blocks * sizeof(uint32_t));

  block_add_kernel<<<blocks, threads>>>(a, b, result, d_block_states,
                                        d_prop_flags, n);
  scan_block_states_kernel<<<1, 1>>>(d_block_states, d_block_carries_in, blocks,
                                     carry_in);
  apply_base_carry_kernel<<<blocks, threads>>>(result, d_block_carries_in,
                                               d_prop_flags, n);

  cudaFree(d_block_states);
  cudaFree(d_prop_flags);
  cudaFree(d_block_carries_in);
}

void add_big_float(const core::BigFloat* a, const core::BigFloat* b,
                   core::BigFloat* result) {
  add(a->limbs, b->limbs, result->limbs, a->num_limbs, 0);
  cudaDeviceSynchronize();
}

void sub_big_float(const core::BigFloat* a, const core::BigFloat* b,
                   core::BigFloat* result) {
  size_t n = b->num_limbs;
  uint32_t* d_flipped_b = nullptr;
  cudaMalloc(&d_flipped_b, n * sizeof(uint32_t));

  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  flip_bits_kernel<<<blocks, threads>>>(b->limbs, d_flipped_b, n);
  add(a->limbs, d_flipped_b, result->limbs, n, 1);

  cudaDeviceSynchronize();
  cudaFree(d_flipped_b);

}  // namespace pi::ops
