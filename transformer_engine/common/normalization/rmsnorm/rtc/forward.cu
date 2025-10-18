/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "utils.cuh"

using namespace transformer_engine;

namespace {

// Data and compute types
using input_t = __INPUT_TYPE__;
using weight_t = __WEIGHT_TYPE__;
using output_t = __OUTPUT_TYPE__;
using compute_t = float;
using index_t = size_t;

// Types for vectorization
constexpr index_t vector_size = __VECTOR_SIZE__;
using input_vec_t = transformer_engine::Vec<input_t, vector_size>;
using weight_vec_t = transformer_engine::Vec<weight_t, vector_size>;
using output_vec_t = transformer_engine::Vec<output_t, vector_size>;

// CUDA block dimensions
constexpr index_t bdim_x = __BLOCK_DIM_X__;
constexpr index_t bdim_y = __BLOCK_DIM_Y__;
static_assert(bdim_x % THREADS_PER_WARP == 0);
constexpr index_t num_warps_m = bdim_y;
constexpr index_t num_warps_n = bdim_x / THREADS_PER_WARP;
constexpr index_t num_warps = num_warps_m * num_warps_n;

// Figure out what data each thread is responsible for
constexpr index_t num_cols = __NUM_COLS__;
constexpr index_t blocks_per_row = __BLOCKS_PER_ROW__;
static_assert(num_cols % blocks_per_row == 0);
constexpr index_t cols_per_block = num_cols / blocks_per_row;
static_assert(cols_per_block % bdim_x == 0);
constexpr index_t elements_per_thread = cols_per_block / bdim_x;
static_assert(elements_per_thread % vector_size == 0);
constexpr index_t vectors_per_thread = elements_per_thread / vector_size;

// Tensor scaling options
constexpr bool with_out_scale = __WITH_OUT_SCALE__;
constexpr bool with_amax = __WITH_AMAX__;

// Other options
constexpr bool zero_centered_gamma = __ZERO_CENTERED_GAMMA__;

}  // namespace

__global__ __launch_bounds__(bdim_x * bdim_y) void rmsnorm_fwd_tuned_kernel(
    compute_t epsilon,
    index_t num_rows,
    index_t blocks_per_col,
    const void * __restrict__ input_ptr_,
    const void * __restrict__ gamma_ptr_,
    void * __restrict__ output_ptr_,
    void * __restrict__ workspace_ptr,
    void * __restrict__ barrier_ptr,
    void * __restrict__ norm_scale_ptr_,
    void * __restrict__ amax_ptr_,
    const void * __restrict__ out_scale_ptr_,
    void * __restrict__ out_scale_inv_ptr_) {
  // CUDA thread and block indices
  __builtin_assume(threadIdx.x >= 0);
  __builtin_assume(threadIdx.x < bdim_x);
  __builtin_assume(threadIdx.y >= 0);
  __builtin_assume(threadIdx.y < bdim_y);
  __builtin_assume(blockIdx.x >= 0);
  __builtin_assume(blockIdx.x < blocks_per_row);
  __builtin_assume(blockIdx.y >= 0);
  const index_t tid_x = threadIdx.x;
  const index_t tid_y = threadIdx.y;
  const index_t bid_x = blockIdx.x;
  const index_t bid_y = blockIdx.y;
  const index_t gid_x = tid_x + bid_x * bdim_x;
  const index_t gid_y = tid_y + bid_y * bdim_y;

  // Warp indices
  const index_t warp_m = tid_y;
  const index_t warp_n = tid_x / THREADS_PER_WARP;
  const index_t lane = tid_x % THREADS_PER_WARP;

  // Data offsets
  const index_t row_start = gid_y;
  const index_t col_start = gid_x * elements_per_thread;

  // Convert pointers to correct types
  const auto *input_ptr = reinterpret_cast<const input_t *>(input_ptr_);
  const auto *gamma_ptr = reinterpret_cast<const weight_t *>(gamma_ptr_);
  auto *output_ptr = reinterpret_cast<output_t *>(output_ptr_);
  auto *norm_scale_ptr = reinterpret_cast<compute_t *>(norm_scale_ptr_);
  auto *amax_ptr = reinterpret_cast<compute_t *>(amax_ptr_);
  const auto *out_scale_ptr = reinterpret_cast<const compute_t *>(out_scale_ptr_);
  auto *out_scale_inv_ptr = reinterpret_cast<compute_t *>(out_scale_inv_ptr_);

  // Objects for stats reductions
  using Reducer = transformer_engine::Reducer<compute_t, blocks_per_row, num_warps_m, num_warps_n>;
  struct ReducerParams {
    void *workspace;
    void *barrier;
    index_t ctas_per_col;
  };
  ReducerParams reducer_params{workspace_ptr, barrier_ptr, blocks_per_col};
  constexpr index_t reducer_smem_size = Reducer::SMEM_BYTES > 0 ? Reducer::SMEM_BYTES : 1;
  __shared__ uint8_t reducer_smem[reducer_smem_size];
  Reducer reducer(reducer_params,
                  bid_y, bid_x, warp_m, warp_n, lane, reducer_smem);

  // Load weights
  compute_t g[elements_per_thread];
  const auto *gamma_vec_ptr = reinterpret_cast<const weight_vec_t *>(&gamma_ptr[col_start]);
#pragma unroll
  for (index_t i = 0; i < vectors_per_thread; ++i) {
    weight_vec_t g_in;
    g_in.load_from(&gamma_vec_ptr[i]);
#pragma unroll
    for (index_t j = 0; j < vector_size; ++j) {
      g[i * vector_size + j] = static_cast<compute_t>(g_in.data.elt[j]);
    }
  }
  if constexpr (zero_centered_gamma) {
#pragma unroll
    for (index_t i = 0; i < elements_per_thread; ++i) {
      g[i] += 1;
    }
  }

  // Load output scale if needed
  compute_t out_scale;
  if constexpr (with_out_scale) {
    out_scale = *out_scale_ptr;
  }

  // Initialize amax if needed
  compute_t amax;
  if constexpr (with_amax) {
    amax = 0;
  }

  // Iterate over data rows
  for (index_t row = row_start; row < num_rows; row += blocks_per_col * bdim_y) {
    // Pointers for vectorized memory accesses
    const auto *input_vec_ptr = reinterpret_cast<const input_vec_t *>(&input_ptr[row * num_cols + col_start]);
    auto *output_vec_ptr = reinterpret_cast<output_vec_t *>(&output_ptr[row * num_cols + col_start]);

    // Load input
    compute_t x[elements_per_thread];
#pragma unroll
    for (index_t i = 0; i < vectors_per_thread; ++i) {
      input_vec_t x_in;
      x_in.load_from(&input_vec_ptr[i]);
#pragma unroll
      for (index_t j = 0; j < vector_size; ++j) {
        x[i * vector_size + j] = static_cast<compute_t>(x_in.data.elt[j]);
      }
    }

    // Compute sum of squares
    compute_t sum_squares = 0;
#pragma unroll
    for (index_t i = 0; i < elements_per_thread; ++i) {
      sum_squares += x[i] * x[i];
    }
    sum_squares = reducer.allreduce(sum_squares);

    // Compute norm scale
    constexpr compute_t recip_num_cols = static_cast<compute_t>(1) / num_cols;
    const compute_t norm_scale = rsqrtf(recip_num_cols * sum_squares + epsilon);
    if (gid_x == 0) {  // One thread per row writes to global memory
      norm_scale_ptr[row] = norm_scale;
    }

    // Compute output values
    compute_t z[elements_per_thread];
#pragma unroll
    for (index_t i = 0; i < elements_per_thread; ++i) {
      z[i] = x[i] * norm_scale * g[i];
    }

    // Compute amax if needed
    if constexpr (with_amax) {
#pragma unroll
      for (index_t i = 0; i < elements_per_thread; ++i) {
        __builtin_assume(amax >= 0);
        amax = fmaxf(amax, fabsf(z[i]));
      }
    }

    // Apply output scale if needed
    if constexpr (with_out_scale) {
#pragma unroll
      for (index_t i = 0; i < elements_per_thread; ++i) {
        z[i] *= out_scale;
      }
    }

    // Store output
#pragma unroll
    for (index_t i = 0; i < vectors_per_thread; ++i) {
      output_vec_t z_out;
#pragma unroll
      for (index_t j = 0; j < vector_size; ++j) {
        z_out.data.elt[j] = static_cast<output_t>(z[i * vector_size + j]);
      }
      z_out.store_to(&output_vec_ptr[i]);
    }
  }

  // Output amax if needed
  if constexpr (with_amax) {
    amax = reduce_max<num_warps_m * num_warps_n>(amax, warp_n + warp_m * num_warps_n);
    if (tid_x == 0 && tid_y == 0) {
      static_assert(std::is_same<compute_t, float>::value);
      atomicMaxFloat(reinterpret_cast<compute_t *>(amax_ptr), amax);
    }
  }

  // Update output scale-inverse if needed
  if constexpr (with_out_scale) {
    if (gid_x == 0 && gid_y == 0 && out_scale_inv_ptr != nullptr) {
      reciprocal<compute_t>(out_scale_inv_ptr, out_scale);
    }
  }
}
