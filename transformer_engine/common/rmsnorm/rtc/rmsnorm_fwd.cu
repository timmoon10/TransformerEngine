/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "utils.cuh"

using namespace transformer_engine;

namespace {

// Parameters
using IType = __ITYPE__;
using OType = __OTYPE__;
using WType = __WTYPE__;
using CType = float;

constexpr size_t load_size = __LOAD_SIZE__;
constexpr size_t warps_per_block_m = __WARPS_M__;
constexpr size_t warps_per_block_n = __WARPS_N__;
constexpr size_t width = __WIDTH__;
constexpr size_t nblocks_n = __NBLOCKS_N__;

constexpr size_t warps_per_block = warps_per_block_m * warps_per_block_n;
constexpr size_t block_size = warps_per_block * THREADS_PER_WARP;

}  // namespace

__global__ void
__launch_bounds__(block_size)
rmsnorm_fwd_optimimzed_kernel(const IType * __restrict__ const input_ptr,
                              const WType * __restrict__ const gamma_ptr,
                              const CType epsilon,
                              IType * __restrict__  const output_ptr,
                              CType * __restrict__ const rsigma_ptr,
                              const CType * __restrict__ const scale_ptr,
                              CType * __restrict__ const amax_ptr,
                              void * __restrict__ const workspace_ptr,
                              int * __restrict__ const barrier_ptr,
                              const size_t height) {
  // Warp indices
  const size_t lane = threadIdx.x % THREADS_PER_WARP;
  const size_t warp = threadIdx.x / THREADS_PER_WARP;
  const size_t warp_m = warp / warps_per_block_n;
  const size_t warp_n = warp % warps_per_block_n;

  // Block indices
  constexpr size_t bdim_m = warps_per_block_m;
  constexpr size_t bdim_n = warps_per_block_n * THREADS_PER_WARP;
  const size_t nblocks_m = height / bdim_m;
  const size_t bid_m = blockIdx.x / nblocks_n;
  const size_t bid_n = blockIdx.x % nblocks_n;

  // Global thread indices
  const size_t gdim_m = bdim_m * nblocks_m;
  constexpr size_t gdim_n = bdim_n * nblocks_n;
  const size_t gid_m = bid_m * bdim_m + warp_m;
  const size_t gid_n = bid_n * bdim_n + warp_n * THREADS_PER_WARP + lane;

  // Objects for stats reductions
  using Reducer = Reducer<CType, nblocks_n, warps_per_block_m, warps_per_block_n>;
  struct ReducerParams {
    int *barrier = barrier_ptr;
    int ctas_per_row = nblocks_m;
    int ctas_per_col = nblocks_n;
    void *workspace = workspace_ptr;
  };
  ReducerParams params;
  constexpr size_t SMEM_BYTES = Reducer::SMEM_BYTES > 0 ? Reducer::SMEM_BYTES : 1;
  __shared__ unsigned char smem[SMEM_BYTES];
  Reducer reducer(params, bid_m, bid_n, warp_m, warp_n, lane, smem);
  Sum<CType> sum;

  // Objects for vectorized memory accesses
  constexpr size_t vec_size = load_size / sizeof(IType);
  constexpr size_t elements_per_iteration = vec_size * gdim_n;
  constexpr size_t num_iterations = width / elements_per_iter;
  using IVec = Vec<IType, vec_size>;
  using OVec = Vec<OType, vec_size>;
  using WVec = Vec<WType, vec_size>;
  using CVec = Vec<CType, vec_size>;

  // FP8 scaling factors
  const CType scale = scale_ptr == nullptr ? 1 : *scale_ptr;
  CType amax = 0;

  // Load weights
  CVec gamma[num_iterations];
  #pragma unroll
  for (size_t it = 0, col = gid_n;
       it < num_iterations;
       ++it, col += elements_per_iteration) {
    WVec g;
    g.load(&gamma_ptr[col]);
    g.to(gamma[it]);
  }

  // Iterate through rows and apply RMSNorm
  for (size_t row = gid_m; row < height; row += gdim_m) {
    // Load input
    CVec input[num_iterations];
    #pragma unroll
    for (size_t it = 0, col = gid_n;
         it < num_iterations;
         ++it, col += elements_per_iteration) {
      IVec x;
      x.load(&input_ptr[row*width + col]);
      x.to(input[it]);
    }

    // Compute variance
    CType sqsigma = 0;
    #pragma unroll
    for (size_t it = 0; it < num_iterations; ++it) {
      #pragma unroll
      for (size_t jt = 0; jt < vec_size; ++jt) {
        const CType x = input[it].data.elt[jt];
        sqsigma += x * x;
      }
    }
    sqsigma = reducer.allreduce(sqsigma, sum) / width;
    CType rsisgma = rsqrtf(sqsigma + epsilon);

    // Write statistics
    if (gid_n == 0) {
      rsigma_ptr[row] = rsigma;
    }

    // Compute output
    #pragma unroll
    for (size_t it = 0, col = gid_n;
         it < num_iterations;
         ++it, col += elements_per_iteration) {
      // Compute output values
      CVec z;
      #pragma unroll
      for (size_t jt = 0; jt < vec_size; ++jt) {
        const CType x = x[it].data.elt[jt];
        const CType g = gamma[it].data.elt[jt];
        z.data.elt[jt] = g * x * rsigma;
      }

      // Apply FP8 factors
      if (amax_ptr != nullptr) {
        #pragma unroll
        for (size_t jt = 0; jt < vec_size; ++jt) {
          __builtin_assume(amax >= 0);
          amax = fmaxf(amax, z.data.elt[jt]);
        }
      }
      if (scale_ptr != nullptr) {
        #pragma unroll
        for (size_t jt = 0; jt < vec_size; ++jt) {
          z.data.elt[jt] *= scale;
        }
      }

      // Store output
      OVec z_out;
      z.to(z_out);
      z_out.store(&output_ptr[row*width + col]);
    }
  }

  // Finalize amax
  if (amax_ptr != nullptr) {
    amax = reduce_max<warps_per_block>(amax, warp);
    if (threadIdx.x == 0) {
      atomicMaxFloat(amax_ptr, amax);
    }
  }
}
