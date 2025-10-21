/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ptx.cuh
 *  \brief BW PTX
 */

#ifndef TRANSFORMER_ENGINE_PTX_CUH_
#define TRANSFORMER_ENGINE_PTX_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#if CUDA_VERSION >= 12080
#include <cuda_fp4.h>
#endif  // CUDA_VERSION >= 12080

namespace transformer_engine {

namespace cuda_arch {

/*! \brief Specification for CUDA arch.
 *
 * For compatibility, the current CUDA arch must have compute
 * capability that matches spec or is newer.
 */
template <int ComputeCapability>
struct Arch {
  /// ID corresponding to __CUDA_ARCH__ macro
  static constexpr int id = ComputeCapability * 10;
};

/*! \brief Specification for CUDA arch with arch-specific features.
 *
 * For compatibility, the current CUDA arch must have compute
 * capability that exactly matches spec and must support arch-specific
 * features.
 */
template <int ComputeCapability>
struct ArchSpecific {
  /// ID corresponding to __CUDA_ARCH_SPECIFIC__ macro
  static constexpr int id = ComputeCapability * 10;
};

/*! \brief Specification for CUDA arch with family-specific features.
 *
 * For compatibility, the current CUDA arch must have compute
 * capability that matches spec family and must support
 * family-specific features.
 */
template <int ComputeCapability>
struct ArchFamily {
  /// ID corresponding to __CUDA_ARCH_FAMILY_SPECIFIC__ macro
  static constexpr int id = ComputeCapability * 10;
};

namespace details {

template <int ArchId, int ArchSpecificId, int ArchFamilyId>
struct _CurrentArch;

// Type trait to determine CUDA arch compatibility
template <typename CurrentArchT, typename ArchT>
struct IsCompatible : std::false_type {};
template <int ArchId, int ArchSpecificId, int ArchFamilyId, int ComputeCapability>
struct IsCompatible<_CurrentArch<ArchId, ArchSpecificId, ArchFamilyId>, Arch<ComputeCapability>>
  : std::bool_constant<ArchId >= Arch<ComputeCapability>::id> {};
template <int ArchId, int ArchSpecificId, int ArchFamilyId, int ComputeCapability>
struct IsCompatible<_CurrentArch<ArchId, ArchSpecificId, ArchFamilyId>, ArchSpecific<ComputeCapability>>
  : std::bool_constant<ArchSpecificId == ArchSpecific<ComputeCapability>::id> {};
template <int ArchId, int ArchSpecificId, int ArchFamilyId, int ComputeCapability>
struct IsCompatible<_CurrentArch<ArchId, ArchSpecificId, ArchFamilyId>, ArchFamily<ComputeCapability>>
  : std::bool_constant<ArchFamilyId / 100 == ArchFamily<ComputeCapability>::id / 100> {};

/*! \brief CUDA arch that is currently being used by the compiler.
 *
 * This class should not be declared directly, but obtained via the
 * NVTE_CURRENT_CUDA_ARCH macro.
 */
template <int ArchId, int ArchSpecificId, int ArchFamilyId>
struct _CurrentArch {

/// ID corresponding to __CUDA_ARCH__ macro
constexpr static int arch_id = ArchId;
/// ID corresponding to __CUDA_ARCH_SPECIFIC__ macro
constexpr static int arch_specific_id = ArchSpecificId;
/// ID corresponding to __CUDA_ARCH_SPECIFIC__ macro
constexpr static int arch_family_id = ArchFamilyId;

/*! /brief Whether current CUDA arch is compatible with a CUDA arch spec */
template <typename ArchT>
static inline constexpr bool compatible() {
  return IsCompatible<_CurrentArch, ArchT>::value;
}

/*! /brief Whether current CUDA arch is compatible with any CUDA arch spec */
template <typename... ArchTs>
static constexpr bool any_compatible() {
  return (... || compatible<ArchTs>());
}

};

}  // namespace details

// If needed, define CUDA arch macros in CUDA 12.9+ format
#if CUDA_VERSION < 12090
#if __CUDA_ARCH_HAS_FEATURE__(SM90_ALL)
#define __CUDA_ARCH_SPECIFIC__ 900
#define __CUDA_ARCH_FAMILY_SPECIFIC__ 900
#endif
#if __CUDA_ARCH_HAS_FEATURE__(SM100_ALL)
#define __CUDA_ARCH_SPECIFIC__ 1000
#define __CUDA_ARCH_FAMILY_SPECIFIC__ 1000
#endif
#if __CUDA_ARCH_HAS_FEATURE__(SM101_ALL)
#define __CUDA_ARCH_SPECIFIC__ 1010
#define __CUDA_ARCH_FAMILY_SPECIFIC__ 1010
#endif
#if __CUDA_ARCH_HAS_FEATURE__(SM120_ALL)
#define __CUDA_ARCH_SPECIFIC__ 1200
#define __CUDA_ARCH_FAMILY_SPECIFIC__ 1200
#endif
#endif

// Helper macros for NVTE_CURRENT_CUDA_ARCH
#ifdef __CUDA_ARCH__
#define _NVTE_CURRENT_CUDA_ARCH_CUDA_ARCH __CUDA_ARCH__
#else
#define _NVTE_CURRENT_CUDA_ARCH_CUDA_ARCH 0
#endif
#ifdef __CUDA_ARCH_SPECIFIC__
#define _NVTE_CURRENT_CUDA_ARCH_CUDA_ARCH_SPECIFIC __CUDA_ARCH_SPECIFIC__
#else
#define _NVTE_CURRENT_CUDA_ARCH_CUDA_ARCH_SPECIFIC 0
#endif
#ifdef __CUDA_ARCH_FAMILY_SPECIFIC__
#define _NVTE_CURRENT_CUDA_ARCH_CUDA_ARCH_FAMILY_SPECIFIC __CUDA_ARCH_FAMILY_SPECIFIC__
#else
#define _NVTE_CURRENT_CUDA_ARCH_CUDA_ARCH_FAMILY_SPECIFIC 0
#endif

// Macro to get currently active CUDA arch
#define NVTE_CURRENT_CUDA_ARCH                              \
  ::transformer_engine::cuda_arch::details::_CurrentArch<   \
    _NVTE_CURRENT_CUDA_ARCH_CUDA_ARCH,                      \
    _NVTE_CURRENT_CUDA_ARCH_CUDA_ARCH_SPECIFIC,             \
    _NVTE_CURRENT_CUDA_ARCH_CUDA_ARCH_FAMILY_SPECIFIC>

// Macros for CUDA features
// TODO(tmoon): Use template metaprogramming instead of macros
#define NVTE_CUDA_ARCHS_IN_BLACKWELL_FAMILY \
  cuda_arch::ArchFamily<100>, cuda_arch::ArchFamily<110>, cuda_arch::ArchFamily<120>
#define NVTE_CUDA_ARCHS_WITH_STOCHASTIC_ROUNDING \
  cuda_arch::ArchSpecific<100>, cuda_arch::ArchSpecific<103>

}  // namespace cuda_arch

namespace ptx {

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-init
__device__ __forceinline__ void mbarrier_init(uint64_t *mbar, const uint32_t count) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(mbar_ptr), "r"(count) : "memory");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-inval
__device__ __forceinline__ void mbarrier_invalid(uint64_t *mbar) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.inval.shared.b64 [%0];" ::"r"(mbar_ptr) : "memory");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive
__device__ __forceinline__ void mbarrier_arrive(uint64_t *mbar) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];" ::"r"(mbar_ptr) : "memory");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t *mbar, const uint32_t tx_count) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(mbar_ptr), "r"(tx_count)
               : "memory");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__device__ __forceinline__ void fence_mbarrier_init_release_cluster() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("fence.mbarrier_init.release.cluster;");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
// global -> shared::cluster
__device__ __forceinline__ void cp_async_bulk_tensor_1d_global_to_shared(
    uint64_t *dst_shmem, const uint64_t *src_global_ptr, const uint32_t size, uint64_t *mbar) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t dst_shmem_ptr = __cvta_generic_to_shared(dst_shmem);
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  // triggers async copy, i.e. the thread continues until wait() on mbarrier
  // barrier condition:
  // - leader must arrive (i.e. 1 thread as set above)
  // - TMA hardware substracts bytes from expect_tx counter, must reach zero
  asm volatile(
      "cp.async.bulk.shared::cta.global"
      ".mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst_shmem_ptr),
      "l"(src_global_ptr), "r"(size), "r"(mbar_ptr)
      : "memory");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
// global -> shared::cluster
__device__ __forceinline__ void cp_async_bulk_tensor_2d_global_to_shared(
    uint64_t *dst_shmem, const uint64_t *tensor_map_ptr, const uint32_t offset_x,
    const uint32_t offset_y, uint64_t *mbar) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t dst_shmem_ptr = __cvta_generic_to_shared(dst_shmem);
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  // triggers async copy, i.e. the thread continues until wait() on mbarrier
  // barrier condition:
  // - leader must arrive (i.e. 1 thread as set above)
  // - TMA hardware substracts bytes from expect_tx counter, must reach zero
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
      ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(dst_shmem_ptr),
      "l"(tensor_map_ptr), "r"(offset_x), "r"(offset_y), "r"(mbar_ptr)
      : "memory");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__device__ __forceinline__ bool mbarrier_try_wait_parity(uint32_t mbar_ptr, const uint32_t parity) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t waitComplete;
  asm volatile(
      "{\n\t .reg .pred P_OUT; \n\t"
      "mbarrier.try_wait.parity.shared::cta.b64  P_OUT, [%1], %2; \n\t"
      "selp.b32 %0, 1, 0, P_OUT; \n"
      "}"
      : "=r"(waitComplete)
      : "r"(mbar_ptr), "r"(parity)
      : "memory");
  return static_cast<bool>(waitComplete);
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  return true;
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t *mbar, const uint32_t parity) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t mbar_ptr = __cvta_generic_to_shared(mbar);
  while (!mbarrier_try_wait_parity(mbar_ptr, parity)) {
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

constexpr uint32_t FP32_MANTISSA_BITS = 23;
constexpr uint32_t FP32_EXPONENT_BIAS = 127;

__device__ __forceinline__ float exp2f_rcp(e8m0_t biased_exp) {
  return (biased_exp == 0) ? 1
                           : __int_as_float((254 - biased_exp)
                                            << FP32_MANTISSA_BITS);  // 127 - (biased_exp - 127)
}

__device__ __forceinline__ float exp2f(e8m0_t biased_exp) {
  return __int_as_float(biased_exp << FP32_MANTISSA_BITS);
}

__device__ __forceinline__ e8m0_t float_to_e8m0(float val) {
  using CurrentCUDAArch = NVTE_CURRENT_CUDA_ARCH;
  if constexpr (CurrentCUDAArch::any_compatible<NVTE_CUDA_ARCHS_IN_BLACKWELL_FAMILY>()) {
    uint16_t out;
    asm volatile(
        "{\n"
        "cvt.rp.satfinite.ue8m0x2.f32  %0, 0.0, %1;\n"
        "}"
        : "=h"(out)
        : "f"(val));
    return *reinterpret_cast<e8m0_t *>(&out);
  } else {
    // TODO: nan/inf needs to be set for any value
    // of nan/inf in input not just amax.
    if (isnan(val)) {
      return 0xFF;
    }
    if (isinf(val)) {
      return 0xFE;
    }
    if (val == 0.0f) {
      return 0x00;
    }
    uint32_t val_u32 = *reinterpret_cast<uint32_t *>(&val);
    e8m0_t exponent = (val_u32 >> FP32_MANTISSA_BITS);
    uint32_t mantissa = val_u32 & 0x7FFFFF;
    // Round up exponent and deal with satfinite.
    if ((mantissa > 0 && exponent != 0xFE) && !(exponent == 0 && mantissa <= 0x400000)) {
      ++exponent;
    }
    return exponent;
  }
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
// shared::cta -> global
__device__ __forceinline__ void cp_async_bulk_tensor_1d_shared_to_global(uint64_t *dst_global_ptr,
                                                                         const uint64_t *src_shmem,
                                                                         const uint32_t size) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint32_t src_shmem_ptr = __cvta_generic_to_shared(src_shmem);
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst_global_ptr),
               "r"(src_shmem_ptr), "r"(size)
               : "memory");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor
// shared::cta -> global
__device__ __forceinline__ void cp_async_bulk_tensor_2d_shared_to_global(
    const uint64_t *tensor_map_ptr, const uint32_t offset_x, const uint32_t offset_y,
    uint64_t *src_shmem) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint32_t src_shmem_ptr = __cvta_generic_to_shared(src_shmem);
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];" ::"l"(
                   tensor_map_ptr),
               "r"(offset_x), "r"(offset_y), "r"(src_shmem_ptr)
               : "memory");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group
__device__ __forceinline__ void cp_async_bulk_wait_group() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.bulk.wait_group 0;");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group
template <size_t W>
__device__ __forceinline__ void cp_async_bulk_wait_group_read() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.bulk.wait_group.read 0;");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}

template <>
__device__ __forceinline__ void cp_async_bulk_wait_group_read<0>() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.bulk.wait_group.read 0;");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}
template <>
__device__ __forceinline__ void cp_async_bulk_wait_group_read<1>() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.bulk.wait_group.read 1;");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}
template <>
__device__ __forceinline__ void cp_async_bulk_wait_group_read<2>() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.bulk.wait_group.read 2;");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}
template <>
__device__ __forceinline__ void cp_async_bulk_wait_group_read<4>() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.bulk.wait_group.read 4;");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-commit-group
__device__ __forceinline__ void cp_async_bulk_commit_group() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("cp.async.bulk.commit_group;");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}

// Proxy fence (bi-directional):
__device__ __forceinline__ void fence_proxy_async() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("fence.proxy.async;");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}

__device__ __forceinline__ void fence_proxy_async_shared_cta() {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("fence.proxy.async.shared::cta;");
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}

template <typename T>
struct alignas(2 * sizeof(T)) FPx2 {
  T x;
  T y;
};

template <typename T>
struct FPx4 {
  T x1;
  T x2;
  T x3;
  T x4;
};

template <typename T>
struct Type2x {};

template <>
struct Type2x<float> {
  using type = float2;
};

template <>
struct Type2x<bf16> {
  using type = __nv_bfloat162;
};

template <>
struct Type2x<fp16> {
  using type = __half2;
};

using floatx2 = FPx2<float>;
using bf16x2 = FPx2<bf16>;
using fp16x2 = FPx2<fp16>;
using fp8e4m3x2 = FPx2<fp8e4m3>;
using fp8e5m2x2 = FPx2<fp8e5m2>;

using floatx4 = FPx4<float>;
using bf16x4 = FPx4<bf16>;
using fp16x4 = FPx4<fp16>;
using fp8e4m3x4 = FPx4<fp8e4m3>;
using fp8e5m2x4 = FPx4<fp8e5m2>;

static_assert(sizeof(floatx2) == 8);
static_assert(sizeof(bf16x2) == 4);
static_assert(sizeof(fp16x2) == 4);
static_assert(sizeof(fp8e4m3x2) == 2);
static_assert(sizeof(fp8e5m2x2) == 2);

#if CUDA_VERSION >= 12080
using fp4e2m1 = __nv_fp4_e2m1;
using fp4e2m1x2 = __nv_fp4x2_e2m1;
using fp4e2m1x4 = __nv_fp4x4_e2m1;
static_assert(sizeof(fp4e2m1x2) == 1);
static_assert(sizeof(fp4e2m1x4) == 2);
#endif  // CUDA_VERSION >= 12080

// When converting to .e2m1x2 data formats, the destination operand d has .b8 type.
// When converting two .f32 inputs to .e2m1x2, each input is converted to the specified format,
// and the converted values are packed in the destination operand d such that the value
// converted from input a is stored in the upper 4 bits of d and the value converted
// from input b is stored in the lower 4 bits of d.

// SIMD like "Fused" cast + multiplication (x4)
#if CUDA_VERSION >= 12080
template <typename Tx2>
__device__ __forceinline__ void mul_cvt_4x(fp4e2m1x4 &out, const Tx2 &in01, const Tx2 &in23,
                                           const float scale) {
  const float x0 = static_cast<float>(in01.x) * scale;
  const float x1 = static_cast<float>(in01.y) * scale;
  const float x2 = static_cast<float>(in23.x) * scale;
  const float x3 = static_cast<float>(in23.y) * scale;
  out = fp4e2m1x4(make_float4(x0, x1, x2, x3));
}
#endif  // CUDA_VERSION >= 12080

// SIMD like "Fused" cast + multiplication (x2)
__device__ __forceinline__ void mul_cvt_2x(fp8e4m3x2 &out, const floatx2 &in,
                                           const floatx2 &scale) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile(
      "{\n"
      ".reg.b64 val_pair; \n\t"
      ".reg.b32 val1; \n\t"
      ".reg.b32 val2; \n\t"
      "mul.f32x2 val_pair, %1, %2; \n\t"
      "mov.b64 {val2,val1}, val_pair; \n\t"
      "cvt.rn.satfinite.e4m3x2.f32 %0, val1, val2; \n\t"
      "}"
      : "=h"(reinterpret_cast<uint16_t &>(out))
      : "l"(reinterpret_cast<const uint64_t &>(in)),
        "l"(reinterpret_cast<const uint64_t &>(scale)));
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
}

__device__ __forceinline__ void mul_cvt_2x(fp8e5m2x2 &out, const floatx2 &in,
                                           const floatx2 &scale) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile(
      "{\n"
      ".reg.b64 val_pair; \n\t"
      ".reg.b32 val1; \n\t"
      ".reg.b32 val2; \n\t"
      "mul.f32x2 val_pair, %1, %2; \n\t"
      "mov.b64 {val2,val1}, val_pair; \n\t"
      "cvt.rn.satfinite.e5m2x2.f32 %0, val1, val2; \n\t"
      "}"
      : "=h"(reinterpret_cast<uint16_t &>(out))
      : "l"(reinterpret_cast<const uint64_t &>(in)),
        "l"(reinterpret_cast<const uint64_t &>(scale)));
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
}

__device__ __forceinline__ void mul_cvt_2x(fp8e4m3x2 &out, const bf16x2 &in, const floatx2 &scale) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile(
      "{\n"
      ".reg.b64 val_pair_before; \n\t"
      ".reg.b64 val_pair_after; \n\t"
      ".reg.b32 val1; \n\t"
      ".reg.b32 val2; \n\t"
      ".reg.b16 val1_bf16; \n\t"
      ".reg.b16 val2_bf16; \n\t"
      "mov.b32 {val1_bf16, val2_bf16} , %1; \n\t"
      "cvt.f32.bf16 val1, val1_bf16; \n\t"
      "cvt.f32.bf16 val2, val2_bf16; \n\t"
      "mov.b64 val_pair_before, {val1,val2}; \n\t"
      "mul.f32x2 val_pair_after, val_pair_before, %2; \n\t"
      "mov.b64 {val2,val1}, val_pair_after; \n\t"
      "cvt.rn.satfinite.e4m3x2.f32 %0, val1, val2; \n\t"
      "}"
      : "=h"(reinterpret_cast<uint16_t &>(out))
      : "r"(reinterpret_cast<const uint32_t &>(in)),
        "l"(reinterpret_cast<const uint64_t &>(scale)));
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
}

__device__ __forceinline__ void mul_cvt_2x(fp8e5m2x2 &out, const bf16x2 &in, const floatx2 &scale) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile(
      "{\n"
      ".reg.b64 val_pair_before; \n\t"
      ".reg.b64 val_pair_after; \n\t"
      ".reg.b32 val1; \n\t"
      ".reg.b32 val2; \n\t"
      ".reg.b16 val1_bf16; \n\t"
      ".reg.b16 val2_bf16; \n\t"
      "mov.b32 {val1_bf16, val2_bf16} , %1; \n\t"
      "cvt.f32.bf16 val1, val1_bf16; \n\t"
      "cvt.f32.bf16 val2, val2_bf16; \n\t"
      "mov.b64 val_pair_before, {val1,val2}; \n\t"
      "mul.f32x2 val_pair_after, val_pair_before, %2; \n\t"
      "mov.b64 {val2,val1}, val_pair_after; \n\t"
      "cvt.rn.satfinite.e5m2x2.f32 %0, val1, val2; \n\t"
      "}"
      : "=h"(reinterpret_cast<uint16_t &>(out))
      : "r"(reinterpret_cast<const uint32_t &>(in)),
        "l"(reinterpret_cast<const uint64_t &>(scale)));
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
}

__device__ __forceinline__ void mul_cvt_2x(fp8e4m3x2 &out, const fp16x2 &in, const floatx2 &scale) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile(
      "{\n"
      ".reg.b64 val_pair_before; \n\t"
      ".reg.b64 val_pair_after; \n\t"
      ".reg.b32 val1; \n\t"
      ".reg.b32 val2; \n\t"
      ".reg.b16 val1_fp16; \n\t"
      ".reg.b16 val2_fp16; \n\t"
      "mov.b32 {val1_fp16, val2_fp16} , %1; \n\t"
      "cvt.f32.f16 val1, val1_fp16; \n\t"
      "cvt.f32.f16 val2, val2_fp16; \n\t"
      "mov.b64 val_pair_before, {val1,val2}; \n\t"
      "mul.f32x2 val_pair_after, val_pair_before, %2; \n\t"
      "mov.b64 {val2,val1}, val_pair_after; \n\t"
      "cvt.rn.satfinite.e4m3x2.f32 %0, val1, val2; \n\t"
      "}"
      : "=h"(reinterpret_cast<uint16_t &>(out))
      : "r"(reinterpret_cast<const uint32_t &>(in)),
        "l"(reinterpret_cast<const uint64_t &>(scale)));
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
}

__device__ __forceinline__ void mul_cvt_2x(fp8e5m2x2 &out, const fp16x2 &in, const floatx2 &scale) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile(
      "{\n"
      ".reg.b64 val_pair_before; \n\t"
      ".reg.b64 val_pair_after; \n\t"
      ".reg.b32 val1; \n\t"
      ".reg.b32 val2; \n\t"
      ".reg.b16 val1_fp16; \n\t"
      ".reg.b16 val2_fp16; \n\t"
      "mov.b32 {val1_fp16, val2_fp16} , %1; \n\t"
      "cvt.f32.f16 val1, val1_fp16; \n\t"
      "cvt.f32.f16 val2, val2_fp16; \n\t"
      "mov.b64 val_pair_before, {val1,val2}; \n\t"
      "mul.f32x2 val_pair_after, val_pair_before, %2; \n\t"
      "mov.b64 {val2,val1}, val_pair_after; \n\t"
      "cvt.rn.satfinite.e5m2x2.f32 %0, val1, val2; \n\t"
      "}"
      : "=h"(reinterpret_cast<uint16_t &>(out))
      : "r"(reinterpret_cast<const uint32_t &>(in)),
        "l"(reinterpret_cast<const uint64_t &>(scale)));
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
}

__device__ __forceinline__ void abs_max_2x(bf16x2 &dst, const bf16x2 &p1, const bf16x2 &p2) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;"
               : "=r"(reinterpret_cast<uint32_t &>(dst))
               : "r"(reinterpret_cast<const uint32_t &>(p1)),
                 "r"(reinterpret_cast<const uint32_t &>(p2)));
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
}

__device__ __forceinline__ void abs_max_2x(fp16x2 &dst, const fp16x2 &p1, const fp16x2 &p2) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
  asm volatile("max.xorsign.abs.f16x2 %0, %1, %2;"
               : "=r"(reinterpret_cast<uint32_t &>(dst))
               : "r"(reinterpret_cast<const uint32_t &>(p1)),
                 "r"(reinterpret_cast<const uint32_t &>(p2)));
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
}

}  // namespace ptx

namespace {

template <int num_barriers, int THREADS_PER_BLOCK>
__forceinline__ __device__ void initialize_barriers(uint64_t *mbar, const bool is_master_thread) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (is_master_thread) {
    // Initialize barrier. All `blockDim.x * blockDim.y` threads in block participate.
#pragma unroll
    for (int iter = 0; iter < num_barriers; ++iter) {
      ptx::mbarrier_init(&mbar[iter], THREADS_PER_BLOCK);
    }
    ptx::fence_proxy_async_shared_cta();
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

template <int num_barriers>
__forceinline__ __device__ void destroy_barriers(uint64_t *mbar, const bool is_master_thread) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  // Destroy barrier. This invalidates the memory region of the barrier. If
  // further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  if (is_master_thread) {
#pragma unroll
    for (int iter = 0; iter < num_barriers; ++iter) {
      ptx::mbarrier_invalid(&mbar[iter]);
    }
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__forceinline__ __device__ void copy_1d_to_shared(void *dst, const void *src,
                                                  const size_t num_bytes, uint64_t *barrier,
                                                  const bool is_master_thread) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (is_master_thread) {
    // Initiate bulk tensor copy
    ptx::cp_async_bulk_tensor_1d_global_to_shared(reinterpret_cast<uint64_t *>(dst),
                                                  reinterpret_cast<const uint64_t *>(src),
                                                  num_bytes, barrier);

    // Arrive on the barrier and tell how many bytes are expected to come in.
    ptx::mbarrier_arrive_expect_tx(barrier, num_bytes);
  } else {
    // Other threads just arrive
    ptx::mbarrier_arrive(barrier);
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__forceinline__ __device__ void copy_2d_to_shared(void *dst, const void *src, const size_t chunk_X,
                                                  const size_t chunk_Y, const size_t num_bytes,
                                                  uint64_t *barrier, const bool is_master_thread) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (is_master_thread) {
    // Initiate bulk tensor copy
    ptx::cp_async_bulk_tensor_2d_global_to_shared(reinterpret_cast<uint64_t *>(dst),
                                                  reinterpret_cast<const uint64_t *>(src), chunk_X,
                                                  chunk_Y, barrier);

    // Arrive on the barrier and tell how many bytes are expected to come in.
    ptx::mbarrier_arrive_expect_tx(barrier, num_bytes);
  } else {
    // Other threads just arrive
    ptx::mbarrier_arrive(barrier);
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__forceinline__ __device__ void copy_2d_to_sharedx2(void *dst, const void *src,
                                                    const size_t chunk_X1, const size_t chunk_Y1,
                                                    void *dst2, const void *src2,
                                                    const size_t chunk_X2, const size_t chunk_Y2,
                                                    const size_t num_bytes, uint64_t *barrier,
                                                    const bool is_master_thread) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (is_master_thread) {
    // Initiate bulk tensor copy
    ptx::cp_async_bulk_tensor_2d_global_to_shared(reinterpret_cast<uint64_t *>(dst),
                                                  reinterpret_cast<const uint64_t *>(src), chunk_X1,
                                                  chunk_Y1, barrier);

    ptx::cp_async_bulk_tensor_2d_global_to_shared(reinterpret_cast<uint64_t *>(dst2),
                                                  reinterpret_cast<const uint64_t *>(src2),
                                                  chunk_X2, chunk_Y2, barrier);

    // Arrive on the barrier and tell how many bytes are expected to come in.
    ptx::mbarrier_arrive_expect_tx(barrier, 2 * num_bytes);
  } else {
    // Other threads just arrive
    ptx::mbarrier_arrive(barrier);
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

__forceinline__ __device__ void copy_2d_to_sharedx3(
    void *dst, const void *src, const size_t chunk_X1, const size_t chunk_Y1, void *dst2,
    const void *src2, const size_t chunk_X2, const size_t chunk_Y2, void *dst3, const void *src3,
    const size_t chunk_X3, const size_t chunk_Y3, const size_t num_bytes, uint64_t *barrier,
    const bool is_master_thread) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (is_master_thread) {
    // Initiate bulk tensor copy
    ptx::cp_async_bulk_tensor_2d_global_to_shared(reinterpret_cast<uint64_t *>(dst),
                                                  reinterpret_cast<const uint64_t *>(src), chunk_X1,
                                                  chunk_Y1, barrier);

    ptx::cp_async_bulk_tensor_2d_global_to_shared(reinterpret_cast<uint64_t *>(dst2),
                                                  reinterpret_cast<const uint64_t *>(src2),
                                                  chunk_X2, chunk_Y2, barrier);

    ptx::cp_async_bulk_tensor_2d_global_to_shared(reinterpret_cast<uint64_t *>(dst3),
                                                  reinterpret_cast<const uint64_t *>(src3),
                                                  chunk_X3, chunk_Y3, barrier);

    // Arrive on the barrier and tell how many bytes are expected to come in.
    ptx::mbarrier_arrive_expect_tx(barrier, 3 * num_bytes);
  } else {
    // Other threads just arrive
    ptx::mbarrier_arrive(barrier);
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

}  // namespace
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_PTX_CUH_
