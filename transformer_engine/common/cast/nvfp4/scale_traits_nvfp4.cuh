/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file scale_traits_nvfp4.cuh
 *  \brief NVFP4 scale-format traits.
 *
 *  Host-callable helpers describing the NVFP4 scale storage formats. Kept free
 *  of ptx.cuh so that sources compiled for the generic Blackwell architectures
 *  (see transformer_engine_cuda_sources in CMakeLists.txt) can use them.
 */

#ifndef TRANSFORMER_ENGINE_NVFP4_SCALE_TRAITS_CUH_
#define TRANSFORMER_ENGINE_NVFP4_SCALE_TRAITS_CUH_

#include <cuda.h>

#include "../../common.h"

namespace transformer_engine {
namespace dispatch {
namespace nvfp4 {

// Central runtime-to-compile-time dispatch for NVFP4 scale storage types.
// SWITCH_FP8UE5M3_TYPE_HANDLE adds UE5M3 when the CUDA toolkit supports it.
#define TRANSFORMER_ENGINE_NVFP4_SCALE_TYPE_SWITCH(SCALE_DTYPE, SCALE_TYPE, ...)          \
  switch (SCALE_DTYPE) {                                                                  \
    case DType::kFloat8E4M3: {                                                            \
      using SCALE_TYPE = fp8e4m3;                                                         \
      { __VA_ARGS__ }                                                                     \
    } break;                                                                              \
      SWITCH_FP8UE5M3_TYPE_HANDLE(SCALE_TYPE, __VA_ARGS__)                                \
    default: {                                                                            \
      NVTE_ERROR("Unsupported NVFP4 scale dtype ", to_string(SCALE_DTYPE),                \
                 ". Expected Float8E4M3, or Float8UE5M3 when compiled with CUDA 13.4+."); \
    }                                                                                     \
  }

namespace core {

#if FP4_TYPE_SUPPORTED

// Scale-format-specific behavior belongs here rather than in individual kernels.
template <typename ScaleType>
struct NVFP4ScaleTraits {
  static constexpr bool is_supported = false;
  static constexpr bool supports_fp16_error_path = false;
  static constexpr float expected_max = 0.0f;
  static constexpr float headroom_max = 0.0f;
};

template <>
struct NVFP4ScaleTraits<fp8e4m3> {
  // E4M3 scales fit in FP16 and can use the packed E4M3-to-FP16 PTX fast
  // path. UE5M3 scales can exceed the FP16 range, so they retain the generic
  // FP32 error path.
  static constexpr bool is_supported = true;
  static constexpr bool supports_fp16_error_path = true;
  static constexpr float expected_max = 448.0f;
  static constexpr float headroom_max = 256.0f;
};

#if CUDA_VERSION >= 13040
template <>
struct NVFP4ScaleTraits<fp8ue5m3> {
  static constexpr bool is_supported = true;
  static constexpr bool supports_fp16_error_path = false;
  static constexpr float expected_max = 114688.0f;
  static constexpr float headroom_max = 65536.0f;
};
#endif

// Return the effective maximum used to derive the global NVFP4 encode scale.
// SCALE_TYPE_MAX is the resolved maximum for ScaleType (e.g., 448 for E4M3
// or 114688 for UE5M3). The headroom maximum keeps the 1.5x map-to-4 scale
// used by 4over6 within the scale format's representable range.
template <typename ScaleType,
          int SCALE_TYPE_MAX = static_cast<int>(NVFP4ScaleTraits<ScaleType>::expected_max)>
__host__ __device__ constexpr float scale_max() {
  using ScaleTraits = NVFP4ScaleTraits<ScaleType>;
  static_assert(ScaleTraits::is_supported, "Unsupported NVFP4 scale type.");
  if constexpr (ScaleTraits::is_supported) {
    static_assert(detail::TypeExtrema<ScaleType>::max == ScaleTraits::expected_max,
                  "Unexpected NVFP4 scale type maximum.");
    static_assert(SCALE_TYPE_MAX == static_cast<int>(ScaleTraits::expected_max) ||
                      SCALE_TYPE_MAX == static_cast<int>(ScaleTraits::headroom_max),
                  "Unsupported NVFP4 scale type maximum.");
    static_assert(ScaleTraits::headroom_max * 1.5f <= ScaleTraits::expected_max,
                  "NVFP4 4over6 scale headroom exceeds scale type maximum.");
    return static_cast<float>(SCALE_TYPE_MAX);
  } else {
    return 0.0f;
  }
}

// Return the full-range maximum for a runtime scale dtype.
inline float scale_max(const DType scale_dtype) {
  float result = 0.0f;
  TRANSFORMER_ENGINE_NVFP4_SCALE_TYPE_SWITCH(scale_dtype, ScaleType,
                                             result = scale_max<ScaleType>();)
  return result;
}

// Return and validate a user-provided maximum for a runtime scale dtype.
inline float scale_max(const DType scale_dtype, const int scale_type_max) {
  float result = 0.0f;
  TRANSFORMER_ENGINE_NVFP4_SCALE_TYPE_SWITCH(scale_dtype, ScaleType, {
    using ScaleTraits = NVFP4ScaleTraits<ScaleType>;
    NVTE_CHECK(scale_type_max == static_cast<int>(ScaleTraits::expected_max) ||
                   scale_type_max == static_cast<int>(ScaleTraits::headroom_max),
               "Unsupported maximum for NVFP4 scale dtype.");
    result = static_cast<float>(scale_type_max);
  })
  return result;
}

#endif  // FP4_TYPE_SUPPORTED

}  // namespace core
}  // namespace nvfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_NVFP4_SCALE_TRAITS_CUH_
