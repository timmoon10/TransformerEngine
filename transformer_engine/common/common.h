/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_COMMON_H_
#define TRANSFORMER_ENGINE_COMMON_COMMON_H_

#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <transformer_engine/transformer_engine.h>
#include "./nvtx.h"
#include "./util/cuda_driver.h"
#include "./util/logging.h"

namespace transformer_engine {

inline size_t product(const std::vector<size_t> &shape, const size_t begin, const size_t end) {
  NVTE_CHECK(begin <= end && end <= shape.size(), "Attempted to access entries ", begin, " to ",
             end, " in a vector with ", shape.size(), " entries");
  size_t ret = 1;
  for (size_t i = begin; i < end; ++i) {
    ret *= shape[i];
  }
  return ret;
}

inline size_t product(const std::vector<size_t> &shape) {
  size_t ret = 1;
  for (const auto &elem : shape) {
    ret *= elem;
  }
  return ret;
}

struct SimpleTensor {
  void *dptr;
  std::vector<size_t> shape;
  DType dtype;

  SimpleTensor(void *dptr, std::vector<size_t> shape, DType dtype)
      : dptr{dptr}, shape{std::move(shape)}, dtype{dtype} {}

  SimpleTensor(const NVTEBasicTensor &tensor)
      : SimpleTensor(tensor.data_ptr,
                     std::vector<size_t>(tensor.shape.data, tensor.shape.data + tensor.shape.ndim),
                     static_cast<DType>(tensor.dtype)) {}

  SimpleTensor() : SimpleTensor(nullptr, {static_cast<size_t>(0)}, DType::kFloat32) {}

  operator NVTEBasicTensor() const {
    const NVTEShape shape = {this->shape.data(), this->shape.size()};
    return {dptr, static_cast<NVTEDType>(dtype), shape};
  }

  size_t numel() const {
    return product(shape);
  }
};

struct Tensor {

  struct SimpleData {
    SimpleTensor data;
  };

  struct FP8Data {
    std::optional<SimpleTensor> data;
    std::optional<SimpleTensor> transpose_data;
    SimpleTensor scale_inv;
    std::optional<SimpleTensor> scale;
    std::optional<SimpleTensor> amax;
  };

  struct MXFP8Data {
    std::optional<std::tuple<SimpleTensor, SimpleTensor>> row_scaled_data;
    std::optional<std::tuple<SimpleTensor, SimpleTensor>> column_scaled_data;
  };

  std::variant<SimpleData, FP8Data, MXFP8Data> data;

  Tensor() : data{} {}

  std::vector<size_t> shape() const {
    auto visitor = [] (const auto& data) -> std::vector<size_t> {
      using DataType = std::decay_t<decltype(data)>;
      if constexpr (std::is_same_v<DataType, SimpleData>) {
        return data.data.shape;
      }
      if constexpr (std::is_same_v<DataType, FP8Data>) {
        if (data.data) {
          return data.data->shape;
        } else if (data.transpose_data) {
          const auto& transpose_shape = data.transpose_data->shape;
          std::vector<size_t> ret;
          if (!transpose_shape.empty()) {
            for (size_t i=1; i<transpose_shape.size(); i++) {
              ret.push_back(transpose_shape[i]);
            }
            ret.push_back(transpose_shape[0]);
          }
          return ret;
        } else {
          NVTE_ERROR("Could not determine shape because "
                     "FP8 tensor is missing both data and transpose data");
        }
      }
      if constexpr (std::is_same_v<DataType, FP8Data>) {
        if (data.row_scaled_data) {
          return std::get<0>(*data.row_scaled_data).shape;
        } else if (data.column_scaled_data) {
          return std::get<0>(*data.column_scaled_data).shape;
        } else {
          NVTE_ERROR("Could not determine shape because "
                     "MXFP8 tensor is missing both row-scaled data and column-scaled data");
        }
      }
      NVTE_ERROR("Could not determine shape because tensor format is not supported");
      return {};  // dummy
    };
    return std::visit(visitor, data);
  }

  size_t numel() const {
    return product(shape());
  }

  bool is_empty() const {
    for (const auto &elem : shape()) {
      if (elem == 0) {
        return true;
      }
    }
    return false;
  }

  NVTEScalingMode scaling_mode() const {
    auto visitor = [] (const auto& data) -> NVTEScalingMode {
      using DataType = std::decay_t<decltype(data)>;
      if constexpr (std::is_same_v<DataType, SimpleData>) {
        // The user-facing API treats simple tensors and FP8 tensors
        // as the same. Consider adding option for simple tensors.
        return NVTE_DELAYED_TENSOR_SCALING;
      }
      if constexpr (std::is_same_v<DataType, FP8Data>) {
        return NVTE_DELAYED_TENSOR_SCALING;
      }
      if constexpr (std::is_same_v<DataType, MXFP8Data>) {
        return NVTE_MXFP8_1D_SCALING;
      }
      return NVTE_INVALID_SCALING;
    };
    return std::visit(visitor, data);
  }

  bool has_data() const {
    if (is_empty()) {
      return false;
    }
    auto visitor = [] (const auto& data) -> bool {
      using DataType = std::decay_t<decltype(data)>;
      if constexpr (std::is_same_v<DataType, SimpleData>) {
        return data.dptr != nullptr;
      }
      if constexpr (std::is_same_v<DataType, FP8Data>) {
        return data.data && data.data->dptr != nullptr;
      }
      if constexpr (std::is_same_v<DataType, MXFP8Data>) {
        return data.row_scaled_data && std::get<0>(*data.row_scaled_data).dtpr != nullptr;
      }
      return false;
    };
    return std::visit(visitor, data);
  }

  bool has_columnwise_data() const {
    if (is_empty()) {
      return false;
    }
    auto visitor = [] (const auto& data) -> bool {
      using DataType = std::decay_t<decltype(data)>;
      if constexpr (std::is_same_v<DataType, FP8Data>) {
        return data.transpose_data && data.transpose_data->dptr != nullptr;
      }
      if constexpr (std::is_same_v<DataType, MXFP8Data>) {
        return data.column_scaled_data && std::get<0>(*data.column_scaled_data).dtpr != nullptr;
      }
      return false;
    };
    return std::visit(visitor, data);
  }

  DType dtype() const {
    auto visitor = [] (const auto& data) -> DType {
      using DataType = std::decay_t<decltype(data)>;
      if constexpr (std::is_same_v<DataType, SimpleData>) {
        return data.data.dtype;
      }
      if constexpr (std::is_same_v<DataType, FP8Data>) {
        if (data.data) {
          return data.data.dtype;
        } else if (data.transpose_data) {
          return data.transpose_data.dtype;
        } else {
          NVTE_ERROR("Could not determine dtype because "
                     "FP8 tensor is missing both data and transpose data");
        }
      }
      if constexpr (std::is_same_v<DataType, MXFP8Data>) {
        if (data.row_scaled_data) {
          return std::get<0>(*data.row_scaled_data).dtype;
        } else if (data.column_scaled_data) {
          return std::get<0>(*data.column_scaled_data).dtype;
        } else {
          NVTE_ERROR("Could not determine dtype because "
                     "MXFP8 tensor is missing both row-scaled data and column-scaled data");
        }
      }
      NVTE_ERROR("Could not determine dtype because tensor format is not supported");
      return DType::kFloat32;  // dummy
    };
    return std::visit(visitor, data);
  }

  /*! Matrix height after tensor is flattened to 2D
   *
   * If a tensor has dimensions (D1, D2, ..., Dn), it is reinterpreted
   * as a (D1*D2*...*D(n-1), Dn) matrix.
   */
  size_t flat_first_dim() const {
    const auto& full_shape = shape();
    if (full_shape.empty()) {
      return 1;
    }
    return product(full_shape, 0, full_shape.size() - 1);
  }

  /*! Matrix width after tensor is flattened to 2D
   *
   * If a tensor has dimensions (D1, D2, ..., Dn), it is reinterpreted
   * as a (D1*D2*...*D(n-1), Dn) matrix.
   */
  size_t flat_last_dim() const {
    const auto& full_shape = shape();
    if (full_shape.empty()) {
      return 1;
    }
    return full_shape.back();
  }

private:

  /*! Kludge for persistently storing data shape
   *
   * The tensor shape class in the user-facing API (NVTEShape) just
   * consists of a pointer and size, so the shape data must be managed
   * elsewhere. This is a temporary hack until we make NVTEShape
   * manage its own memory.
   */
  mutable std::vector<size_t> _shape;
  /*! Kludge for persistently storing column-wise data shape
   *
   * The tensor shape class in the user-facing API (NVTEShape) just
   * consists of a pointer and size, so the shape data must be managed
   * elsewhere. This is a temporary hack until we make NVTEShape
   * manage its own memory.
   */
  mutable std::vector<size_t> _columnwise_shape;
  /*! Kludge for persistently storing scale-inverse shape
   *
   * The tensor shape class in the user-facing API (NVTEShape) just
   * consists of a pointer and size, so the shape data must be managed
   * elsewhere. This is a temporary hack until we make NVTEShape
   * manage its own memory.
   */
  mutable std::vector<size_t> _scale_inv_shape;

  friend NVTEShape nvte_tensor_shape(const NVTETensor tensor);
  friend NVTEShape nvte_tensor_columnwise_shape(const NVTETensor tensor);

};


template <typename T>
constexpr T DIVUP(const T &x, const T &y) {
  return (((x) + ((y)-1)) / (y));
}

using byte = uint8_t;
using int32 = int32_t;
using int64 = int64_t;
using fp32 = float;
using fp16 = half;
using bf16 = nv_bfloat16;
using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;
#if CUDA_VERSION >= 12080
using fp8e8m0 = __nv_fp8_e8m0;
#endif
using e8m0_t = uint8_t;

namespace detail {

template <typename T>
constexpr inline const char *type_name() noexcept;
#define TRANSFORMER_ENGINE_TYPE_NAME(T)                  \
  template <>                                            \
  inline constexpr const char *type_name<T>() noexcept { \
    return #T;                                           \
  }
TRANSFORMER_ENGINE_TYPE_NAME(uint8_t)
TRANSFORMER_ENGINE_TYPE_NAME(int32_t)
TRANSFORMER_ENGINE_TYPE_NAME(int64_t)
TRANSFORMER_ENGINE_TYPE_NAME(float)
TRANSFORMER_ENGINE_TYPE_NAME(half)
TRANSFORMER_ENGINE_TYPE_NAME(nv_bfloat16)
TRANSFORMER_ENGINE_TYPE_NAME(__nv_fp8_e4m3)
TRANSFORMER_ENGINE_TYPE_NAME(__nv_fp8_e5m2)
#if CUDA_VERSION >= 12080
TRANSFORMER_ENGINE_TYPE_NAME(__nv_fp8_e8m0)
#endif
#undef TRANSFORMER_ENGINE_TYPE_NAME

}  // namespace detail

template <typename T>
struct TypeInfo {
  using types = std::tuple<byte, int32, int64, fp32, fp16, bf16, fp8e4m3, fp8e5m2>;

  template <typename U, DType current>
  struct Helper {
    constexpr static DType getType() {
      constexpr int i = static_cast<int>(current);
      if (std::is_same<U, typename std::tuple_element<i, types>::type>::value) {
        return current;
      } else {
        return Helper<U, static_cast<DType>(i + 1)>::getType();
      }
    }
  };

  template <typename U>
  struct Helper<U, DType::kNumTypes> {
    constexpr static DType getType() { return DType::kNumTypes; }
  };

  template <typename U>
  constexpr static DType getType() {
    return Helper<U, DType::kByte>::getType();
  }

  constexpr static DType dtype = getType<T>();
  constexpr static size_t size = sizeof(T);
  constexpr static const char *name = detail::type_name<T>();
};

#define TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(dtype, type, ...) \
  switch (dtype) {                                           \
    using namespace transformer_engine;                      \
    case DType::kByte: {                                     \
      using type = unsigned char;                            \
      { __VA_ARGS__ }                                        \
    } break;                                                 \
    case DType::kInt32: {                                    \
      using type = int32_t;                                  \
      { __VA_ARGS__ }                                        \
    } break;                                                 \
    case DType::kInt64: {                                    \
      using type = int64_t;                                  \
      { __VA_ARGS__ }                                        \
    } break;                                                 \
    case DType::kFloat32: {                                  \
      using type = float;                                    \
      { __VA_ARGS__ }                                        \
    } break;                                                 \
    case DType::kFloat16: {                                  \
      using type = fp16;                                     \
      { __VA_ARGS__ }                                        \
    } break;                                                 \
    case DType::kBFloat16: {                                 \
      using type = bf16;                                     \
      { __VA_ARGS__ }                                        \
    } break;                                                 \
    case DType::kFloat8E4M3: {                               \
      using type = fp8e4m3;                                  \
      { __VA_ARGS__ }                                        \
    } break;                                                 \
    case DType::kFloat8E5M2: {                               \
      using type = fp8e5m2;                                  \
      { __VA_ARGS__ }                                        \
    } break;                                                 \
    case DType::kFloat8E8M0: {                               \
      using type = byte;                                     \
      { __VA_ARGS__ }                                        \
    } break;                                                 \
    default:                                                 \
      NVTE_ERROR("Invalid type.");                           \
  }

#define TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(dtype, type, ...) \
  switch (dtype) {                                              \
    using namespace transformer_engine;                         \
    case DType::kFloat32: {                                     \
      using type = float;                                       \
      { __VA_ARGS__ }                                           \
    } break;                                                    \
    case DType::kFloat16: {                                     \
      using type = fp16;                                        \
      { __VA_ARGS__ }                                           \
    } break;                                                    \
    case DType::kBFloat16: {                                    \
      using type = bf16;                                        \
      { __VA_ARGS__ }                                           \
    } break;                                                    \
    case DType::kFloat8E5M2: {                                  \
      using type = fp8e5m2;                                     \
      { __VA_ARGS__ }                                           \
    } break;                                                    \
    case DType::kFloat8E4M3: {                                  \
      using type = fp8e4m3;                                     \
      { __VA_ARGS__ }                                           \
    } break;                                                    \
    default:                                                    \
      NVTE_ERROR("Invalid type.");                              \
  }

#define TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(dtype, type, ...) \
  switch (dtype) {                                                   \
    using namespace transformer_engine;                              \
    case DType::kFloat32: {                                          \
      using type = float;                                            \
      { __VA_ARGS__ }                                                \
    } break;                                                         \
    case DType::kFloat16: {                                          \
      using type = fp16;                                             \
      { __VA_ARGS__ }                                                \
    } break;                                                         \
    case DType::kBFloat16: {                                         \
      using type = bf16;                                             \
      { __VA_ARGS__ }                                                \
    } break;                                                         \
    default:                                                         \
      NVTE_ERROR("Invalid type.");                                   \
  }

#define TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(dtype, type, ...) \
  switch (dtype) {                                               \
    using namespace transformer_engine;                          \
    case DType::kFloat8E5M2: {                                   \
      using type = fp8e5m2;                                      \
      { __VA_ARGS__ }                                            \
    } break;                                                     \
    case DType::kFloat8E4M3: {                                   \
      using type = fp8e4m3;                                      \
      { __VA_ARGS__ }                                            \
    } break;                                                     \
    default:                                                     \
      NVTE_ERROR("Invalid type.");                               \
  }

#define TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(dtype, type, ...) \
  switch (dtype) {                                             \
    using namespace transformer_engine;                        \
    case DType::kFloat32: {                                    \
      using type = float;                                      \
      { __VA_ARGS__ }                                          \
    } break;                                                   \
    case DType::kFloat16: {                                    \
      using type = fp16;                                       \
      { __VA_ARGS__ }                                          \
    } break;                                                   \
    case DType::kBFloat16: {                                   \
      using type = bf16;                                       \
      { __VA_ARGS__ }                                          \
    } break;                                                   \
    case DType::kFloat8E5M2:                                   \
    case DType::kFloat8E4M3: {                                 \
      NVTE_ERROR("FP8 type not instantiated for input.");      \
    } break;                                                   \
    default:                                                   \
      NVTE_ERROR("Invalid type.");                             \
  }

#define TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(dtype, type, ...) \
  switch (dtype) {                                             \
    using namespace transformer_engine;                        \
    case DType::kFloat16: {                                    \
      using type = fp16;                                       \
      __VA_ARGS__;                                             \
      break;                                                   \
    }                                                          \
    case DType::kBFloat16: {                                   \
      using type = bf16;                                       \
      __VA_ARGS__;                                             \
      break;                                                   \
    }                                                          \
    default:                                                   \
      NVTE_ERROR("Invalid type for 16 bit.");                  \
  }

#define TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(SCALE_DIM, DIM, ...) \
  switch (SCALE_DIM) {                                              \
    case 1: {                                                       \
      constexpr size_t DIM = 1;                                     \
      { __VA_ARGS__ }                                               \
    } break;                                                        \
    case 32: {                                                      \
      constexpr size_t DIM = 32;                                    \
      { __VA_ARGS__ }                                               \
    } break;                                                        \
    default: {                                                      \
      NVTE_ERROR("Invalid size of the MX scaling factor.");         \
    }                                                               \
  }

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}

template <size_t B>
inline size_t alignTo(size_t x) {
  size_t r = x % B;
  if (r == 0) return x;

  return x + B - r;
}

template <typename T>
struct is_fp8 : std::false_type {};

template <>
struct is_fp8<fp8e4m3> : std::true_type {};

template <>
struct is_fp8<fp8e5m2> : std::true_type {};

// [128,4] rowwise and [4,128] colwise alignment requirements for the tensor with scaling factors
constexpr size_t scale_tensor_alignment_X_rowwise = 4;
constexpr size_t scale_tensor_alignment_Y_rowwise = 128;
constexpr size_t scale_tensor_alignment_X_colwise = 128;
constexpr size_t scale_tensor_alignment_Y_colwise = 4;

size_t typeToSize(const DType type);

void CheckNoopTensor(const Tensor &t, const std::string &name);
void CheckInputTensor(const Tensor &t, const std::string &name);
void CheckOutputTensor(const Tensor &t, const std::string &name, bool allow_empty = false);

bool is_fp8_dtype(const DType t);

std::string to_string(const DType type);
std::string to_string(const NVTEScalingMode &type);

inline bool is_tensor_scaling(const NVTEScalingMode &mode) {
  return mode == NVTE_DELAYED_TENSOR_SCALING;
}

inline bool is_block_scaling(const NVTEScalingMode &mode) {
  return mode != NVTE_DELAYED_TENSOR_SCALING;
}

inline bool is_delayed_tensor_scaling(const NVTEScalingMode &mode) {
  return is_tensor_scaling(mode);
}

inline bool is_mxfp_scaling(const NVTEScalingMode &mode) { return mode == NVTE_MXFP8_1D_SCALING; }

/*! \brief Update a tensor's FP8 scale-inverse
 *
 * The FP8 scale-inverse (dequantization scaling factor) is updated
 * with the reciprocal of the FP8 scale (quantization scaling factor).
 */
void update_tensor_scale_inv(Tensor *t, cudaStream_t stream);

#define NVTE_API_CALL(api_name) \
  transformer_engine::nvtx::NVTXWrapper _##api_name##_nvtx_wrapper(#api_name);

void checkCuDriverContext(CUstream stream);

CUtensorMapDataType get_CUtensorMapDataType(DType dtype);

inline bool isPointerAligned(const void *const ptr, const int alignment);

// Set up parameters to create TMA descriptor.
void create_2D_tensor_map(CUtensorMap &tensorMap, const SimpleTensor &tensor,
                          const uint64_t globalY, const uint64_t globalX, const uint32_t shmemY,
                          const uint32_t shmemX, const uint32_t stride_elems,
                          const uint32_t offset_elems, const size_t type_size);

bool is_supported_by_CC_100();

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_COMMON_H_
