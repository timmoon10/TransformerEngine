/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transformer_engine.h>

#include <iostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>

#include "common.h"

namespace transformer_engine {

size_t typeToSize(const DType type) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(type, T,
                                     return TypeInfo<T>::size;);  // NOLINT(*)
}

bool is_fp8_dtype(const DType t) { return t == DType::kFloat8E4M3 || t == DType::kFloat8E5M2; }

std::string to_string(const DType type) {
  switch (type) {
    case DType::kByte:
      return "Byte";
    case DType::kBFloat16:
      return "BFloat16";
    case DType::kFloat16:
      return "Float16";
    case DType::kFloat32:
      return "Float32";
    case DType::kFloat8E4M3:
      return "Float8E4M3";
    case DType::kFloat8E5M2:
      return "Float8E5M2";
    case DType::kFloat8E8M0:
      return "Float8E8M0";
    case DType::kInt32:
      return "Int32";
    case DType::kInt64:
      return "Int64";
    default:
      return concat_strings("Invalid type ", static_cast<int>(type));
  }
}

std::string to_string(const NVTEScalingMode &mode) {
  switch (mode) {
    case NVTE_DELAYED_TENSOR_SCALING:
      return "Delayed Tensor Scaling";
    case NVTE_MXFP8_1D_SCALING:
      return "MXFP8 1D Scaling";
    case NVTE_INVALID_SCALING:
      return "Invalid Scaling";
  }
  return "Invalid Scaling";
}

void CheckNoopTensor(const Tensor &t, const std::string &name) {
  NVTE_CHECK(t.data.index == 0,
             "Expected noop tensor \"", name, "\" to have simple data, but found format ",
             t.data.index, ". Scaling mode is ", to_string(t.scaling_mode()), ".");
  if (t.has_data()) {
    NVTE_CHECK(t.numel() == 1,
               "Expected noop tensor \"", name, "\" to have one entry, but found shape=", t.shape());
    NVTE_CHECK(t.dtype() == DType::kFloat32,
               "Expected noop tensor \"", name, "\" to have dtype=Float32, but found dtype=",
               to_string(t.dtype()));
  }
}

namespace {

void CheckFP8Tensor(const Tensor &t, const Tensor::FP8Data &data, const std::string &name) {
  // Either data or transpose data are required
  NVTE_CHECK(data.data || data.transpose_data,
             "FP8 tensor \"", name, "\" is missing both data and transpose data");

  // Check data
  if (data.data) {
    NVTE_CHECK(data.data->dptr != nullptr,
               "FP8 tensor \"", name, "\" has unallocated data");
    NVTE_CHECK(is_fp8_dtype(data.data->dtype),
               "FP8 tensor \"", name, "\" has data with dtype=", to_string(data.data->dtype));
  }

  // Check transpose data
  if (data.transpose_data) {
    NVTE_CHECK(data.data_transpose->dptr != nullptr,
               "FP8 tensor \"", name, "\" has unallocated data transpose");
    NVTE_CHECK(is_fp8_dtype(data.transpose_data->dtype),
               "FP8 tensor \"", name, "\" has transpose data with dtype=",
               to_string(data.transpose_data->dtype));
    const auto& shape = t.shape();
    const auto& transpose_shape = data.transpose_data->shape;
    bool is_transpose_shape_valid = shape.size() == transpose_shape.size();
    if (is_transpose_shape_valid) {
      for (size_t i=0; i < transpose_shape.size(); i++) {
        const auto& d1 = transpose_shape[i];
        const auto& d2 = i == 0 ? shape.back() : shape[i-1];
        if (d1 != d2) {
          is_transpose_shape_valid = false;
          break;
        }
      }
    }
    NVTE_CHECK(!is_transpose_shape_valid,
               "FP8 tensor \"", name, "\" has transpose data with invalid shape (data has shape=",
               shape, ", data transpose has shape=", transpose_shape, ")");
  }

  // Check scale-inverse
  NVTE_CHECK(data.scale_inv.dptr != nullptr,
             "FP8 tensor \"", name, "\" has unallocated scale-inverse");
  NVTE_CHECK(data.scale_inv.numel() == 1,
             "FP8 tensor \"", name, "\" has invalid scale-inverse shape (expected (1), got ",
             data.scale_inv.shape, ")");
  NVTE_CHECK(data.scale_inv.dtype == DType::kFloat32,
             "FP8 tensor \"", name, "\" has invalid scale-inverse dtype (expected Float32, got ",
             to_string(data.scale_inv.dtype));

  // Check scale
  if (data.scale) {
    NVTE_CHECK(data.scale->dptr != nullptr,
               "FP8 tensor \"", name, "\" has unallocated scale");
    NVTE_CHECK(data.scale->numel() == 1,
               "FP8 tensor \"", name, "\" has invalid scale shape (expected (1), got ",
               data.scale->shape, ")");
    NVTE_CHECK(data.scale->dtype == DType::kFloat32,
               "FP8 tensor \"", name, "\" has invalid scale dtype (expected Float32, got ",
               to_string(data.scale->dtype));
  }

  // Check amax
  if (data.amax) {
    NVTE_CHECK(data.amax->dptr != nullptr,
               "FP8 tensor \"", name, "\" has unallocated amax");
    NVTE_CHECK(data.amax->numel() == 1,
               "FP8 tensor \"", name, "\" has invalid amax shape (expected (1), got ",
               data.amax->shape, ")");
    NVTE_CHECK(data.amax->dtype == DType::kFloat32,
               "FP8 tensor \"", name, "\" has invalid amax dtype (expected Float32, got ",
               to_string(data.amax->dtype));
  }
}

void CheckMXFP8Tensor(const Tensor &t, const Tensor::MXFP8Data &data, const std::string &name) {
  // Either row-scaled data or column-scaled data are required
  NVTE_CHECK(data.row_scaled_data || data.column_scaled_data,
             "MXFP8 tensor \"", name,
             "\" is missing both row-scaled data and column-scaled data");

  // Data dimensions must be divisible by 32
  const size_t flat_x = t.flat_first_dim();
  const size_t flat_y = t.flat_last_dim();
  NVTE_CHECK(flat_x % 32 == 0 && flat_y % 32 == 0, "MXFP8 tensor \"", name,
             "\" requires dims that are divisible by 32, but got shape=", t.shape());

  // Check row-scaled data
  if (data.row_scaled_data) {
    SimpleTensor fp8_data, scale_inv;
    std::tie(fp8_data, scale_inv) = *data.row_scaled_data;

    // Check row-scaled FP8 data
    NVTE_CHECK(fp8_data.dptr != nullptr,
               "MXFP8 tensor \"", name, "\" has unallocated row-scaled data");
    NVTE_CHECK(is_fp8_dtype(fp8_data.dtype),
               "MXFP8 tensor \"", name, "\" has row-scaled data with dtype=",
               to_string(fp8_data.dtype));

    // Check scale-inverse for row-scaled data
    NVTE_CHECK(scale_inv.dptr != nullptr, "MXFP8 tensor \"", name,
               "\" has unallocated scale-inverses for row-scaled data");
    NVTE_CHECK(scale_inv.dtype == DType::kFloat8EM0,
               "MXFP8 tensor \"", name, "\" has invalid scale-inverses for row-scaled data ",
               "(expected dtype=Float8E8M0, got dtype=", to_string(scale_inv.dtype), ")");
    const size_t alignment_x = 128;
    const size_t expected_x = DIVUP(DIVUP(flat_x, 1ull), alignment_x) * alignment_x;
    const size_t alignment_y = 4;
    const size_t expected_y = DIVUP(DIVUP(flat_y, 32ull), alignment_y) * alignment_y;
    NVTE_CHECK(scale_inv.shape.size() == 2
               && scale_inv.shape[0] == expected_x
               && scale_inv.shape[1] == expected_y,
               "MXFP8 tensor \"", name, "\" has invalid scale-inverses for row-scaled data "
               "(expected shape=(", alignment_x, ", ", alignment_y, "), got shape=",
               scale_inv.shape, ")");
  }

  // Check column-scaled data
  if (data.column_scaled_data) {
    SimpleTensor fp8_data, scale_inv;
    std::tie(fp8_data, scale_inv) = *data.column_scaled_data;

    // Check column-scaled FP8 data
    NVTE_CHECK(fp8_data.dptr != nullptr,
               "MXFP8 tensor \"", name, "\" has unallocated column-scaled data");
    NVTE_CHECK(is_fp8_dtype(fp8_data.dtype),
               "MXFP8 tensor \"", name, "\" has column-scaled data with dtype=",
               to_string(fp8_data.dtype));
    NVTE_CHECK(fp8_data.shape == t.shape(), "MXFP8 tensor \"", name,
               "\" has invalid shape for column-scaled data (expected ", t.shape(), ", got ",
               fp8_data.shape, ")");

    // Check scale-inverse for column-scaled data
    NVTE_CHECK(scale_inv.dptr != nullptr, "MXFP8 tensor \"", name,
               "\" has unallocated scale-inverses for column-scaled data");
    NVTE_CHECK(scale_inv.dtype == DType::kFloat8EM0,
               "MXFP8 tensor \"", name, "\" has invalid scale-inverses for column-scaled data ",
               "(expected dtype=Float8E8M0, got dtype=", to_string(scale_inv.dtype), ")");
    const size_t alignment_x = 4;
    const size_t expected_x = DIVUP(DIVUP(flat_x, 32ull), alignment_x) * alignment_x;
    const size_t alignment_y = 128;
    const size_t expected_y = DIVUP(DIVUP(flat_y, 1ull), alignment_y) * alignment_y;
    NVTE_CHECK(scale_inv.shape.size() == 2
               && scale_inv.shape[0] == expected_x
               && scale_inv.shape[1] == expected_y,
               "MXFP8 tensor \"", name, "\" has invalid column scale-inverse shape (expected (",
               alignment_x, ", ", alignment_y, "), got ", scale_inv.shape, ")");
  }
}

}  // namespace

void CheckTensor(const Tensor &t, const std::string &name) {
  auto visitor = [&t, &name] (const auto& data) -> void {
    using DataType = std::decay_t<decltype(data)>;
    if constexpr (std::is_same_v<DataType, Tensor::SimpleData>) {
      NVTE_CHECK(!is_fp8_dtype(data.data.dtype),
                 "Tensor \"", name, "\" has simple format, but dtype=",
                 to_string(data.data.dtype));
    }
    if constexpr (std::is_same_v<DataType, Tensor::FP8Data>) {
      CheckFP8Tensor(t, data, name);
    }
    if constexpr (std::is_same_v<DataType, Tensor::MXFP8Data>) {
      CheckMXFP8Tensor(t, data, name);
    }
  }
  std::visit(visitor, t.data);
}

void CheckInputTensor(const Tensor &t, const std::string &name) {
  CheckTensor(t, name);
}

void CheckOutputTensor(const Tensor &t, const std::string &name) {
  CheckTensor(t, name);
}

}  // namespace transformer_engine

NVTETensor nvte_create_tensor(NVTEScalingMode scaling_mode) {
  transformer_engine::Tensor *ret = new transformer_engine::Tensor;
  switch (scaling_mode) {
  case NVTE_DELAYED_TENSOR_SCALING:
    // The user-facing API treats simple tensors and FP8 tensors as
    // the same. For now we initialize an FP8 tensor as the default,
    // but consider adding option for simple tensors.
    ret->data = Tensor::FP8Data{};
    break;
  case NVTE_MXFP8_1D_SCALING:
    ret->data = Tensor::MXFP8Data{};
    break;
  default:
    NVTE_ERROR("Unexpected scaling mode (", to_string(scaling_mode), ")");
  }
  return ret;
}

void nvte_destroy_tensor(NVTETensor tensor) {
  if (tensor == nullptr) return;
  auto *t = reinterpret_cast<transformer_engine::Tensor *>(tensor);
  delete t;
}

NVTEDType nvte_tensor_type(const NVTETensor tensor) {
  if (tensor == nullptr) return kNVTEFloat32;
  return static_cast<NVTEDType>(
      reinterpret_cast<const transformer_engine::Tensor *>(tensor)->dtype());
}

NVTEShape nvte_tensor_shape(const NVTETensor tensor) {
  if (tensor == nullptr) return {nullptr, 0};
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  t._shape = t.shape();
  NVTEShape ret;
  ret.data = t._shape.data();
  ret.ndim = t._shape.size();
  return ret;
}

NVTEShape nvte_tensor_columnwise_shape(const NVTETensor tensor) {
  if (tensor == nullptr) return {nullptr, 0};
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  const auto& shape = t.shape();
  t._columnwise_shape.clear();
  if (!shape.empty()) {
    t._columnwise_shape.push_back(shape.back());
    for (size_t i=0; i < shape.size() - 1; i++) {
      t._columnwise_shape.push_back(shape[i]);
    }
  }
  NVTEShape ret;
  ret.data = t._columnwise_shape.data();
  ret.ndim = t._columnwise_shape.size();
  return ret;
}

size_t nvte_tensor_ndim(const NVTETensor tensor) {
  if (tensor == nullptr) return 0;
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  return t.shape().size();
}

size_t nvte_tensor_size(const NVTETensor tensor, const size_t dim) {
  if (tensor == nullptr) return 0;
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  const auto &shape = t.shape();
  NVTE_CHECK(dim >= 0 && dim < shape.size(),
             "Attempted to access dim ", dim, " in a ", shape.size(), "-D tensor");
  return shape[dim];
}

size_t nvte_tensor_numel(const NVTETensor tensor) {
  if (tensor == nullptr) return 0;
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  return t.numel();
}

size_t nvte_tensor_element_size(const NVTETensor tensor) {
  if (tensor == nullptr) return sizeof(float);
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  return transformer_engine::typeToSize(t.dtype());
}

void *nvte_tensor_data(const NVTETensor tensor) {
  if (tensor == nullptr) return nullptr;
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  auto visitor = [] (const auto& data) -> void * {
    using DataType = std::decay_t<decltype(data)>;
    if constexpr (std::is_same_v<DataType, SimpleData>) {
      return data.data.dptr;
    }
    if constexpr (std::is_same_v<DataType, FP8Data>) {
      if (data.data) {
        return data.data->dptr;
      }
    }
    if constexpr (std::is_same_v<DataType, MXFP8Data>) {
      if (data.row_scaled_data) {
        return data.row_scaled_data->dptr;
      }
    }
    return nullptr;
  };
  return std::visit(visitor, t.data);
}

void *nvte_tensor_columnwise_data(const NVTETensor tensor) {
  if (tensor == nullptr) return nullptr;
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  auto visitor = [] (const auto& data) -> void * {
    using DataType = std::decay_t<decltype(data)>;
    if constexpr (std::is_same_v<DataType, FP8Data>) {
      if (data.transpose_data) {
        return data.transpose_data->dptr;
      }
    }
    if constexpr (std::is_same_v<DataType, MXFP8Data>) {
      if (data.column_scaled_data) {
        return data.column_scaled_data->dptr;
      }
    }
    return nullptr;
  };
  return std::visit(visitor, t.data);
}

float *nvte_tensor_amax(const NVTETensor tensor) {
  if (tensor == nullptr) return nullptr;
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  auto visitor = [&t] (const auto& data) -> float * {
    using DataType = std::decay_t<decltype(data)>;
    if constexpr (std::is_same_v<DataType, FP8Data>) {
      if (data.amax) {
        NVTE_CHECK(data.amax->numel() == 1,
                   "FP8 tensor \"", name, "\" has invalid amax shape (expected (1), got ",
                   data.amax->shape, ")");
        NVTE_CHECK(data.amax->dtype == DType::kFloat32,
                   "FP8 tensor \"", name, "\" has invalid amax dtype (expected Float32, got ",
                   to_string(data.amax->dtype));
        return reinterpret_cast<float*>(data.amax->dptr);
      } else {
        return static_cast<float*>(nullptr);
      }
    }
    NVTE_ERROR("Attempted to access amax pointer, which is only supported for FP8 tensors ",
               "(scaling mode is ", to_string(t.scaling_mode), ", dtype is ", to_string(t.dtype()),
               ")");
    return static_cast<float*>(nullptr);  // dummy
  };
  return std::visit(visitor, t.data);
}

float *nvte_tensor_scale(const NVTETensor tensor) {
  if (tensor == nullptr) return nullptr;
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  auto visitor = [&t] (const auto& data) -> float * {
    using DataType = std::decay_t<decltype(data)>;
    if constexpr (std::is_same_v<DataType, FP8Data>) {
      if (data.scale) {
        NVTE_CHECK(data.scale->numel() == 1,
                   "FP8 tensor \"", name, "\" has invalid scale shape (expected (1), got ",
                   data.scale->shape, ")");
        NVTE_CHECK(data.scale->dtype == DType::kFloat32,
                   "FP8 tensor \"", name, "\" has invalid scale dtype (expected Float32, got ",
                   to_string(data.scale->dtype));
        return reinterpret_cast<float*>(data.scale->dptr);
      } else {
        return static_cast<float*>(nullptr);
      }
    }
    NVTE_ERROR("Attempted to access scale pointer, which is only supported for FP8 tensors ",
               "(scaling mode is ", to_string(t.scaling_mode), ", dtype is ", to_string(t.dtype()),
               ")");
    return static_cast<float*>(nullptr);  // dummy
  };
  return std::visit(visitor, t.data);
}

float *nvte_tensor_scale_inv(const NVTETensor tensor) {
  if (tensor == nullptr) return nullptr;
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  auto visitor = [&t] (const auto& data) -> float * {
    using DataType = std::decay_t<decltype(data)>;
    if constexpr (std::is_same_v<DataType, FP8Data>) {
      NVTE_CHECK(data.scale_inv.numel() == 1,
                 "FP8 tensor \"", name, "\" has invalid scale shape (expected (1), got ",
                 data.scale_inv.shape, ")");
      NVTE_CHECK(data.scale_inv.dtype == DType::kFloat32,
                 "FP8 tensor \"", name, "\" has invalid scale dtype (expected Float32, got ",
                 to_string(data.scale_inv.dtype));
      return reinterpret_cast<float*>(data.scale_inv.dptr);
    }
    if constexpr (std::is_same_v<DataType, MXFP8Data>) {
      if (data.row_scaled_data) {
        const auto& scale_inv = std::get<1>(*data.row_scaled_data);
        NVTE_CHECK(scale_inv.dtype == DType::kFloat8EM0,
                   "MXFP8 tensor \"", name, "\" has invalid scale-inverses for row-scaled data ",
                   "(expected dtype=Float8E8M0, got dtype=", to_string(scale_inv.dtype), ")");
        return reinterpret_cast<float*>(scale_inv.dptr);  // TODO Change API to output void*
      } else {
        return static_cast<float*>(nullptr);
      }
    }
    NVTE_ERROR("Attempted to access scale-inverse pointer, which is only supported for "
               "FP8 or MXFP8 tensors (scaling mode is ", to_string(t.scaling_mode), ", dtype is ",
               to_string(t.dtype()), ")");
    return static_cast<float*>(nullptr);  // dummy
  };
  return std::visit(visitor, t.data);
}

void *nvte_tensor_columnwise_scale_inv(const NVTETensor tensor) {
  if (tensor == nullptr) return nullptr;
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  auto visitor = [&t] (const auto& data) -> void * {
    using DataType = std::decay_t<decltype(data)>;
    if constexpr (std::is_same_v<DataType, FP8Data>) {
      NVTE_CHECK(data.scale_inv.numel() == 1,
                 "FP8 tensor \"", name, "\" has invalid scale shape (expected (1), got ",
                 data.scale_inv.shape, ")");
      NVTE_CHECK(data.scale_inv.dtype == DType::kFloat32,
                 "FP8 tensor \"", name, "\" has invalid scale dtype (expected Float32, got ",
                 to_string(data.scale_inv.dtype));
      return reinterpret_cast<float*>(data.scale_inv.dptr);
    }
    if constexpr (std::is_same_v<DataType, MXFP8Data>) {
      if (data.column_scaled_data) {
        const auto& scale_inv = std::get<1>(*data.column_scaled_data);
        NVTE_CHECK(scale_inv.dtype == DType::kFloat8EM0,
                   "MXFP8 tensor \"", name, "\" has invalid scale-inverses for column-scaled data ",
                   "(expected dtype=Float8E8M0, got dtype=", to_string(scale_inv.dtype), ")");
        return reinterpret_cast<float*>(scale_inv.dptr);  // TODO Change API to return void*
      } else {
        return static_cast<float*>(nullptr);
      }
    }
    NVTE_ERROR("Attempted to access column-wise scale-inverse pointer, "
               "which is only supported for FP8 and MXFP8 tensors (scaling mode is ",
               to_string(t.scaling_mode), ", dtype is ", to_string(t.dtype()), ")");
    return nullptr;  // dummy
  };
  return std::visit(visitor, t.data);
}

NVTEShape nvte_tensor_scale_inv_shape(const NVTETensor tensor) {
  if (tensor == nullptr) return {nullptr, 0};
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  auto visitor = [&t] (const auto& data) -> std::vector<size_t> {
    using DataType = std::decay_t<decltype(data)>;
    if constexpr (std::is_same_v<DataType, FP8Data>) {
      NVTE_CHECK(data.scale_inv.numel() == 1,
                 "FP8 tensor \"", name, "\" has invalid scale shape (expected (1), got ",
                 data.scale_inv.shape, ")");
      return data.scale_inv.shape;
    }
    if constexpr (std::is_same_v<DataType, MXFP8Data>) {
      if (data.row_scaled_data) {
        const auto& scale_inv = std::get<1>(*data.row_scaled_data);
        return scale_inv.shape;
      } else {
        return {0ull};
      }
    }
    NVTE_ERROR("Attempted to access scale-inverse pointer, which is only supported for "
               "FP8 or MXFP8 tensors (scaling mode is ", to_string(t.scaling_mode), ", dtype is ",
               to_string(t.dtype()), ")");
    return {0ull};  // dummy
  };
  t._scale_inv_shape = std::visit(visitor, t.data);
  NVTEShape ret;
  ret.data = t._scale_inv_shape.data();
  ret.ndim = t._scale_inv_shape.size();
  return ret;
}

namespace transformer_engine {
namespace {

void set_fp8_tensor_param(Tensor::FP8Data &data, NVTETensorParam param_name, BasicTensor &&param) {
  switch (param_name) {
    case kNVTERowwiseData:
      data.data = std::move(param);
      break;
    case kNVTEColumnwiseData:
      data.transpose_data = std::move(param);
      break;
    case kNVTEScale:
      data.scale = std::move(param);
      break;
    case kNVTEAmax:
      data.amax = std::move(param);
      break;
    case kNVTERowwiseScaleInv:
    case kNVTEColumnwiseScaleInv:
      data.scale_inv = std::move(param);
      break;
    default:
      NVTE_ERROR("Unsupported parameter for FP8 tensor (", (int)param_name, ")");
  }
}

void set_mxfp8_tensor_param(Tensor::MXFP8Data &data, NVTETensorParam param_name, BasicTensor &&param) {
  auto maybe_make_row_scaled_data = [&data] () -> void {
    if (!data.row_scaled_data) {
      data.row_scaled_data.emplace(std::make_tuple(SimpleTensor{}, SimpleTensor{}));
    }
  };
  auto maybe_make_column_scaled_data = [&data] () -> void {
    if (!data.column_scaled_data) {
      data.column_scaled_data.emplace(std::make_tuple(SimpleTensor{}, SimpleTensor{}));
    }
  };
  switch (param_name) {
    case kNVTERowwiseData:
      maybe_make_row_scaled_data()
      std::get<0>(*data.row_scaled_data) = std::move(param);
      break;
    case kNVTEColumnwiseData:
      maybe_make_column_scaled_data()
      std::get<0>(*data.column_scaled_data) = std::move(param);
      break;
    case kNVTERowwiseScaleInv:
      maybe_make_row_scaled_data()
      std::get<1>(*data.row_scaled_data) = std::move(param);
      break;
    case kNVTEColumnwiseScaleInv:
      maybe_make_column_scaled_data()
      std::get<1>(*data.column_scaled_data) = std::move(param);
      break;
    default:
      NVTE_ERROR("Unsupported parameter for MXFP8 tensor (", (int)param_name, ")");
  }
}

}  // namespace
}  // namespace transformer_engine

void nvte_set_tensor_param(NVTETensor *tensor, NVTETensorParam param_name,
                           const NVTEBasicTensor *param) {
  NVTE_CHECK(tensor != nullptr, "Got null pointer to tensor.");
  NVTE_CHECK(*tensor != nullptr, "Tensor is not allocated.");
  NVTE_CHECK(param != nullptr, "Got null pointer to param.");
  auto &t = *reinterpret_cast<transformer_engine::Tensor *>(*tensor);
  SimpleTensor p = *param;

  // The user-facing API treats simple tensors and FP8 tensors as the
  // same, so the intended layout is ambiguous at construction time.
  // Reset the layout data if the user-provided param is not
  // compatible.
  if (t.scaling_mode() == NVTE_DELAYED_TENSOR_SCALING) {
    if (t.data.index == 1 && param_name == kNVTERowwiseData && !is_fp8_dtype(p.dtype)) {
      // FP8 tensor only supports FP8 data. Make simple tensor if
      // non-FP8 data is provided.
      t.data = Tensor::SimpleData{};
    }
    if (t.data.index == 0 && (param_name != kNVTERowwiseData || is_fp8_dtype(p.dtype))) {
      // Simple tensor only supports a single data tensor with non-FP8
      // dtype. Make FP8 tensor if anything else is provided.
      t.data = Tensor::FP8Data{};
    }
  }

  // Visitor pattern to set tensor param
  auto visitor = [&t, &param_name, &p] (const auto& data) -> std::vector<size_t> {
    using DataType = std::decay_t<decltype(data)>;
    if constexpr (std::is_same_v<DataType, SimpleData>) {
      if (param_name == kNVTERowwiseData) {
        data.data = std::move(p);
      } else {
        NVTE_ERROR("Unsupported parameter for simple tensor (", (int)param_name, ")");
      }
    } else if constexpr (std::is_same_v<DataType, FP8Data>) {
      set_fp8_tensor_param(data, param_name, std::move(p));
    } else if constexpr (std::is_same_v<DataType, FP8Data>) {
      set_mxfp8_tensor_param(data, param_name, std::move(p));
    } else {
      NVTE_ERROR("Could set tensor parameter because tensor format is not supported");
    }
  };
  std::visit(visitor, t.data);
}

NVTEBasicTensor nvte_get_tensor_param(const NVTETensor tensor, NVTETensorParam param_name) {
  if (tensor == nullptr) {
    return {nullptr, kNVTEFloat32, {nullptr, 0}};
  }
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  switch (param_name) {
    case kNVTERowwiseData:
      return t.data;
    case kNVTEColumnwiseData:
      return t.columnwise_data;
    case kNVTEScale:
      return t.scale;
    case kNVTEAmax:
      return t.amax;
    case kNVTERowwiseScaleInv:
      return t.scale_inv;
    case kNVTEColumnwiseScaleInv:
      return t.columnwise_scale_inv;
    default:
      NVTE_ERROR("Unknown tensor parameter!");
  }
}

NVTEScalingMode nvte_tensor_scaling_mode(const NVTETensor tensor) {
  /// TODO Impl
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  return t.scaling_mode();
}

void nvte_tensor_pack_create(NVTETensorPack *pack) {
  for (int i = 0; i < pack->MAX_SIZE; i++) {
    pack->tensors[i] = reinterpret_cast<NVTETensor>(new transformer_engine::Tensor);
  }
}

void nvte_tensor_pack_destroy(NVTETensorPack *pack) {
  for (int i = 0; i < pack->MAX_SIZE; i++) {
    auto *t = reinterpret_cast<transformer_engine::Tensor *>(pack->tensors[i]);
    delete t;
  }
}

void nvte_zero_tensor(const NVTETensor tensor, cudaStream_t stream) {
  /// TODO Impl
  const auto &t = *reinterpret_cast<const transformer_engine::Tensor *>(tensor);
  // Zero out tensor data if allocated
  if (t.data.dptr != nullptr) {
    size_t size_in_bytes = nvte_tensor_element_size(tensor) * nvte_tensor_numel(tensor);
    cudaMemsetAsync(t.data.dptr, 0, size_in_bytes, stream);
  }
  // Set amax to 0 if allocated
  if (t.amax.dptr != nullptr) {
    float zero = 0.0f;
    cudaMemcpyAsync(t.amax.dptr, &zero, sizeof(float), cudaMemcpyHostToDevice, stream);
  }
  cudaStreamSynchronize(stream);
}
