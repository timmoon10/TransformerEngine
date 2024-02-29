/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

#include <string>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

void fused_amax_and_scale_update(const at::Tensor &amax_history,
                                 const at::Tensor &scale,
                                 const at::Tensor &scale_inv,
                                 const at::Tensor &scale_inv_mask,
                                 at::Tensor updated_amax_history,
                                 at::Tensor updated_scale,
                                 at::Tensor updated_scale_inv,
                                 const std::string& amax_compute_algo,
                                 transformer_engine::DType fp8_dtype,
                                 float margin) {
  nvte_delayed_scaling_recipe_amax_and_scale_update(
    makeTransformerEngineTensor(amax_history).data(),
    makeTransformerEngineTensor(scale).data(),
    makeTransformerEngineTensor(scale_inv).data(),
    makeTransformerEngineTensor(scale_inv_mask).data(),
    makeTransformerEngineTensor(updated_amax_history).data(),
    makeTransformerEngineTensor(updated_scale).data(),
    makeTransformerEngineTensor(updated_scale_inv).data(),
    amax_compute_algo.c_str(),
    static_cast<NVTEDType>(fp8_dtype),
    margin,
    at::cuda::getCurrentCUDAStream());
}

void fused_amax_and_scale_update_after_reduction(const at::Tensor &amax_reduction_buffer,
                                 std::vector<at::Tensor> amax_histories,
                                 std::vector<at::Tensor> scales,
                                 std::vector<at::Tensor> scale_invs,
                                 std::vector<at::Tensor> scale_inv_masks,
                                 const std::string &amax_compute_algo,
                                 transformer_engine::DType fp8_dtype,
                                 float margin) {
  using namespace transformer_engine;
  size_t num_tensors = amax_histories.size();
  std::vector<Tensor> t_amax_histories(num_tensors);
  std::vector<Tensor> t_scales(num_tensors);
  std::vector<Tensor> t_scale_invs(num_tensors);
  std::vector<Tensor> t_scale_inv_masks(num_tensors);
  std::vector<NVTETensor> te_amax_histories(num_tensors);
  std::vector<NVTETensor> te_scales(num_tensors);
  std::vector<NVTETensor> te_scale_invs(num_tensors);
  std::vector<NVTETensor> te_scale_inv_masks(num_tensors);
  for (size_t i = 0; i < num_tensors; i++) {
    t_amax_histories[i].data.dptr = amax_histories[i].data_ptr();
    auto amax_sizes = amax_histories[i].sizes().vec();
    std::vector<size_t> amax_shape{amax_sizes.begin(), amax_sizes.end()};
    t_amax_histories[i].data.shape = amax_shape;
    t_amax_histories[i].data.dtype = DType::kFloat32;

    t_scales[i].data.dptr = scales[i].data_ptr();
    auto scale_sizes = scales[i].sizes().vec();
    std::vector<size_t> scale_shape{scale_sizes.begin(), scale_sizes.end()};
    t_scales[i].data.shape = scale_shape;
    t_scales[i].data.dtype = DType::kFloat32;

    t_scale_invs[i].data.dptr = scale_invs[i].data_ptr();
    auto scale_inv_sizes = scale_invs[i].sizes().vec();
    std::vector<size_t> scale_inv_shape{scale_inv_sizes.begin(), scale_inv_sizes.end()};
    t_scale_invs[i].data.shape = scale_inv_shape;
    t_scale_invs[i].data.dtype = DType::kFloat32;

    t_scale_inv_masks[i].data.dptr = scale_inv_masks[i].data_ptr();
    auto mask_sizes = scale_inv_masks[i].sizes().vec();
    std::vector<size_t> mask_shape{mask_sizes.begin(), mask_sizes.end()};
    t_scale_inv_masks[i].data.shape = mask_shape;
    t_scale_inv_masks[i].data.dtype = DType::kByte;

    te_amax_histories[i] = reinterpret_cast<NVTETensor>(&t_amax_histories[i]);
    te_scales[i] = reinterpret_cast<NVTETensor>(&t_scales[i]);
    te_scale_invs[i] = reinterpret_cast<NVTETensor>(&t_scale_invs[i]);
    te_scale_inv_masks[i] = reinterpret_cast<NVTETensor>(&t_scale_inv_masks[i]);
  }
  nvte_delayed_scaling_recipe_amax_and_scale_update_after_reduction(
    makeTransformerEngineTensor(amax_reduction_buffer).data(),
    te_amax_histories,
    te_scales,
    te_scale_invs,
    te_scale_inv_masks,
    amax_compute_algo.c_str(),
    static_cast<NVTEDType>(fp8_dtype),
    margin,
    at::cuda::getCurrentCUDAStream());
}