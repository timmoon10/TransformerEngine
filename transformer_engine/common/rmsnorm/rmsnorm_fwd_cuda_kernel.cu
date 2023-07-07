/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "rmsnorm.h"
#include "rmsnorm_fwd_kernels.cuh"
#include "rmsnorm_kernel_traits.h"

using namespace transformer_engine::rmsnorm;

namespace {

// String with RTC kernel implementation
#include "string_code_rmsnorm_rtc_rmsnorm_fwd_cu.h"

}  // namespace

template <typename weight_t, typename input_t, typename output_t, typename compute_t,
          typename index_t, int HIDDEN_SIZE, int CTAS_PER_ROW, int WARPS_M, int WARPS_N,
          int BYTES_PER_LDG>
void launch_tuned_(LaunchParams<FwdParams> &launch_params, const bool configure_params) {  // NOLINT(*)
    using Kernel_traits = Kernel_traits<weight_t, input_t, output_t, compute_t, index_t,
                                        HIDDEN_SIZE, CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG>;
    constexpr const char *itype_name = TypeInfo<input_t>::name;
    constexpr const char *otype_name = TypeInfo<output_t>::name;
    constexpr const char *wtype_name = TypeInfo<weight_t>::name;
    static_assert(std::is_same<compute_t, float>::value);

    // Compile NVRTC kernel if needed
    NVTE_CHECK(rtc::is_enabled(), "optimized RMSNorm kernel requires NVRTC support");
    auto& rtc_manager = rtc::KernelManager::instance();
    const std::string kernel_label = concat_strings("rmsnorm_fwd"
                                                    ",itype=", itype_name,
                                                    ",otype=", otype_name,
                                                    ",wtype=", wtype_name,
                                                    ",load_size=", BYTES_PER_LDG,
                                                    ",warps_m=", WARPS_M,
                                                    ",warps_n=", WARPS_N,
                                                    ",width=", HIDDEN_SIZE,
                                                    ",nblocks_n=", CTAS_PER_ROW);
    if (!rtc_manager.is_compiled(kernel_label)) {
        std::string code = string_code_rmsnorm_rtc_rmsnorm_fwd_cu;
        code = regex_replace(code, "__ITYPE__", itype_name);
        code = regex_replace(code, "__OTYPE__", otype_name);
        code = regex_replace(code, "__WTYPE__", wtype_name);
        code = regex_replace(code, "__LOAD_SIZE__", BYTES_PER_LDG);
        code = regex_replace(code, "__WARPS_M__", WARPS_M);
        code = regex_replace(code, "__WARPS_N__", WARPS_N);
        code = regex_replace(code, "__WIDTH__", HIDDEN_SIZE);
        code = regex_replace(code, "__NBLOCKS_N__", CTAS_PER_ROW);
        rtc_manager.compile(kernel_label,
                            "rmsnorm_fwd_optimimzed_kernel",
                            code,
                            "transformer_engine/common/rmsnorm/rtc/rmsnorm_fwd.cu");
    }
    auto &kernel = rtc_manager.get_kernel(kernel_label);

    // Configure kernel params
    const size_t block_size = warps_per_block_m * warps_per_block_n * THREADS_PER_WARP;
    if (configure_params) {
        int ctas_per_sm;
        NVTE_CALL_CHECK_CUDA_DRIVER(cuOccupancyMaxActiveBlocksPerMultiprocessor,
                                    &ctas_per_sm,
                                    kernel,
                                    block_size,
                                    0);
        launch_params.params.ctas_per_row = CTAS_PER_ROW;
        launch_params.params.ctas_per_col =
            launch_params.multiprocessorCount * ctas_per_sm / launch_params.params.ctas_per_row;
        launch_params.barrier_size = 0;
        launch_params.workspace_bytes = 0;
        if (CTAS_PER_ROW > 1) {
            launch_params.barrier_size = 2 * launch_params.params.ctas_per_col;
            launch_params.workspace_bytes = launch_params.params.ctas_per_col *
                                            WARPS_M * CTAS_PER_ROW *
                                            sizeof(typename Kernel_traits::Stats::stats_t) * 2;
        }
        return;
    }

    // Launch cooperative kernel
    const size_t num_blocks = launch_params.params.ctas_per_col * CTAS_PER_ROW;
    kernel.launch_cooperative(cuda::current_device(),
                              num_blocks,
                              block_size,
                              0,
                              launch_params.stream,
                              static_cast<const input_t *>(input.data.dptr),
                              static_cast<const weight_t *>(gamma.data.dptr),
                              static_cast<compute_t>(epsilon),
                              static_cast<output_t *>(output.data.dptr),
                              static_cast<compute_t *>(rsigma),
                              static_cast<const compute_t *>(scale),
                              static_cast<compute_t *>(amax),
                              static_cast<void *>(workspace.data.dptr),
                              static_cast<int *>(barrier.data.dptr),
                              static_cast<size_t>(num_rows));
}

template <typename weight_t, typename input_t, typename output_t, typename compute_t,
          typename index_t, int HIDDEN_SIZE, int WARPS_M, int WARPS_N, int BYTES_PER_LDG>
void launch_general_(LaunchParams<FwdParams> &launch_params, const bool configure_params) {  // NOLINT(*)
    using Kernel_traits = Kernel_traits<weight_t, input_t, output_t, compute_t, index_t,
                                        HIDDEN_SIZE, 1, WARPS_M, WARPS_N, BYTES_PER_LDG>;
    auto kernel = &rmsnorm_fwd_general_kernel<Kernel_traits>;
    auto ceil_div = [](int x, int y) -> int { return (x + y - 1) / y; };

    // Configure kernel params
    const int rows = launch_params.params.rows;
    const int cols = launch_params.params.cols;
    int ctas_per_col = launch_params.params.ctas_per_col;
    int ctas_per_row = launch_params.params.ctas_per_row;
    if (configure_params) {
        int ctas_per_sm;
        cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &ctas_per_sm, kernel, Kernel_traits::THREADS_PER_CTA, 0);
        const int max_ctas = launch_params.multiprocessorCount * ctas_per_sm;
        ctas_per_row = ceil_div(cols, HIDDEN_SIZE);
        ctas_per_col = std::min(ceil_div(rows, WARPS_M), max_ctas / ctas_per_row);
        launch_params.params.ctas_per_row = ctas_per_row;
        launch_params.params.ctas_per_col = ctas_per_col;

        launch_params.barrier_size = 0;
        launch_params.workspace_bytes = 0;
        if (launch_params.params.ctas_per_row > 1) {
            launch_params.barrier_size = 2 * ctas_per_col;
            launch_params.workspace_bytes =
                (ctas_per_col * WARPS_M * ctas_per_row * sizeof(compute_t) * 2);
        }
        return;
    }

    // Launch kernel
    auto stream = launch_params.stream;
    dim3 grid(ctas_per_row * ctas_per_col);
    dim3 block(Kernel_traits::THREADS_PER_CTA);
    if (ctas_per_row == 1) {
        kernel<<<grid, block, 0, stream>>>(launch_params.params);
    } else {
        void *params_ = reinterpret_cast<void *>(&launch_params.params);
        cudaLaunchCooperativeKernel(reinterpret_cast<void *>(kernel), grid, block,
                                    reinterpret_cast<void **>(&params_), 0, stream);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#define REGISTER_FWD_TUNED_LAUNCHER(HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE,                       \
                              CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG)                       \
    void rmsnorm_fwd_tuned_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE(                  \
            LaunchParams<FwdParams> &launch_params,                                                \
            const bool configure_params) {                                                         \
        launch_tuned_<WTYPE, ITYPE, OTYPE, CTYPE, uint32_t, HIDDEN_SIZE, CTAS_PER_ROW,             \
        WARPS_M, WARPS_N, BYTES_PER_LDG>(                                                          \
            launch_params, configure_params);                                                      \
    }                                                                                              \
    static FwdTunedRegistrar<WTYPE, ITYPE, OTYPE, CTYPE, HIDDEN_SIZE>                              \
           reg_tuned_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE(                        \
        rmsnorm_fwd_tuned_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE)

#define REGISTER_FWD_GENERAL_LAUNCHER(HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE,                     \
                              WARPS_M, WARPS_N, BYTES_PER_LDG)                                     \
    void rmsnorm_fwd_general_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE(                \
            LaunchParams<FwdParams> &launch_params,                                                \
            const bool configure_params) {                                                         \
        launch_general_<WTYPE, ITYPE, OTYPE, CTYPE, uint32_t, HIDDEN_SIZE,                         \
        WARPS_M, WARPS_N, BYTES_PER_LDG>(                                                          \
            launch_params, configure_params);                                                      \
    }                                                                                              \
    static FwdGeneralRegistrar<WTYPE, ITYPE, OTYPE, CTYPE, HIDDEN_SIZE>                            \
           reg_general_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE(                      \
        rmsnorm_fwd_general_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE)

////////////////////////////////////////////////////////////////////////////////////////////////////

// Create rmsnorm tuned launch function and register. Macro signature:
//  HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE, CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG

REGISTER_FWD_TUNED_LAUNCHER(512, bf16, bf16, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(512, fp16, fp16, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(512, fp32, fp32, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(512, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(512, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(512, bf16, bf16, bf16, fp32, 1, 4, 1, 16);

REGISTER_FWD_TUNED_LAUNCHER(768, bf16, bf16, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(768, fp16, fp16, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(768, fp32, fp32, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(768, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(768, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(768, bf16, bf16, bf16, fp32, 1, 4, 1, 16);

REGISTER_FWD_TUNED_LAUNCHER(1024, bf16, bf16, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(1024, fp16, fp16, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(1024, fp32, fp32, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(1024, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(1024, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(1024, bf16, bf16, bf16, fp32, 1, 4, 1, 16);

REGISTER_FWD_TUNED_LAUNCHER(2048, bf16, bf16, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(2048, fp16, fp16, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(2048, fp32, fp32, fp8e4m3, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(2048, fp32, fp32, fp32, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(2048, fp16, fp16, fp16, fp32, 1, 4, 1, 16);
REGISTER_FWD_TUNED_LAUNCHER(2048, bf16, bf16, bf16, fp32, 1, 4, 1, 16);

REGISTER_FWD_TUNED_LAUNCHER(4096, bf16, bf16, fp8e4m3, fp32, 1, 1, 4, 16);
REGISTER_FWD_TUNED_LAUNCHER(4096, fp16, fp16, fp8e4m3, fp32, 1, 1, 4, 16);
REGISTER_FWD_TUNED_LAUNCHER(4096, fp32, fp32, fp8e4m3, fp32, 1, 1, 4, 16);
REGISTER_FWD_TUNED_LAUNCHER(4096, fp32, fp32, fp32, fp32, 1, 1, 4, 16);
REGISTER_FWD_TUNED_LAUNCHER(4096, fp16, fp16, fp16, fp32, 1, 1, 4, 16);
REGISTER_FWD_TUNED_LAUNCHER(4096, bf16, bf16, bf16, fp32, 1, 1, 4, 16);

// Create rmsnorm general launch function and register. Macro signature:
//  HIDDEN_SIZE, WTYPE, ITYPE, OTYPE, CTYPE, WARPS_M, WARPS_N, BYTES_PER_LDG

REGISTER_FWD_GENERAL_LAUNCHER(128, bf16, bf16, fp8e4m3, fp32, 4, 1, 8);
REGISTER_FWD_GENERAL_LAUNCHER(512, bf16, bf16, fp8e4m3, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(1024, bf16, bf16, fp8e4m3, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(2048, bf16, bf16, fp8e4m3, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(8192, bf16, bf16, fp8e4m3, fp32, 1, 4, 16);

REGISTER_FWD_GENERAL_LAUNCHER(128, fp16, fp16, fp8e4m3, fp32, 4, 1, 8);
REGISTER_FWD_GENERAL_LAUNCHER(512, fp16, fp16, fp8e4m3, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(1024, fp16, fp16, fp8e4m3, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(2048, fp16, fp16, fp8e4m3, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(8192, fp16, fp16, fp8e4m3, fp32, 1, 4, 16);

REGISTER_FWD_GENERAL_LAUNCHER(128, fp32, fp32, fp8e4m3, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(512, fp32, fp32, fp8e4m3, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(1024, fp32, fp32, fp8e4m3, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(2048, fp32, fp32, fp8e4m3, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(8192, fp32, fp32, fp8e4m3, fp32, 1, 4, 16);

REGISTER_FWD_GENERAL_LAUNCHER(128, fp32, fp32, fp32, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(128, fp16, fp16, fp16, fp32, 4, 1, 8);
REGISTER_FWD_GENERAL_LAUNCHER(128, fp32, fp32, fp16, fp32, 4, 1, 8);
REGISTER_FWD_GENERAL_LAUNCHER(128, bf16, bf16, bf16, fp32, 4, 1, 8);
REGISTER_FWD_GENERAL_LAUNCHER(128, fp32, fp32, bf16, fp32, 4, 1, 8);

REGISTER_FWD_GENERAL_LAUNCHER(512, fp32, fp32, fp32, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(512, fp16, fp16, fp16, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(512, fp32, fp32, fp16, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(512, bf16, bf16, bf16, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(512, fp32, fp32, bf16, fp32, 4, 1, 16);

REGISTER_FWD_GENERAL_LAUNCHER(1024, fp32, fp32, fp32, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(1024, fp16, fp16, fp16, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(1024, fp32, fp32, fp16, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(1024, bf16, bf16, bf16, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(1024, fp32, fp32, bf16, fp32, 4, 1, 16);

REGISTER_FWD_GENERAL_LAUNCHER(2048, fp32, fp32, fp32, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(2048, fp16, fp16, fp16, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(2048, fp32, fp32, fp16, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(2048, bf16, bf16, bf16, fp32, 4, 1, 16);
REGISTER_FWD_GENERAL_LAUNCHER(2048, fp32, fp32, bf16, fp32, 4, 1, 16);

REGISTER_FWD_GENERAL_LAUNCHER(8192, fp32, fp32, fp32, fp32, 1, 4, 16);
REGISTER_FWD_GENERAL_LAUNCHER(8192, fp16, fp16, fp16, fp32, 1, 4, 16);
REGISTER_FWD_GENERAL_LAUNCHER(8192, fp32, fp32, fp16, fp32, 1, 4, 16);
REGISTER_FWD_GENERAL_LAUNCHER(8192, bf16, bf16, bf16, fp32, 1, 4, 16);
REGISTER_FWD_GENERAL_LAUNCHER(8192, fp32, fp32, bf16, fp32, 1, 4, 16);
