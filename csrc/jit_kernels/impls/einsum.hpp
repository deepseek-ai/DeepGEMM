#pragma once

#include <torch/python.h>
#include <cstdint>
#include <hggc_fp8.h>
#include "cute/tensor.hpp"
#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../../utils/utils.hpp"
#include "../../utils/layout_type_name.hpp"
#include "../../utils/layout.hpp"
#include "cute/arch/mma.hpp"
#include "../heuristics/common_fp8.hpp"
#include "../heuristics/common_int8.hpp"
#include "cutlass/gemm/gemm.h"
#include "util/include/cutlass/util/packed_stride.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include <deep_gemm/common/utils_rtc.cuh>
#include <deep_gemm/common/profiling_interface.cuh>
#include "fp8_gemm.hpp"
#include "int8_gemm.hpp"
#include "fused_permute.hpp"

using namespace deep_gemm_fp8_common;
namespace deep_gemm {

using ConfigTuple = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;

static void fp8_bmm_impl(const torch::Tensor& a, const torch::Tensor& sfa,
             const torch::Tensor& b, const torch::Tensor& sfb,
             const torch::Tensor& d,
             const std::optional<torch::Tensor>& c,
             std::optional<ConfigTuple> configs = std::nullopt,
             bool transposed_out = true) {
    const auto& [groups, m, k] = get_shape<3>(a);
    const auto& [_, n, __] = get_shape<3>(b);

    auto sfa_aligned = get_mn_major_tma_aligned_tensor(sfa);

    int num_sms = get_num_sms();
    ConfigTuple selected_config;
    if (configs.has_value()) {
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        selected_config = std::make_tuple(ns, bm, bn, bk, wm, wn, nst,
            deep_gemm_fp8_common::get_smem_config(nst, k, bm, bn, bk));
    } else {
        selected_config = deep_gemm_fp8_common::get_best_configs(
            m, n, k, groups, num_sms, GemmType::BatchGemm);
    }

    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config] = selected_config;
    auto SMSIZE = std::get<0>(smem_config);
    uint32_t kNumGroups = groups;

    using StrideA = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideB = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using ScaleGranularityShape = cute::Shape<cute::_1, cute::_128, cute::_128>;
    using ScaleConfig =
        decltype(cutlass::detail::ppu_trivial_blockwise_scale_config<ScaleGranularityShape, false, true>(
            ScaleGranularityShape{}));
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, groups));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, groups));
    // Transposed `[M, B, N]` output: a row spans every batch, and the scheduler advances
    // `ptr_D` by one row segment per batch. Plain `[B, M, N]` output: rows are contiguous
    // within a batch, and the scheduler advances `ptr_D` by a whole `M x N` matrix.
    auto stride_D = transposed_out
                        ? StrideA{(int64_t)groups * n, cute::Int<1>{}, (int64_t)n}
                        : StrideA{(int64_t)n, cute::Int<1>{}, (int64_t)m * n};

    LayoutSFA layout_SFA;
    LayoutSFB layout_SFB;
    layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, groups));
    layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, groups));

    cutlass::float_e4m3_t* converted_input_a =
        reinterpret_cast<cutlass::float_e4m3_t*>(a.data_ptr<at::Float8_e4m3fn>());
    cutlass::float_e4m3_t* converted_input_b =
        reinterpret_cast<cutlass::float_e4m3_t*>(b.data_ptr<at::Float8_e4m3fn>());
    cutlass::bfloat16_t* converted_output =
        reinterpret_cast<cutlass::bfloat16_t*>(d.data_ptr<at::BFloat16>());
    float* scales_a_ptr = sfa_aligned.data_ptr<float>();
    float* scales_b_ptr = sfb.data_ptr<float>();

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.cu_count = num_sms_new;
    dim3 const block = (block_m / warp_m) * (block_n / warp_n) * 32;
    dim3 grid = get_grid_shape(hw_info.cu_count);

    static constexpr GemmType kGemmType = GemmType::BatchGemm;

    const auto gemm_args = FP8GemmRuntime::GemmArguments{
        .mode = cutlass::gemm::GemmUniversalMode::kGemm,
        .problem_shape = {m, n, k, groups},
        .mainloopargs = {converted_input_a, stride_A, converted_input_b, stride_B, 4,
                        scales_a_ptr, layout_SFA, scales_b_ptr, layout_SFB},
        .epilogueargs =
            {
                {1, 0},
                nullptr,
                stride_D,
                converted_output,
                stride_D,
            },
        .hw_info = hw_info,
        .scheduler = {(uint32_t)m, nullptr},
        .signal = nullptr};

    FP8GemmRuntime::GemmKernelParams params = FP8GemmRuntime::to_underlying_arguments_rtc(gemm_args, nullptr);

    auto args = FP8GemmRuntime::Args{.launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups, num_stages,
                                                     "BatchGemm", "Default", "batch_fp8_deep_gemm", false,
                                                     transposed_out},
                                     .launch_args = {grid, block, SMSIZE},
                                     .kernel_params = params};

    const auto& code = FP8GemmRuntime::generate(args);
    const auto& runtime = compiler->build("batch_fp8_deep_gemm", code);
    const auto& kernel = runtime->kernel;
    int blocks_per_cu = 0;
    HGresult result = hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, block.x, SMSIZE);
    args.launch_args.grid_dim.x *= blocks_per_cu;
    if (result != HGGC_SUCCESS || blocks_per_cu == 0) {
        throw std::runtime_error("Failed to get max active blocks per multiprocessor");
    }

    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_params(kGemmType, false, std::string("fp8"), kNumGroups, m, n, k, 0, nullptr,
                                  current_stream());
    }
    ProfilingInterface::Instance().instrument(true, dg_prof_params);

    FP8GemmRuntime::launch(runtime, args);

    ProfilingInterface::Instance().instrument(false, dg_prof_params);
    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        int numRegs = 0, localSize = 0;
        hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
        hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

        printf("[BatchGemm_FP8:]\n");
        printf("group:%d, problem:[%d, %d, %d]\n", kNumGroups, m, n, k);
        printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu,
                args.launch_args.grid_dim.x);
        printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], num_stages:%d\n", block_m, block_n,
                block_k, warp_m, warp_n, block_k, num_stages);
        printf("SMSIZE:%d, vreg:%d, stack:%d\n", int(SMSIZE), int(numRegs), int(localSize));
    }
}

static void int8_bmm_impl(const torch::Tensor& a, const torch::Tensor& sfa,
              const torch::Tensor& b, const torch::Tensor& sfb,
              const torch::Tensor& d,
              const std::optional<torch::Tensor>& c,
              std::optional<ConfigTuple> configs = std::nullopt,
              bool transposed_out = true) {
    const auto& [groups, m, k] = get_shape<3>(a);
    const auto& [_, n, __] = get_shape<3>(b);

    int num_sms = get_num_sms();

    ConfigTuple selected_config;
    if (configs.has_value()) {
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        selected_config = std::make_tuple(ns, bm, bn, bk, wm, wn, nst,
            deep_gemm_int8::get_smem_config(nst, k, bm, bn, bk, 1));
    } else {
        if (is_ppu1v5_device()) {
            selected_config = deep_gemm_int8::get_best_configs_dense_ppu1v5(m, n, k, groups, num_sms);
        } else {
            selected_config = deep_gemm_int8::get_best_configs(
                m, n, k, groups, num_sms, GemmType::BatchGemm);
        }
    }

    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config] = selected_config;
    auto SMSIZE = std::get<0>(smem_config);
    uint32_t kNumGroups = groups;

    using StrideA = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideB = cute::Stride<int64_t, cute::Int<1>, int64_t>;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, groups));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, groups));
    // Transposed `[M, B, N]` output: a row spans every batch, and the scheduler advances
    // `ptr_D` by one row segment per batch. Plain `[B, M, N]` output: rows are contiguous
    // within a batch, and the scheduler advances `ptr_D` by a whole `M x N` matrix.
    auto stride_D = transposed_out
                        ? StrideA{(int64_t)groups * n, cute::Int<1>{}, (int64_t)n}
                        : StrideA{(int64_t)n, cute::Int<1>{}, (int64_t)m * n};

    torch::Dtype dtype = a.dtype().toScalarType();
    const void* converted_input_a = nullptr;
    const void* converted_input_b = nullptr;
    cutlass::bfloat16_t* converted_output =
        reinterpret_cast<cutlass::bfloat16_t*>(d.data_ptr<at::BFloat16>());
    std::string type_info;
    std::string kernel_name;
    std::string profile_type;

    if (dtype == torch::kInt8) {
        converted_input_a = a.data_ptr<int8_t>();
        converted_input_b = b.data_ptr<int8_t>();
        type_info = "int8_t";
        kernel_name = "batch_int8_deep_gemm";
        profile_type = "int8";
    } else {
        converted_input_a = reinterpret_cast<const void*>(a.data_ptr<at::Float8_e4m3fn>());
        converted_input_b = reinterpret_cast<const void*>(b.data_ptr<at::Float8_e4m3fn>());
        type_info = "cutlass::float_e4m3_t";
        kernel_name = "batch_fp8_deep_gemm_channelwise";
        profile_type = "fp8";
    }

    float* scales_a_ptr = sfa.data_ptr<float>();
    float* scales_b_ptr = sfb.data_ptr<float>();

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.cu_count = num_sms_new;
    dim3 const block = (block_m / warp_m) * (block_n / warp_n) * 32;
    dim3 grid = get_grid_shape(hw_info.cu_count);

    static constexpr GemmType kGemmType = GemmType::BatchGemm;

    const auto gemm_args = INT8GemmCutlass3Runtime::GemmArguments{
        .mode = cutlass::gemm::GemmUniversalMode::kGemm,
        .problem_shape = {m, n, k, groups},
        .mainloopargs = {converted_input_a, stride_A, converted_input_b, stride_B,
                         scales_a_ptr, scales_b_ptr},
        .epilogueargs =
            {
                {1, 0},
                nullptr,
                stride_D,
                converted_output,
                stride_D,
            },
        .hw_info = hw_info,
        .scheduler = {(uint32_t)m, nullptr},
        .signal = nullptr};

    INT8GemmCutlass3Runtime::GemmKernelParams params =
        INT8GemmCutlass3Runtime::to_underlying_arguments_rtc(gemm_args, nullptr);

    auto args = INT8GemmCutlass3Runtime::Args{
        .launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups, num_stages,
                        "BatchGemm", "Default", kernel_name, false, transposed_out},
        .launch_args = {grid, block, SMSIZE},
        .kernel_params = params,
        .type_info = type_info};

    const auto& code = INT8GemmCutlass3Runtime::generate(args);
    const auto& runtime = compiler->build(kernel_name, code);
    const auto& kernel = runtime->kernel;
    int blocks_per_cu = 0;
    HGresult result = hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, block.x, SMSIZE);
    args.launch_args.grid_dim.x *= blocks_per_cu;

    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_params(kGemmType, false, profile_type, kNumGroups, m, n, k, 0, nullptr,
                                  current_stream());
    }
    ProfilingInterface::Instance().instrument(true, dg_prof_params);

    INT8GemmCutlass3Runtime::launch(runtime, args);

    ProfilingInterface::Instance().instrument(false, dg_prof_params);

    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        int numRegs = 0, localSize = 0;
        hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
        hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

        printf("[BatchGemm_INT8:]\n");
        printf("group:%d, problem:[%d, %d, %d]\n", kNumGroups, m, n, k);
        printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu,
               args.launch_args.grid_dim.x);
        printf("ThreadblockShape[%d, %d, %d], WarpShape[%d, %d, %d], num_stages:%d\n", block_m, block_n,
               block_k, warp_m, warp_n, block_k, num_stages);
        printf("SMSIZE:%d, vreg:%d, stack:%d\n", int(SMSIZE), int(numRegs), int(localSize));
    }

}

} // namespace deep_gemm
