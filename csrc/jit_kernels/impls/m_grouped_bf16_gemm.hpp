#pragma once

#include <torch/python.h>
#include <cstdint>
#include "cute/tensor.hpp"
#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../../utils/utils.hpp"
#include "../../utils/layout_type_name.hpp"
#include "cute/arch/mma.hpp"
#include "../heuristics/common_bf16.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/gemm.h"
#include "util/include/cutlass/util/packed_stride.hpp"
#include "fp8_gemm.hpp"
#include "bf16_gemm.hpp"
#include "m_grouped_int8_gemm.hpp"

using namespace deep_gemm_bf16_common;
namespace deep_gemm {

using Config = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;

static void m_grouped_gemm_bf16_bf16_bf16_nt_contiguous_impl(const torch::Tensor& lhs, const torch::Tensor& rhs,
                                                             const torch::Tensor& out, const torch::Tensor& m_indices,
                                                             const int& m, const int& n, const int& k,
                                                             const int& num_groups, std::optional<ConfigTuple> configs) {
    int num_sms = get_num_sms();
    Config cfg;
    if (configs.has_value()) {
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        cfg = std::make_tuple(ns, bm, bn, bk, wm, wn, nst,
            deep_gemm_bf16_common::get_smem_config(nst, k, bm, bn, bk, 2));
    } else {
        cfg = deep_gemm_bf16_common::get_best_configs(
            m, n, k, 1, num_sms, GemmType::GroupedContiguous);
    }

    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config] = cfg;
    int expected_m = ceil_div(m, num_groups);
    auto SMSIZE = std::get<0>(smem_config);

    int kNumGroups = num_groups;
    static constexpr GemmType kGemmType = GemmType::GroupedContiguous;
    using StrideA = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideB = cute::Stride<int64_t, cute::Int<1>, int64_t>;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, n, 1));

    int32_t* grouped_layout = reinterpret_cast<int32_t*>(m_indices.data_ptr<int32_t>());
    cutlass::bfloat16_t* input_b = reinterpret_cast<cutlass::bfloat16_t*>(rhs.data_ptr<at::BFloat16>());
    cutlass::bfloat16_t* input_a = reinterpret_cast<cutlass::bfloat16_t*>(lhs.data_ptr<at::BFloat16>());
    cutlass::bfloat16_t* output = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>());

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.cu_count = num_sms_new;
    dim3 const block = (block_m / warp_m) * (block_n / warp_n) * 32;
    dim3 grid = get_grid_shape(hw_info.cu_count);
    bool enable_sbo_overlap = false;
    const auto gemm_args = BF16GemmCutlass3Runtime::GemmArguments{
        .mode = cutlass::gemm::GemmUniversalMode::kGemm,
        .problem_shape = {m, n, k, 1},
        .mainloopargs = {input_a, stride_A, input_b, stride_B},
        .epilogueargs =
            {
                {1, 0},
                output,
                stride_D,
                output,
                stride_D,
            },
        .hw_info = hw_info,
        .scheduler = {(uint32_t)m, grouped_layout},
        .signal = nullptr};

    BF16GemmCutlass3Runtime::GemmKernelParams params =
        BF16GemmCutlass3Runtime::to_underlying_arguments_rtc(gemm_args, nullptr);
    auto args = BF16GemmCutlass3Runtime::Args{
        .launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups, num_stages, "GroupedContiguous",
                        "Default", "bf16_grouped_deep_gemm_contiguous", enable_sbo_overlap},
        .launch_args = {grid, block, SMSIZE},
        .kernel_params = params};
    const auto& code = BF16GemmCutlass3Runtime::generate(args);
    const auto& runtime = compiler->build("bf16_grouped_deep_gemm_contiguous", code);
    const auto& kernel = runtime->kernel;
    int blocks_per_cu = 0;
    HGresult result = hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, block.x, SMSIZE);
    args.launch_args.grid_dim.x *= blocks_per_cu;

    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_params(kGemmType, false, std::string("bf16"), kNumGroups, m, n, k, expected_m,
                                  grouped_layout, current_stream());
    }
    ProfilingInterface::Instance().instrument(true, dg_prof_params);

    BF16GemmCutlass3Runtime::launch(runtime, args);

    ProfilingInterface::Instance().instrument(false, dg_prof_params);

    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        int numRegs = 0, localSize = 0;
        hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
        hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

        printf("[GroupedContiguous:]\n");
        printf("group:%d, problem:[%d, %d, %d]\n", num_groups, m, n, k);
        printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu,
               args.launch_args.grid_dim.x);
        printf("ThreadblockShape[%d, %d, %d], expected_m:%d, WarpShape[%d, %d, %d], num_stages:%d\n", block_m,
               block_n, block_k, expected_m, warp_m, warp_n, block_k, num_stages);
        printf("SMSIZE:%d, vreg:%d, stack:%d\n",int(SMSIZE), int(numRegs), int(localSize));
    }

}

static std::pair<int, int> m_grouped_gemm_bf16_bf16_bf16_nt_masked_impl(const torch::Tensor& lhs, const torch::Tensor& rhs,
                                                         const torch::Tensor& out, const torch::Tensor& masked_m,
                                                         const int& m, const int& n, const int& k,
                                                         const int& num_groups, const int& expected_m,
                                                         std::optional<ConfigTuple> configs, int max_block_n,
                                                         bool enable_sbo_overlap, const torch::Tensor& signal) {
    int num_sms = get_num_sms();

    Config cfg;
    if (configs.has_value()) {
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        cfg = std::make_tuple(ns, bm, bn, bk, wm, wn, nst,
            deep_gemm_bf16_common::get_smem_config(nst, k, bm, bn, bk, 2));
    } else {
        cfg = deep_gemm_bf16_common::get_best_configs(
            expected_m, n, k, num_groups, num_sms, GemmType::GroupedMasked, max_block_n);
    }

    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config] = cfg;

    // int expected_m = ceil_div(m, num_groups);
    auto SMSIZE = std::get<0>(smem_config);

    int kNumGroups = num_groups;
    static constexpr GemmType kGemmType = GemmType::GroupedMasked;
    using StrideA = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideB = cute::Stride<int64_t, cute::Int<1>, int64_t>;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, n, 1));

    int32_t* grouped_layout = reinterpret_cast<int32_t*>(masked_m.data_ptr<int32_t>());
    int32_t* signal_ptr = reinterpret_cast<int32_t*>(signal.data_ptr<int32_t>());

    cutlass::bfloat16_t* input_b = reinterpret_cast<cutlass::bfloat16_t*>(rhs.data_ptr<at::BFloat16>());
    cutlass::bfloat16_t* input_a = reinterpret_cast<cutlass::bfloat16_t*>(lhs.data_ptr<at::BFloat16>());
    cutlass::bfloat16_t* output = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>());

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.cu_count = num_sms_new;
    dim3 const block = (block_m / warp_m) * (block_n / warp_n) * 32;
    dim3 grid = get_grid_shape(hw_info.cu_count);
    const auto gemm_args = BF16GemmCutlass3Runtime::GemmArguments{
        .mode = cutlass::gemm::GemmUniversalMode::kGemm,
        .problem_shape = {m, n, k, 1},
        .mainloopargs = {input_a, stride_A, input_b, stride_B},
        .epilogueargs =
            {
                {1, 0},
                output,
                stride_D,
                output,
                stride_D,
            },
        .hw_info = hw_info,
        .scheduler = {(uint32_t)m, grouped_layout},
        .signal = signal_ptr};

    BF16GemmCutlass3Runtime::GemmKernelParams params =
        BF16GemmCutlass3Runtime::to_underlying_arguments_rtc(gemm_args, nullptr);

    auto args = BF16GemmCutlass3Runtime::Args{.launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups,
                                                              num_stages, "GroupedMasked", "Default",
                                                              "bf16_grouped_deep_gemm_masked", enable_sbo_overlap},
                                              .launch_args = {grid, block, SMSIZE},
                                              .kernel_params = params};
    const auto& code = BF16GemmCutlass3Runtime::generate(args);
    const auto& runtime = compiler->build("bf16_grouped_deep_gemm_masked", code);
    const auto& kernel = runtime->kernel;
    int blocks_per_cu = 0;
    HGresult result = hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, block.x, SMSIZE);
    args.launch_args.grid_dim.x *= blocks_per_cu;

    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_params(kGemmType, false, std::string("bf16"), kNumGroups, m, n, k, expected_m,
                                  grouped_layout, current_stream());
    }
    ProfilingInterface::Instance().instrument(true, dg_prof_params);

    BF16GemmCutlass3Runtime::launch(runtime, args);

    ProfilingInterface::Instance().instrument(false, dg_prof_params);

    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        int numRegs = 0, localSize = 0;
        hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
        hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

        printf("[GroupedMasked:]\n");
        printf("group:%d, problem:[%d, %d, %d]\n", num_groups, m, n, k);
        printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu,
               args.launch_args.grid_dim.x);
        printf("ThreadblockShape[%d, %d, %d], expected_m:%d, WarpShape[%d, %d, %d], num_stages:%d\n", block_m,
               block_n, block_k, expected_m, warp_m, warp_n, block_k, num_stages);
        printf("SMSIZE:%d, vreg:%d, stack:%d\n",int(SMSIZE), int(numRegs), int(localSize));
    }

    return std::make_pair(block_m, ceil_div(n, block_n));
}

static void m_grouped_gemm_bf16_bf16_bf16_nt_nopad_impl(const torch::Tensor& lhs, const torch::Tensor& rhs,
                                                        const torch::Tensor& out, const torch::Tensor& m_indices,
                                                        const int& m, const int& n, const int& k, const int& num_groups,
                                                        std::optional<const torch::Tensor> m_rows,
                                                        std::optional<ConfigTuple> configs) {
    int BlockSize, ThreadPerN, NUM_UNROLL, SWZL_SIZE_M, NPerThread;
    bool SMALL_K;
    int num_sms = get_num_sms();
    int expected_m = ceil_div(m, num_groups);
    static constexpr GemmType kGemmType = GemmType::GroupedNoPad;
    bool use_gemv = false;
    if ((k % 16 == 0 && ((m <= 2 * num_groups * 0.75 && k <= 32 * 8) || (m < 0.65 * num_groups && k > 256)) &&
         !is_ppu1v5_device()) ||
        (m < 0.8 * num_groups && is_ppu1v5_device())) {
        std::tie(BlockSize, ThreadPerN, NUM_UNROLL, SWZL_SIZE_M, NPerThread, SMALL_K) =
            deep_gemm_bf16_common::get_gemv_best_configs(m, n, k, num_groups, num_sms, torch::kBFloat16);
        if (ThreadPerN != -1) {
            // GemV kernel requires N % NPerBlock == 0: both the host-side grid_y
            // and the per-thread boundary check assume full N-tiles.  Fall back to
            // the general GEMM path when N is not a multiple of NPerBlock.
            int NPerBlock = NPerThread * BlockSize / ThreadPerN;
            if (n % NPerBlock == 0) {
                use_gemv = true;
            }
        }
    }
    int* layout_info = reinterpret_cast<int32_t*>(m_indices.data_ptr<int32_t>());
    if (use_gemv) {
        DgProfParam dg_prof_params;
        if (ProfilingInterface::Instance().get_op_info()) {
            dg_prof_params.set_params(GemmType::GroupedNoPad, true, std::string("bf16"), num_groups, m, n, k, expected_m,
                                      layout_info, current_stream());
        }
        cutlass::bfloat16_t* converted_output = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>());
        auto gemmv_args = GemvRuntime::GemvtArgs{
            .N = n,
            .K = k,
            .a_ptr = lhs.data_ptr<at::BFloat16>(),
            .b_ptr = rhs.data_ptr<at::BFloat16>(),
            .c_ptr = converted_output,
            .topk_weights_ptr = nullptr,
            .alphaCol = nullptr,
            .alphaRow = nullptr,
            .num_tokens = m,
            .num_experts = num_groups,
            .expert_ids_ptr = layout_info,
            .stride_am = k,
            .stride_ak = 1,
            .stride_be = n * k,
            .stride_bk = 1,
            .stride_bn = k,
            .stride_cm = n,
            .stride_cn = 1,
        };
        using load_atype = int4;
        using src_type = __ppu_bfloat16;
        if (SMALL_K == false) {
            size_t grid_x = m;
            int NPerBlock = NPerThread * BlockSize / ThreadPerN;
            size_t grid_y = ceil_div(n, NPerBlock);
            gemmv_args.total_blocks = grid_x * grid_y;
            dim3 grid = grid_x * grid_y;
            dim3 block = BlockSize;
            // check GEMM_K alignment
            if (gemmv_args.K % (NUM_UNROLL * ThreadPerN * sizeof(load_atype) / sizeof(src_type)) != 0) {
                printf("K alignment mismatch, K = %d, NUM_UNROLL = %d, ThreadPerN = %d, sizeof(load_atype) = %d, "
                       "sizeof(src_type) = %d",
                       gemmv_args.K, NUM_UNROLL, ThreadPerN, sizeof(load_atype), sizeof(src_type));
                return;
            }
            auto args = GemvRuntime::Args{.launch_info = {BlockSize, ThreadPerN, NPerThread, NUM_UNROLL, SWZL_SIZE_M,
                                                          SMALL_K, "__ppu_bfloat16", "float", "batched_gemvt_kernel"},
                                          .launch_args = {grid, block, 0},
                                          .kernel_params = gemmv_args};
            const auto& code = GemvRuntime::generate(args);
            const auto& runtime = compiler->build("batched_gemvt_kernel", code);
            const auto& kernel = runtime->kernel;
            ProfilingInterface::Instance().instrument(true, dg_prof_params);
            GemvRuntime::launch(runtime, args);
            ProfilingInterface::Instance().instrument(false, dg_prof_params);

            char* pEnv_params = std::getenv("show_log");
            if (pEnv_params && isdigit(*pEnv_params)) {
                int numRegs = 0, localSize = 0;
                hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
                hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

                printf("[GemV-BF16:]\n");
                printf("group:%d, problem:[%d, %d, %d]\n", num_groups, m, n, k);
                printf("BlockSize:%d, NPerThread:%d, ThreadPerN:%d, NPerBlock:%d, NUM_UNROLL:%d, SWZL_SIZE_M:%d\n",
                       BlockSize, NPerThread, ThreadPerN, NPerBlock, NUM_UNROLL, SWZL_SIZE_M);
                printf("threadblock_count:%d, vreg:%d, stack:%d\n", gemmv_args.total_blocks, int(numRegs),
                       int(localSize));
            }
            return;
        } else {
            size_t grid_x = gemmv_args.num_tokens;
            int NPerBlock = NPerThread * BlockSize / ThreadPerN;
            size_t grid_y = ceil_div(n, NPerBlock);
            gemmv_args.total_blocks = grid_x * grid_y;
            int MAX_K = NUM_UNROLL * ThreadPerN * sizeof(load_atype) / sizeof(src_type);
            int MIN_ALIGNMENT = 16 / sizeof(src_type); // for int4 copy

            if (gemmv_args.K > MAX_K) {
                printf("unsupported K, K = %d, MAX_K = %d, NUM_UNROLL = %d, ThreadPerN = %d, sizeof(load_atype) = %d, "
                       "sizeof(src_type) = %d",
                       gemmv_args.K, MAX_K, NUM_UNROLL, ThreadPerN, sizeof(load_atype), sizeof(src_type));
                return;
            }

            if (gemmv_args.stride_am % MIN_ALIGNMENT != 0 || gemmv_args.stride_bn % MIN_ALIGNMENT != 0) {
                printf("unsupported stride, stride_am = %d, stride_bn = %d\n", gemmv_args.stride_am,
                       gemmv_args.stride_bn);
                return;
            }
            dim3 grid = grid_x * grid_y;
            dim3 block = BlockSize;
            auto args =
                GemvRuntime::Args{.launch_info = {BlockSize, ThreadPerN, NPerThread, NUM_UNROLL, SWZL_SIZE_M, SMALL_K,
                                                  "__ppu_bfloat16", "float", "batched_gemvt_small_k_kernel"},
                                  .launch_args = {grid_x * grid_y, block, 0},
                                  .kernel_params = gemmv_args};
            const auto& code = GemvRuntime::generate(args);
            const auto& runtime = compiler->build("batched_gemvt_small_k_kernel", code);
            const auto& kernel = runtime->kernel;
            ProfilingInterface::Instance().instrument(true, dg_prof_params);
            GemvRuntime::launch(runtime, args);
            ProfilingInterface::Instance().instrument(false, dg_prof_params);

            char* pEnv_params = std::getenv("show_log");
            if (pEnv_params && isdigit(*pEnv_params)) {
                int numRegs = 0, localSize = 0;
                hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
                hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

                printf("[GemV-BF16:]\n");
                printf("group:%d, problem:[%d, %d, %d]\n", num_groups, m, n, k);
                printf("BlockSize:%d, NPerThread:%d, ThreadPerN:%d, NPerBlock:%d, NUM_UNROLL:%d, SWZL_SIZE_M:%d\n",
                       BlockSize, NPerThread, ThreadPerN, NPerBlock, NUM_UNROLL, SWZL_SIZE_M);
                printf("threadblock_count:%d, vreg:%d, stack:%d\n", gemmv_args.total_blocks, int(numRegs),
                       int(localSize));
            }
            return;
        }
    }
    Config cfg;
    if (configs.has_value()) {
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        cfg = std::make_tuple(ns, bm, bn, bk, wm, wn, nst,
            deep_gemm_bf16_common::get_smem_config(nst, k, bm, bn, bk, 2));
    } else {
        cfg = deep_gemm_bf16_common::get_best_configs(
            expected_m, n, k, num_groups, num_sms, GemmType::GroupedNoPad);
    }

    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config] = cfg;

    // int expected_m = ceil_div(m, num_groups);
    auto SMSIZE = std::get<0>(smem_config);

    int kNumGroups = num_groups;

    using StrideA = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideB = cute::Stride<int64_t, cute::Int<1>, int64_t>;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, n, 1));

    cutlass::bfloat16_t* input_b = reinterpret_cast<cutlass::bfloat16_t*>(rhs.data_ptr<at::BFloat16>());
    cutlass::bfloat16_t* input_a = reinterpret_cast<cutlass::bfloat16_t*>(lhs.data_ptr<at::BFloat16>());
    cutlass::bfloat16_t* output = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>());

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.cu_count = num_sms_new;
    dim3 const block = (block_m / warp_m) * (block_n / warp_n) * 32;
    dim3 grid = get_grid_shape(hw_info.cu_count);
    bool enable_sbo_overlap = false;
    at::Tensor m_rows_tensor;

    if (!m_rows.has_value() || !m_rows->defined()) {
        // counts = torch.bincount(m_indices)
        at::Tensor counts = at::bincount(m_indices);

        int64_t min_n = std::min<int64_t>(counts.size(0), num_groups);

        at::Tensor experts_for_rows =
            at::zeros({num_groups}, at::TensorOptions().dtype(at::kInt).device(m_indices.device()));

        if (min_n > 0) {
            // experts_for_rows[:min_n] = counts[:min_n]
            experts_for_rows.narrow(0, 0, min_n).copy_(counts.narrow(0, 0, min_n).to(at::kInt));
        }

        m_rows_tensor = experts_for_rows;
    } else {
        m_rows_tensor = *m_rows;
    }
    int64_t block_m_info_size = (num_groups + ceil_div(m + 1 - num_groups, block_m)) * 4;

    at::Tensor block_m_info =
        at::empty({block_m_info_size}, at::TensorOptions().dtype(at::kInt).device(m_rows_tensor.device()));
    // int* layout_info = grouped_layout;
    layout_info = reinterpret_cast<int32_t*>(m_rows_tensor.data_ptr<int32_t>());
    bool kIsNoPadPreprocessLayout = kNumGroups >= 128;
    if (kIsNoPadPreprocessLayout) {
        uint32_t block_size = std::max(32, next_power_of_two(kNumGroups));
        auto compute_block_info_args = ComputeBlockInfoKernelRuntime::Args{
            // reinterpret_cast<const uint32_t*>(m_rows_tensor.data_ptr<int32_t>())
            .launch_attr_args = {reinterpret_cast<const uint32_t*>(m_rows_tensor.data_ptr<int32_t>()), kNumGroups,
                                 reinterpret_cast<uint32_t*>(
                                     block_m_info.data_ptr<
                                         int32_t>())}, // reinterpret_cast<uint32_t*>(block_m_info.data_ptr<int32_t>())
            .launch_args = {1, block_size, 0},
        };
        const auto& code_blockinfo = ComputeBlockInfoKernelRuntime::generate(block_m);
        const auto& runtime_blockinfo = compiler->build("computeBlockInfoKernel", code_blockinfo);
        ComputeBlockInfoKernelRuntime::launch(runtime_blockinfo, compute_block_info_args);
        layout_info = reinterpret_cast<int32_t*>(block_m_info.data_ptr<int32_t>());
    }
    const auto gemm_args = BF16GemmCutlass3Runtime::GemmArguments{
        .mode = cutlass::gemm::GemmUniversalMode::kGemm,
        .problem_shape = {m, n, k, 1},
        .mainloopargs = {input_a, stride_A, input_b, stride_B},
        .epilogueargs =
            {
                {1, 0},
                output,
                stride_D,
                output,
                stride_D,
            },
        .hw_info = hw_info,
        .scheduler = {(uint32_t)m, layout_info},
        .signal = nullptr};

    BF16GemmCutlass3Runtime::GemmKernelParams params =
        BF16GemmCutlass3Runtime::to_underlying_arguments_rtc(gemm_args, nullptr);
    auto args = BF16GemmCutlass3Runtime::Args{.launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups,
                                                              num_stages, "GroupedNoPad", "Default",
                                                              "bf16_grouped_deep_gemm_NoPad", false},
                                              .launch_args = {grid, block, SMSIZE},
                                              .kernel_params = params};
    const auto& code = BF16GemmCutlass3Runtime::generate(args);
    const auto& runtime = compiler->build("bf16_grouped_deep_gemm_NoPad", code);
    const auto& kernel = runtime->kernel;
    int blocks_per_cu = 0;
    HGresult result = hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, block.x, SMSIZE);
    args.launch_args.grid_dim.x *= blocks_per_cu;

    DgProfParam dg_prof_params;
    if (ProfilingInterface::Instance().get_op_info()) {
        dg_prof_params.set_params(kGemmType, false, std::string("bf16"), kNumGroups, m, n, k, expected_m,
                                  m_rows_tensor.data_ptr<int32_t>(), current_stream());
    }
    ProfilingInterface::Instance().instrument(true, dg_prof_params);

    BF16GemmCutlass3Runtime::launch(runtime, args);

    ProfilingInterface::Instance().instrument(false, dg_prof_params);

    char* pEnv_params = std::getenv("show_log");
    if (pEnv_params && isdigit(*pEnv_params)) {
        int numRegs = 0, localSize = 0;
        hgFuncGetAttribute(&numRegs, HG_FUNC_ATTRIBUTE_NUM_REGS, kernel);
        hgFuncGetAttribute(&localSize, HG_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kernel);

        printf("[GroupedNoPad:]\n");
        printf("group:%d, problem:[%d, %d, %d]\n", num_groups, m, n, k);
        printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu,
               args.launch_args.grid_dim.x);
        printf("ThreadblockShape[%d, %d, %d], expected_m:%d, WarpShape[%d, %d, %d], num_stages:%d\n", block_m,
               block_n, block_k, expected_m, warp_m, warp_n, block_k, num_stages);
        printf("SMSIZE:%d, vreg:%d, stack:%d\n",int(SMSIZE), int(numRegs), int(localSize));
    }


}

} // namespace deep_gemm
