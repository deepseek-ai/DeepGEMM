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
#include "cute/arch/mma.hpp"
#include "../heuristics/common_int8.hpp"
#include "../heuristics/common_bf16.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/gemm.h"
#include "util/include/cutlass/util/packed_stride.hpp"
#include <deep_gemm/common/utils_rtc.cuh>
#include <deep_gemm/common/profiling_interface.cuh>
#include "int8_gemm.hpp"

using namespace deep_gemm_int8;
namespace deep_gemm {

class DynamicTileRuntime final : public LaunchRuntime<DynamicTileRuntime> {
public:
    struct LinearCombinationArgs {
        float alpha = 1.0f;
        float beta = 0.0f;
        float const* alpha_ptr = nullptr;
        float const* beta_ptr = nullptr;
        float const* const* alpha_ptr_array = nullptr;
        float const* const* beta_ptr_array = nullptr;
        float scale_a = float(1);
        float scale_b = float(1);
        float scale_c = float(1);
        float scale_d = float(1);
        float const* scale_a_ptr = nullptr;
        float const* scale_b_ptr = nullptr;
        float const* scale_c_ptr = nullptr;
        float const* scale_d_ptr = nullptr;
    };

    struct EpilogueArgs {
        LinearCombinationArgs callback;
        cutlass::bfloat16_t* ptr_C;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_C;

        cutlass::bfloat16_t* ptr_D;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_D;
    };

    struct LaunchInfo {
        int num_groups, num_stages, shape_n, shape_k;
        std::string gemm_type, kernel_type, kernel_name;
        bool k_large_em;
    };

    struct GemmArguments {
        const void* ptr_A;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_A;
        const void* ptr_B;
        cute::Stride<int64_t, cute::Int<1>, int64_t> stride_B;
        float const* ptr_scale_A;
        float const* ptr_scale_B;
        EpilogueArgs epi_params;
        uint32_t shape_m;
        int* grouped_layout;
    };

    struct Args {
        //   GemmArguments gemm_args;
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        GemmArguments kernel_params;
        std::string type_info;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#define INT8_HGRTC
#include <deep_gemm/impls/int8_gemm_cutlass3.cuh>
namespace deep_gemm {{
using namespace cute;
using cutlass::KernelHardwareInfo;

constexpr int SHAPE_N = {};
constexpr int SHAPE_K = {};
constexpr int NUM_GROUPS = {};
constexpr int STAGES = {};
static constexpr GemmType kGemmType = GemmType::{};
static constexpr bool kLargeEM = {};

using         ElementA    = {};
using         LayoutA     = cutlass::layout::RowMajor;

using         ElementB    = ElementA;
using         LayoutB     = cutlass::layout::ColumnMajor;

using         ElementD    = cutlass::bfloat16_t;
using         LayoutD     = cutlass::layout::RowMajor;

using         ElementC    = ElementD;
using         LayoutC     = LayoutD;

using ElementAccumulator  = cute::conditional_t<
    cute::is_same_v<ElementA, int8_t>,
    int32_t,
    float
>;
using ElementCompute      = float;
using GemmKernel = cutlass::gemm::kernel::DeepGemmDynamicTile<
                    kGemmType, ElementA, ElementB, ElementD, ElementAccumulator, ElementCompute,
                    SHAPE_N, SHAPE_K, NUM_GROUPS, kLargeEM>;

extern "C"
__launch_bounds__(GemmKernel::MaxThreadsPerBlock, GemmKernel::MinBlocksPerMultiprocessor)
__global__ void {}(
  typename GemmKernel::Params params
) {{
  extern __shared__ char smem[];
  int* grouped_layout = nullptr;
  GemmKernel op;
  op(params, smem);
}}
}}
)",
                           args.launch_info.shape_n, args.launch_info.shape_k, args.launch_info.num_groups,
                           args.launch_info.num_stages, args.launch_info.gemm_type, args.launch_info.k_large_em,
                           args.type_info, args.launch_info.kernel_name);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

class GemvRuntime final : public LaunchRuntime<GemvRuntime> {
public:
    struct GemvtArgs {
        int N;
        int K;
        void* a_ptr;
        void* b_ptr;
        void* c_ptr;

        // use for int8
        void* topk_weights_ptr{nullptr};
        const float* alphaCol{nullptr};
        const float* alphaRow{nullptr};

        int64_t num_tokens;
        int num_experts;
        int* expert_ids_ptr;

        int64_t stride_am;
        int64_t stride_ak;
        int64_t stride_be;
        int64_t stride_bk;
        int64_t stride_bn;
        int64_t stride_cm;
        int64_t stride_cn;
        int64_t stride_asm;
        int64_t stride_ask;
        int64_t stride_bse;
        int64_t stride_bsk;
        int64_t stride_bsn;
        int64_t total_blocks;
    };

    struct LaunchInfo {
        int BlockSize, ThreadPerN, NPerThread, NUM_UNROLL, SWZL_SIZE_M;
        bool SmallK;
        std::string src_type, acc_type, kernel_name;
    };

    struct Args {
        //   GemmArguments gemm_args;
        LaunchInfo launch_info;
        LaunchArgs launch_args;
        GemvtArgs kernel_params;
        std::string type_info;
    };

    static std::string generate_impl(const Args& args) {
        if (!args.launch_info.SmallK) {
            return fmt::format(R"(
#define INT8_HGRTC
#include <deep_gemm/impls/gemvt.cuh>
namespace deep_gemm {{
constexpr int BlockSize = {};
constexpr int ThreadPerN = {};
constexpr int NPerThread = {};
constexpr int NUM_UNROLL = {};
constexpr int SWZL_SIZE_M = {};

using load_atype = int4;
using load_btype = int4;

extern "C" __global__
void {}(const GemvtArgs args) {{
    batched_gemvt_kernel_impl<
        {}, __ppu_bfloat16, {},
        load_atype, load_btype,
        BlockSize, ThreadPerN, NPerThread, NUM_UNROLL, SWZL_SIZE_M
    >(args);
}}

}}
)",
                               args.launch_info.BlockSize, args.launch_info.ThreadPerN, args.launch_info.NPerThread,
                               args.launch_info.NUM_UNROLL, args.launch_info.SWZL_SIZE_M, args.launch_info.kernel_name,
                               args.launch_info.src_type, args.launch_info.acc_type);
        } else {
            return fmt::format(R"(
#define INT8_HGRTC
#include <deep_gemm/impls/gemvt.cuh>
namespace deep_gemm {{
constexpr int BlockSize = {};
constexpr int ThreadPerN = {};
constexpr int NPerThread = {};
constexpr int NUM_UNROLL = {};
constexpr int SWZL_SIZE_M = {};

using load_atype = int4;
using load_btype = int4;

extern "C" __global__
void {}(const GemvtArgs args) {{
    batched_gemvt_kernel_small_k_impl<
        {}, __ppu_bfloat16, {},
        load_atype, load_btype,
        BlockSize, ThreadPerN, NPerThread, 1, SWZL_SIZE_M, 5
    >(args);
}}

}}
)",
                               args.launch_info.BlockSize, args.launch_info.ThreadPerN, args.launch_info.NPerThread,
                               args.launch_info.NUM_UNROLL, args.launch_info.SWZL_SIZE_M, args.launch_info.kernel_name,
                               args.launch_info.src_type, args.launch_info.acc_type);
        }
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_HGGC_CHECK(launch_kernel(kernel, config, args.kernel_params));
    }
};

using ConfigTuple = std::tuple<int, int, int, int, int, int, int, std::tuple<int, int, int>>;

static void m_grouped_gemm_a8w8_per_channel_nt_contiguous_impl(
    const torch::Tensor& lhs, const torch::Tensor& lhs_scales, const torch::Tensor& rhs,
    const torch::Tensor& rhs_scales, const torch::Tensor& out, const torch::Tensor& m_indices, const int& m,
    const int& n, const int& k, const int& num_groups, std::optional<ConfigTuple> configs) {
    int num_sms = get_num_sms();
    ConfigTuple cfg;
    if (configs.has_value()) {
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        cfg = std::make_tuple(ns, bm, bn, bk, wm, wn, nst,
            deep_gemm_int8::get_smem_config(nst, k, bm, bn, bk, 1));
    } else {
        cfg = deep_gemm_int8::get_best_configs(
            m, n, k, num_groups, num_sms, GemmType::GroupedContiguous);
    }

    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config] = cfg;
    int expected_m = ceil_div(m, num_groups);
    auto SMSIZE = std::get<0>(smem_config);

    int kNumGroups = num_groups;

    using StrideA = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideB = cute::Stride<int64_t, cute::Int<1>, int64_t>;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, n, 1));

    int32_t* grouped_layout = reinterpret_cast<int32_t*>(m_indices.data_ptr<int32_t>());
    float* scales_a_ptr = lhs_scales.data_ptr<float>();
    float* scales_b_ptr = rhs_scales.data_ptr<float>();

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.cu_count = num_sms_new;
    dim3 const block = (block_m / warp_m) * (block_n / warp_n) * 32;
    dim3 grid = get_grid_shape(hw_info.cu_count);

    static constexpr GemmType kGemmType = GemmType::GroupedContiguous;
    torch::Dtype dtype = lhs.dtype().toScalarType();

    const void* converted_input_a = nullptr;
    const void* converted_input_b = nullptr;
    cutlass::bfloat16_t* converted_output = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>());
    std::string type_info;
    std::string kernel_name;
    std::string profile_type;
    if (dtype == torch::kInt8) {
        converted_input_a = lhs.data_ptr<int8_t>();
        converted_input_b = rhs.data_ptr<int8_t>();
        type_info = "int8_t";
        kernel_name = "int8_grouped_deep_gemm_contiguous";
        profile_type = "int8";
    } else {
        converted_input_a = reinterpret_cast<const void*>(lhs.data_ptr<at::Float8_e4m3fn>());
        converted_input_b = reinterpret_cast<const void*>(rhs.data_ptr<at::Float8_e4m3fn>());
        type_info = "cutlass::float_e4m3_t";
        kernel_name = "fp8_grouped_deep_gemm_contiguous";
        profile_type = "fp8";
    }

    const auto gemm_args = INT8GemmCutlass3Runtime::GemmArguments{
        .mode = cutlass::gemm::GemmUniversalMode::kGemm,
        .problem_shape = {m, n, k, 1},
        .mainloopargs = {converted_input_a, stride_A, converted_input_b, stride_B, scales_a_ptr, scales_b_ptr},
        .epilogueargs =
            {
                {1, 0},
                nullptr,
                stride_D,
                converted_output,
                stride_D,
            },
        .hw_info = hw_info,
        .scheduler = {(uint32_t)m, grouped_layout},
        .signal = nullptr};

    INT8GemmCutlass3Runtime::GemmKernelParams params =
        INT8GemmCutlass3Runtime::to_underlying_arguments_rtc(gemm_args, nullptr);

    auto args = INT8GemmCutlass3Runtime::Args{.launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups,
                                                              num_stages, "GroupedContiguous", "Default",
                                                              kernel_name, false},
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
        dg_prof_params.set_params(kGemmType, false, profile_type, kNumGroups, m, n, k, expected_m, grouped_layout,
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

        printf("[GroupedContiguous:]\n");
        printf("group:%d, problem:[%d, %d, %d]\n", num_groups, m, n, k);
        printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu,
               args.launch_args.grid_dim.x);
        printf("ThreadblockShape[%d, %d, %d], expected_m:%d, WarpShape[%d, %d, %d], num_stages:%d\n", block_m,
               block_n, block_k, expected_m, warp_m, warp_n, block_k, num_stages);
        printf("SMSIZE:%d, vreg:%d, stack:%d\n", int(SMSIZE), int(numRegs), int(localSize));
    }

}

// real path: m_grouped_gemm_a8w8_per_channel_nt_masked_impl
static std::pair<int, int> m_grouped_gemm_a8w8_per_channel_nt_masked_impl(
    const torch::Tensor& lhs, const torch::Tensor& lhs_scales, const torch::Tensor& rhs,
    const torch::Tensor& rhs_scales, const torch::Tensor& out, const torch::Tensor& masked_m, const int& m,
    const int& n, const int& k, const int& num_groups, const int& expected_m, std::optional<ConfigTuple> configs,
    int max_block_n, bool enable_sbo_overlap, const torch::Tensor& signal) {
    int num_sms = get_num_sms();
    ConfigTuple cfg;
    if (configs.has_value()) {
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        cfg = std::make_tuple(ns, bm, bn, bk, wm, wn, nst,
            deep_gemm_int8::get_smem_config(nst, k, bm, bn, bk, 1));
    } else {
        cfg = deep_gemm_int8::get_best_configs(
            expected_m, n, k, num_groups, num_sms, GemmType::GroupedMasked, max_block_n);
    }

    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config] = cfg;
    auto SMSIZE = std::get<0>(smem_config);
    int kNumGroups = num_groups;

    const auto extra_info = get_extra_info();
    bool enable_moe_dynamic_tile = extra_info.at("use_moe_dynamic_tile");
    std::vector<std::pair<int, int>> supported_N_K_list = {
        {4096, 7168},
        {7168, 2048}, // dpsk
        {4096, 8192},
        {8192, 2048} // qwen3.5
    };

    if (is_ppu1v5_device() && std::find(supported_N_K_list.begin(), supported_N_K_list.end(), std::make_pair(n, k)) !=
                                  supported_N_K_list.end()) {
        enable_moe_dynamic_tile = true;
    }

    if (enable_sbo_overlap) {
        // disable dynamic tile if enable_sbo_overlap, as the kNumNBlocks is not static
        enable_moe_dynamic_tile = false;
    }

    using StrideA = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideB = cute::Stride<int64_t, cute::Int<1>, int64_t>;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, n, 1));

    int32_t* grouped_layout = reinterpret_cast<int32_t*>(masked_m.data_ptr<int32_t>());
    float* scales_a_ptr = lhs_scales.data_ptr<float>();
    float* scales_b_ptr = rhs_scales.data_ptr<float>();

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.cu_count = num_sms_new;
    dim3 const block = (block_m / warp_m) * (block_n / warp_n) * 32;
    dim3 grid = get_grid_shape(hw_info.cu_count);

    static constexpr GemmType kGemmType = GemmType::GroupedMasked;
    torch::Dtype dtype = lhs.dtype().toScalarType();

    int32_t* signal_ptr = reinterpret_cast<int32_t*>(signal.data_ptr<int32_t>());

    const void* converted_input_a = nullptr;
    const void* converted_input_b = nullptr;
    cutlass::bfloat16_t* converted_output = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>());
    std::string type_info;
    std::string kernel_name;
    std::string profile_type;

    if (dtype == torch::kInt8) {
        converted_input_a = lhs.data_ptr<int8_t>();
        converted_input_b = rhs.data_ptr<int8_t>();
        type_info = "int8_t";
        kernel_name = "int8_grouped_deep_gemm_masked";
        profile_type = "int8";
    } else {
        converted_input_a = reinterpret_cast<const void*>(lhs.data_ptr<at::Float8_e4m3fn>());
        converted_input_b = reinterpret_cast<const void*>(rhs.data_ptr<at::Float8_e4m3fn>());
        type_info = "cutlass::float_e4m3_t";
        kernel_name = "fp8_grouped_deep_gemm_masked";
        profile_type = "fp8";
    }

    if (enable_moe_dynamic_tile) {
        kernel_name += "_dynamic_tile";
        auto gemm_args = DynamicTileRuntime::GemmArguments{
            .ptr_A = converted_input_a,
            .stride_A = stride_A,
            .ptr_B = converted_input_b,
            .stride_B = stride_B,
            .ptr_scale_A = scales_a_ptr,
            .ptr_scale_B = scales_b_ptr,
            .epi_params = {{1.0f, 0.0f}, converted_output, stride_D, converted_output, stride_D},
            .shape_m = m,
            .grouped_layout = grouped_layout};
        bool kLargeEM = false;
        if (expected_m > 73) {
            kLargeEM = true;
        }
        int flag = 0;
        static constexpr bool IsEP = (kGemmType == GemmType::GroupedMasked);

        if (IsEP) {
            if (kLargeEM) {
                flag = 1; // KernelAiuDynamicTileLargeEM  {256, 256, 64, 64, 128, 4}
                block_m = 256;
                block_n = 256;
                block_k = 128;
                warp_m = 64;
                warp_n = 64;
                num_stages = 4;
            } else {
                if (k > 2048) {
                    flag = 2; // KernelAiuDynamicTileLargeK {192, 128, 48, 64, 128, 3}
                    block_m = 192;
                    block_n = 128;
                    block_k = 128;
                    warp_m = 48;
                    warp_n = 64;
                    num_stages = 3;
                } else {
                    flag = 3; // KernelAiuDynamicTileTPSmallK {128, 128, 64, 64, 128, 2}
                    block_m = 128;
                    block_n = 128;
                    block_k = 128;
                    warp_m = 64;
                    warp_n = 64;
                    num_stages = 2;
                }
            }
        } else {
            if (k > 384) {
                flag = 4; // KernelAiuDynamicTileTPLargeK {128, 256, 64, 64, 128, 2}
                block_m = 128;
                block_n = 256;
                block_k = 128;
                warp_m = 64;
                warp_n = 64;
                num_stages = 2;
            } else {
                flag = 3; // KernelAiuDynamicTileTPSmallK
                block_m = 128;
                block_n = 128;
                block_k = 128;
                warp_m = 64;
                warp_n = 64;
                num_stages = 2;
            }
        }
        dim3 const block_new = (block_m / warp_m) * (block_n / warp_n) * 32;
        smem_config = deep_gemm_int8::get_smem_config(num_stages, k, block_m, block_n, block_k, 1);
        SMSIZE = std::get<0>(smem_config);
        auto args = DynamicTileRuntime::Args{
            .launch_info = {kNumGroups, num_stages, n, k, "GroupedMasked", "MoeDynamicTile", kernel_name, kLargeEM},
            .launch_args = {grid, block_new, SMSIZE},
            .kernel_params = gemm_args,
            .type_info = type_info};
        const auto& code = DynamicTileRuntime::generate(args);
        const auto& runtime = compiler->build(kernel_name, code);
        const auto& kernel = runtime->kernel;
        int blocks_per_cu = 0;
        HGresult result = hgOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, kernel, block_new.x, SMSIZE);
        args.launch_args.grid_dim.x *= blocks_per_cu;

        DgProfParam dg_prof_params;
        if (ProfilingInterface::Instance().get_op_info()) {
            dg_prof_params.set_params(kGemmType, false, profile_type, kNumGroups, m, n, k, expected_m, grouped_layout,
                                      current_stream());
        }
        ProfilingInterface::Instance().instrument(true, dg_prof_params);

        DynamicTileRuntime::launch(runtime, args);

        ProfilingInterface::Instance().instrument(false, dg_prof_params);
        return std::make_pair(block_m, ceil_div(n, block_n));
    }

    const auto gemm_args = INT8GemmCutlass3Runtime::GemmArguments{
        .mode = cutlass::gemm::GemmUniversalMode::kGemm,
        .problem_shape = {m, n, k, 1},
        .mainloopargs = {converted_input_a, stride_A, converted_input_b, stride_B, scales_a_ptr, scales_b_ptr},
        .epilogueargs =
            {
                {1, 0},
                nullptr,
                stride_D,
                converted_output,
                stride_D,
            },
        .hw_info = hw_info,
        .scheduler = {(uint32_t)m, grouped_layout},
        .signal = signal_ptr};

    INT8GemmCutlass3Runtime::GemmKernelParams params =
        INT8GemmCutlass3Runtime::to_underlying_arguments_rtc(gemm_args, nullptr);

    auto args = INT8GemmCutlass3Runtime::Args{.launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups,
                                                              num_stages, "GroupedMasked", "Default", kernel_name,
                                                              enable_sbo_overlap},
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
        dg_prof_params.set_params(kGemmType, false, profile_type, kNumGroups, m, n, k, expected_m, grouped_layout,
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
        printf("[GroupedMasked:]\n");
        printf("group:%d, problem:[%d, %d, %d]\n", num_groups, m, n, k);
        printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu,
               args.launch_args.grid_dim.x);
        printf("ThreadblockShape[%d, %d, %d], expected_m:%d, WarpShape[%d, %d, %d], num_stages:%d\n", block_m,
               block_n, block_k, expected_m, warp_m, warp_n, block_k, num_stages);
        printf("SMSIZE:%d, vreg:%d, stack:%d\n", int(SMSIZE), int(numRegs), int(localSize));
    }


    return std::make_pair(block_m, ceil_div(n, block_n));
}

static void m_grouped_gemm_a8w8_per_channel_nt_nopad_impl(const torch::Tensor& lhs, const torch::Tensor& lhs_scales,
                                                          const torch::Tensor& rhs, const torch::Tensor& rhs_scales,
                                                          const torch::Tensor& out, const torch::Tensor& m_indices,
                                                          const int& m, const int& n, const int& k,
                                                          const int& num_groups,
                                                          std::optional<const torch::Tensor> m_rows,
                                                          std::optional<ConfigTuple> configs) {
    int BlockSize, ThreadPerN, NUM_UNROLL, SWZL_SIZE_M, NPerThread;
    bool SMALL_K;
    int num_sms = get_num_sms();
    int expected_m = ceil_div(m, num_groups);
    bool use_gemv = false;
    if (k % 16 == 0 && ((m <= 2 * num_groups * 0.75 && k <= 32 * 8) || (m < 0.65 * num_groups && k > 256)) &&
        !is_ppu1v5_device()) {
        std::tie(BlockSize, ThreadPerN, NUM_UNROLL, SWZL_SIZE_M, NPerThread, SMALL_K) =
            deep_gemm_bf16_common::get_gemv_best_configs(m, n, k, num_groups, num_sms, torch::kInt8);
        if (ThreadPerN != -1 and lhs.dtype().toScalarType() == torch::kInt8) {
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
            dg_prof_params.set_params(GemmType::GroupedNoPad, true, std::string("int8"), num_groups, m, n, k, expected_m,
                                      layout_info, current_stream());
        }
        cutlass::bfloat16_t* converted_output = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>());
        auto gemmv_args = GemvRuntime::GemvtArgs{
            .N = n,
            .K = k,
            .a_ptr = lhs.data_ptr<int8_t>(),
            .b_ptr = rhs.data_ptr<int8_t>(),
            .c_ptr = converted_output,
            .topk_weights_ptr = nullptr,
            .alphaCol = rhs_scales.data_ptr<float>(),
            .alphaRow = lhs_scales.data_ptr<float>(),
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
        using src_type = int8_t;
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
                                                          SMALL_K, "int8_t", "int", "batched_gemvt_kernel"},
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

                printf("[GemV-INT8:]\n");
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
            auto args = GemvRuntime::Args{.launch_info = {BlockSize, ThreadPerN, NPerThread, NUM_UNROLL, SWZL_SIZE_M,
                                                          SMALL_K, "int8_t", "int", "batched_gemvt_small_k_kernel"},
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

                printf("[GemV-INT8:]\n");
                printf("group:%d, problem:[%d, %d, %d]\n", num_groups, m, n, k);
                printf("BlockSize:%d, NPerThread:%d, ThreadPerN:%d, NPerBlock:%d, NUM_UNROLL:%d, SWZL_SIZE_M:%d\n",
                       BlockSize, NPerThread, ThreadPerN, NPerBlock, NUM_UNROLL, SWZL_SIZE_M);
                printf("threadblock_count:%d, vreg:%d, stack:%d\n", gemmv_args.total_blocks, int(numRegs),
                       int(localSize));
            }
            return;
        }
    }

    ConfigTuple cfg;
    if (configs.has_value()) {
        auto [ns, bm, bn, bk, wm, wn, nst, _sc] = *configs;
        cfg = std::make_tuple(ns, bm, bn, bk, wm, wn, nst,
            deep_gemm_int8::get_smem_config(nst, k, bm, bn, bk, 1));
    } else {
        cfg = deep_gemm_int8::get_best_configs(
            expected_m, n, k, num_groups, num_sms, GemmType::GroupedNoPad);
    }

    auto [num_sms_new, block_m, block_n, block_k, warp_m, warp_n, num_stages, smem_config] = cfg;
    auto SMSIZE = std::get<0>(smem_config);

    int kNumGroups = num_groups;

    using StrideA = cute::Stride<int64_t, cute::Int<1>, int64_t>;
    using StrideB = cute::Stride<int64_t, cute::Int<1>, int64_t>;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, n, 1));

    float* scales_a_ptr = lhs_scales.data_ptr<float>();
    float* scales_b_ptr = rhs_scales.data_ptr<float>();

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.cu_count = num_sms_new;
    dim3 const block = (block_m / warp_m) * (block_n / warp_n) * 32;
    dim3 grid = get_grid_shape(hw_info.cu_count);

    static constexpr GemmType kGemmType = GemmType::GroupedNoPad;

    at::Tensor m_rows_tensor;
    if (!m_rows.has_value() || !m_rows->defined()) {
        at::Tensor counts = at::bincount(m_indices);
        int64_t min_n = std::min<int64_t>(counts.size(0), num_groups);
        at::Tensor experts_for_rows =
            at::zeros({num_groups}, at::TensorOptions().dtype(at::kInt).device(m_indices.device()));
        if (min_n > 0) {
            experts_for_rows.narrow(0, 0, min_n).copy_(counts.narrow(0, 0, min_n).to(at::kInt));
        }
        m_rows_tensor = experts_for_rows;
    } else {
        m_rows_tensor = *m_rows;
    }

    int64_t block_m_info_size = (num_groups + ceil_div(m + 1 - num_groups, block_m)) * 4;
    at::Tensor block_m_info =
        at::empty({block_m_info_size}, at::TensorOptions().dtype(at::kInt).device(m_rows_tensor.device()));

    layout_info = m_rows_tensor.data_ptr<int32_t>();
    torch::Dtype dtype = lhs.dtype().toScalarType();
    const void* converted_input_a = nullptr;
    const void* converted_input_b = nullptr;
    cutlass::bfloat16_t* converted_output = reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>());
    std::string type_info;
    std::string kernel_name;
    std::string profile_type;

    if (dtype == torch::kInt8) {
        converted_input_a = lhs.data_ptr<int8_t>();
        converted_input_b = rhs.data_ptr<int8_t>();
        type_info = "int8_t";
        kernel_name = "int8_grouped_deep_gemm_nopad";
        profile_type = "int8";
    } else {
        converted_input_a = reinterpret_cast<const void*>(lhs.data_ptr<at::Float8_e4m3fn>());
        converted_input_b = reinterpret_cast<const void*>(rhs.data_ptr<at::Float8_e4m3fn>());
        type_info = "cutlass::float_e4m3_t";
        kernel_name = "fp8_grouped_deep_gemm_nopad";
        profile_type = "fp8";
    }

    bool kIsNoPadPreprocessLayout = kNumGroups >= 128;
    if (kIsNoPadPreprocessLayout) {
        uint32_t block_size = std::max(32, next_power_of_two(kNumGroups));
        auto compute_block_info_args = ComputeBlockInfoKernelRuntime::Args{
            .launch_attr_args = {reinterpret_cast<const uint32_t*>(m_rows_tensor.data_ptr<int32_t>()), kNumGroups,
                                 reinterpret_cast<uint32_t*>(block_m_info.data_ptr<int32_t>())},
            .launch_args = {1, block_size, 0},
        };
        const auto& code_blockinfo = ComputeBlockInfoKernelRuntime::generate(block_m);
        const auto& runtime_blockinfo = compiler->build("computeBlockInfoKernel", code_blockinfo);
        ComputeBlockInfoKernelRuntime::launch(runtime_blockinfo, compute_block_info_args);
        layout_info = block_m_info.data_ptr<int32_t>();
    }

    const auto gemm_args = INT8GemmCutlass3Runtime::GemmArguments{
        .mode = cutlass::gemm::GemmUniversalMode::kGemm,
        .problem_shape = {m, n, k, 1},
        .mainloopargs = {converted_input_a, stride_A, converted_input_b, stride_B, scales_a_ptr, scales_b_ptr},
        .epilogueargs =
            {
                {1, 0},
                nullptr,
                stride_D,
                converted_output,
                stride_D,
            },
        .hw_info = hw_info,
        .scheduler = {(uint32_t)m, layout_info},
        .signal = nullptr};

    INT8GemmCutlass3Runtime::GemmKernelParams params =
        INT8GemmCutlass3Runtime::to_underlying_arguments_rtc(gemm_args, nullptr);

    // Default, MultistageOnN, MoeDynamicTile, OverlapPrologue, OverlapMainloop
    auto kernel_type = "Default";
    if (k < 4096 && is_ppu1v5_device()) {
        kernel_type = "OverlapPrologue";
    }
    auto args =
        INT8GemmCutlass3Runtime::Args{.launch_info = {block_m, block_n, block_k, warp_m, warp_n, kNumGroups,
                                                      num_stages, "GroupedNoPad", kernel_type, kernel_name, false},
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
        dg_prof_params.set_params(kGemmType, false, profile_type, kNumGroups, m, n, k, expected_m, layout_info,
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
        printf("[GroupedNoPad:]\n");
        printf("group:%d, problem:[%d, %d, %d]\n", num_groups, m, n, k);
        printf("num_sms:%d, max_active_tb_num:%d, threadblock_count:%d\n", num_sms_new, blocks_per_cu,
               args.launch_args.grid_dim.x);
        printf("ThreadblockShape[%d, %d, %d], expected_m:%d, WarpShape[%d, %d, %d], num_stages:%d\n", block_m,
               block_n, block_k, expected_m, warp_m, warp_n, block_k, num_stages);
        printf("SMSIZE:%d, vreg:%d, stack:%d\n", int(SMSIZE), int(numRegs), int(localSize));
    }


}
static void m_grouped_gemm_int8_int8_bf16_nt_contiguous_impl(const torch::Tensor& lhs, const torch::Tensor& lhs_scales,
                                                             const torch::Tensor& rhs, const torch::Tensor& rhs_scales,
                                                             const torch::Tensor& out, const torch::Tensor& m_indices,
                                                             const int& m, const int& n, const int& k,
                                                             const int& num_groups, std::optional<ConfigTuple> configs) {
    // Route to per-channel implementation
    m_grouped_gemm_a8w8_per_channel_nt_contiguous_impl(lhs, lhs_scales, rhs, rhs_scales, out, m_indices, m, n, k,
                                                       num_groups, configs);
}

// Routing layer: m_grouped_gemm_int8_int8_bf16_nt_xxx_impl -> m_grouped_gemm_a8w8_per_channel_nt_xxx
static std::pair<int, int> m_grouped_gemm_int8_int8_bf16_nt_masked_impl(
    const torch::Tensor& lhs, const torch::Tensor& lhs_scales, const torch::Tensor& rhs,
    const torch::Tensor& rhs_scales, const torch::Tensor& out, const torch::Tensor& masked_m, const int& m,
    const int& n, const int& k, const int& num_groups, const int& expected_m, std::optional<ConfigTuple> configs,
    int max_block_n, bool enable_sbo_overlap, const torch::Tensor& signal) {
    // Route to per-channel implementation
    return m_grouped_gemm_a8w8_per_channel_nt_masked_impl(lhs, lhs_scales, rhs, rhs_scales, out, masked_m, m, n, k,
                                                          num_groups, expected_m, configs, max_block_n,
                                                          enable_sbo_overlap, signal);
}

static void m_grouped_gemm_int8_int8_bf16_nt_nopad_impl(const torch::Tensor& lhs, const torch::Tensor& lhs_scales,
                                                        const torch::Tensor& rhs, const torch::Tensor& rhs_scales,
                                                        const torch::Tensor& out, const torch::Tensor& m_indices,
                                                        const int& m, const int& n, const int& k, const int& num_groups,
                                                        std::optional<const torch::Tensor> m_rows,
                                                        std::optional<ConfigTuple> configs) {
    // Route to per-channel implementation
    m_grouped_gemm_a8w8_per_channel_nt_nopad_impl(lhs, lhs_scales, rhs, rhs_scales, out, m_indices, m, n, k, num_groups,
                                                  m_rows, configs);
}

} // namespace deep_gemm
