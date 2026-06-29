#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <torch/python.h>
#include <unordered_set>
#include <vector>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/mega_moe_split.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/mega_moe.hpp"

namespace deep_gemm {

class SM100FP8FP4MegaMoESplitDispatchL1SwigluRuntime final :
    public LaunchRuntime<SM100FP8FP4MegaMoESplitDispatchL1SwigluRuntime> {
public:
    struct Args {
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        int num_sms;
        float activation_clamp;
        bool fast_math;
        bool local_only;
        MegaMoEConfig config;

        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        CUtensorMap tensor_map_l1_weights_sf;
        CUtensorMap tensor_map_l1_output;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe_split/dispatch_l1_swiglu.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &mega_moe_split::sm100_fp8_fp4_mega_moe_split_dispatch_l1_swiglu_impl<
            {},
            {}, {},
            {}, {},
            {},
            {}, {}, {},
            {},
            {}, {},
            {},
            {},
            {},
            {}, {}, {},
            {}, {},
            {},
            {},
            {}
        >);
}};
)", args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.num_experts_per_wave,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.store_block_m,
    args.config.sf_block_m, args.config.sf_block_n,
    args.config.num_max_pool_tokens,
    args.config.num_padded_sf_pool_tokens,
    args.config.num_stages,
    args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
    args.num_sms, args.num_ranks,
    to_string(args.activation_clamp),
    args.fast_math ? "true" : "false",
    args.local_only ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.cumulative_local_expert_recv_stats,
            args.num_tokens,
            args.sym_buffer_ptrs,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_weights,
            args.tensor_map_l1_weights_sf,
            args.tensor_map_l1_output
        ));
    }
};

class SM100FP8FP4MegaMoESplitL2CombineRuntime final :
    public LaunchRuntime<SM100FP8FP4MegaMoESplitL2CombineRuntime> {
public:
    struct Args {
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        int kernel1_sms, kernel2_sms;
        MegaMoEConfig config;

        void* state;
        uint32_t num_work_iters;
        layout::SymBuffer<> sym_buffer_ptrs;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        CUtensorMap tensor_map_l2_weights_sf;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe_split/l2_combine.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &mega_moe_split::sm100_fp8_fp4_mega_moe_split_l2_combine_impl<
            {},
            {}, {},
            {}, {},
            {}, {}, {},
            {},
            {}, {},
            {},
            {},
            {},
            {},
            {},
            {}, {}, {}
        >);
}};
)", args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.block_m, args.config.block_n, args.config.block_k,
    args.config.store_block_m,
    args.config.sf_block_m, args.config.sf_block_n,
    args.config.num_max_pool_tokens,
    args.config.num_padded_sf_pool_tokens,
    args.config.num_stages,
    args.config.num_non_epilogue_threads,
    args.config.num_epilogue_threads,
    args.kernel1_sms, args.kernel2_sms, args.num_ranks);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.state,
            args.num_work_iters,
            args.sym_buffer_ptrs,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.tensor_map_l2_weights_sf
        ));
    }
};

class SM100FP8FP4MegaMoESplitCombineReduceRuntime final :
    public LaunchRuntime<SM100FP8FP4MegaMoESplitCombineReduceRuntime> {
public:
    struct Args {
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_topk;
        int num_ranks;
        MegaMoEConfig config;

        void* y;
        void* state;
        uint32_t num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe_split/combine_reduce.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &mega_moe_split::sm100_fp8_fp4_mega_moe_split_combine_reduce_impl<
            {},
            {}, {},
            {}, {},
            {},
            {},
            512
        >);
}};
)", args.num_max_tokens_per_rank,
    args.hidden, args.intermediate_hidden,
    args.num_experts, args.num_topk,
    args.config.num_padded_sf_pool_tokens,
    args.num_ranks);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.y,
            args.state,
            args.num_tokens,
            args.sym_buffer_ptrs
        ));
    }
};

static void check_split_skeleton_tensor(const torch::Tensor& tensor, const at::ScalarType& dtype) {
    DG_HOST_ASSERT(tensor.is_cuda());
    DG_HOST_ASSERT(tensor.is_contiguous());
    DG_HOST_ASSERT(tensor.scalar_type() == dtype);
}

// SMEM layout / pipeline depth for the split dispatch_l1_swiglu (K1) kernel. K1 uses a
// route-based dispatch with a per-dispatch-warp token send/pull buffer in SMEM, so its SMEM
// formula differs from the fused megamoe heuristic (which sizes a pull buffer instead). This
// mirrors the K1 kernel's own SMEM partitioning exactly.
static std::pair<int, int> get_mega_moe_split_kernel1_pipeline(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k, const int& store_block_m,
    const int& sf_block_m, const int& sf_block_n, const int& gran_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps) {
    constexpr int kSmemAlignment = 1024;
    constexpr int kNumEpilogueStages = 2;
    constexpr int kNumTMAStoreStages = 2;

    const int load_block_m = block_m / 2;

    // Dispatch region: expert counts + per-dispatch-warp token send/pull buffers
    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(hidden), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;

    // C/D output region: max of L1 FP8 (2 TMA stages, BLOCK_N/2 post-SwiGLU) and L2 BF16 (1 stage)
    const auto num_epilogue_warpgroups = num_epilogue_warps / 4;
    const int smem_cd_l1 = num_epilogue_warpgroups * store_block_m * (block_n / 2) * kNumTMAStoreStages;
    const int smem_cd_l2 = num_epilogue_warpgroups * store_block_m * block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd = std::max(smem_cd_l1, smem_cd_l2);

    // Barriers: dispatch + tensor-memory full/empty + combine (2 per epilogue warp)
    const int smem_barriers = (num_dispatch_warps + kNumEpilogueStages * 2 + num_epilogue_warps * 2) * 8;

    // Amax reduction + tensor-memory pointer
    const int smem_amax_reduction = store_block_m * num_epilogue_warps * static_cast<int>(sizeof(float));
    const int smem_tmem_ptr = 4;

    const int smem_sfa_per_stage = sf_block_m * (block_k / gran_k);
    const int smem_sfb_per_stage = sf_block_n * (block_k / gran_k);
    const int smem_per_stage = load_block_m * block_k + block_n * block_k + smem_sfa_per_stage + smem_sfb_per_stage + 2 * 8;
    const int smem_fixed = smem_dispatch_size + smem_cd + smem_amax_reduction + smem_barriers + smem_tmem_ptr;

    const int num_stages = (smem_capacity - smem_fixed) / smem_per_stage;
    DG_HOST_ASSERT(num_stages >= 2);
    return {num_stages, smem_fixed + num_stages * smem_per_stage};
}

static MegaMoEConfig get_mega_moe_split_kernel1_config(
    const int& num_ranks,
    const int& num_experts,
    const int& num_experts_per_rank,
    const int& num_max_tokens_per_rank,
    const int& num_tokens,
    const int& num_topk,
    const int& hidden,
    const int& intermediate_hidden,
    const int& num_padded_sf_pool_tokens,
    const int& num_sms
) {
    auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden, num_padded_sf_pool_tokens);
    config.num_experts_per_wave = get_num_experts_per_wave_for_mega_moe(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, config.block_m, config.block_n, num_sms);
    // Override the pipeline depth / SMEM size with the split-K1 layout (the fused heuristic
    // sizes a different dispatch region, which under-allocates K1's send buffers).
    constexpr int kGranK = 32;
    const auto [num_stages, smem_size] = get_mega_moe_split_kernel1_pipeline(
        SM100ArchSpec::smem_capacity,
        num_experts, hidden,
        config.block_m, config.block_n, config.block_k, config.store_block_m,
        config.sf_block_m, config.sf_block_n, kGranK,
        config.num_dispatch_threads / 32, config.num_epilogue_threads / 32);
    config.num_stages = num_stages;
    config.smem_size = smem_size;
    return config;
}

static void print_mega_moe_split_kernel1_config(const SM100FP8FP4MegaMoESplitDispatchL1SwigluRuntime::Args& args) {
    if (not (get_env<int>("DG_PRINT_SPLIT_K1_CONFIG") or
             get_env<int>("DG_PRINT_CONFIGS") or
             get_env<int>("DG_JIT_DEBUG"))) {
        return;
    }

    const auto& config = args.config;
    const auto key = fmt::format(
        "split_k1:num_max_tokens_per_rank={},num_tokens={},hidden={},intermediate_hidden={},"
        "num_experts={},num_topk={},num_ranks={},num_sms={},activation_clamp={},fast_math={},"
        "local_only={},block_m={},block_n={},block_k={},store_block_m={},num_stages={}",
        args.num_max_tokens_per_rank, args.num_tokens, args.hidden, args.intermediate_hidden,
        args.num_experts, args.num_topk, args.num_ranks, args.num_sms,
        to_string(args.activation_clamp), args.fast_math, args.local_only,
        config.block_m, config.block_n, config.block_k, config.store_block_m, config.num_stages);

    static std::unordered_set<std::string> printed;
    if (printed.count(key) > 0)
        return;
    printed.insert(key);

    const auto num_threads = config.num_dispatch_threads +
        config.num_non_epilogue_threads + config.num_epilogue_threads;
    std::cout
        << "\n"
        << "SM100 FP8/FP4 MegaMoE Split K1 config\n"
        << " > kernel: sm100_fp8_fp4_mega_moe_split_dispatch_l1_swiglu_impl\n"
        << " > problem: num_tokens=" << args.num_tokens
        << ", num_max_tokens_per_rank=" << args.num_max_tokens_per_rank
        << ", hidden=" << args.hidden
        << ", intermediate_hidden=" << args.intermediate_hidden
        << ", num_experts=" << args.num_experts
        << ", num_topk=" << args.num_topk
        << ", num_ranks=" << args.num_ranks << "\n"
        << " > runtime: num_sms=" << args.num_sms
        << ", activation_clamp=" << args.activation_clamp
        << ", fast_math=" << args.fast_math
        << ", local_only=" << args.local_only << "\n"
        << " > launch: grid=(" << args.launch_args.grid_dim.first
        << ", " << args.launch_args.grid_dim.second
        << "), threads=" << args.launch_args.num_threads
        << ", smem=" << args.launch_args.smem_size
        << " bytes, cluster_dim=" << args.launch_args.cluster_dim
        << ", enable_pdl=" << args.launch_args.enable_pdl << "\n"
        << " > thread layout: dispatch=" << config.num_dispatch_threads
        << ", non_epilogue=" << config.num_non_epilogue_threads
        << ", epilogue=" << config.num_epilogue_threads
        << ", total=" << num_threads << "\n"
        << " > tiles: block_m=" << config.block_m
        << ", block_n=" << config.block_n
        << ", block_k=" << config.block_k
        << ", load_block_m=" << config.load_block_m
        << ", load_block_n=" << config.load_block_n
        << ", store_block_m=" << config.store_block_m << "\n"
        << " > scale-factor tiles: sf_block_m=" << config.sf_block_m
        << ", sf_block_n=" << config.sf_block_n
        << ", num_padded_sf_pool_tokens=" << config.num_padded_sf_pool_tokens << "\n"
        << " > pool/waves: num_max_pool_tokens=" << config.num_max_pool_tokens
        << ", num_experts_per_wave=" << config.num_experts_per_wave << "\n"
        << " > swizzle: acts=" << config.swizzle_acts_mode
        << ", weights=" << config.swizzle_weights_mode << "\n"
        << " > pipeline: num_stages=" << config.num_stages
        << ", smem_size=" << config.smem_size << " bytes\n"
        << " > template compact: <"
        << args.num_max_tokens_per_rank << ", "
        << args.hidden << ", " << args.intermediate_hidden << ", "
        << args.num_experts << ", " << args.num_topk << ", "
        << config.num_experts_per_wave << ", "
        << config.block_m << ", " << config.block_n << ", " << config.block_k << ", "
        << config.store_block_m << ", "
        << config.sf_block_m << ", " << config.sf_block_n << ", "
        << config.num_max_pool_tokens << ", "
        << config.num_padded_sf_pool_tokens << ", "
        << config.num_stages << ", "
        << config.num_dispatch_threads << ", " << config.num_non_epilogue_threads << ", "
        << config.num_epilogue_threads << ", "
        << args.num_sms << ", " << args.num_ranks << ", "
        << to_string(args.activation_clamp) << ", "
        << (args.fast_math ? "true" : "false") << ", "
        << (args.local_only ? "true" : "false") << ">\n"
        << " > template parameters:\n"
        << "   [00] kNumMaxTokensPerRank     = " << args.num_max_tokens_per_rank << "\n"
        << "   [01] kHidden                  = " << args.hidden << "\n"
        << "   [02] kIntermediateHidden      = " << args.intermediate_hidden << "\n"
        << "   [03] kNumExperts              = " << args.num_experts << "\n"
        << "   [04] kNumTopk                 = " << args.num_topk << "\n"
        << "   [05] kNumExpertsPerWave       = " << config.num_experts_per_wave << "\n"
        << "   [06] BLOCK_M                  = " << config.block_m << "\n"
        << "   [07] BLOCK_N                  = " << config.block_n << "\n"
        << "   [08] BLOCK_K                  = " << config.block_k << "\n"
        << "   [09] STORE_BLOCK_M            = " << config.store_block_m << "\n"
        << "   [10] SF_BLOCK_M               = " << config.sf_block_m << "\n"
        << "   [11] SF_BLOCK_N               = " << config.sf_block_n << "\n"
        << "   [12] kNumMaxPoolTokens        = " << config.num_max_pool_tokens << "\n"
        << "   [13] kNumPaddedSFPoolTokens   = " << config.num_padded_sf_pool_tokens << "\n"
        << "   [14] kNumStages               = " << config.num_stages << "\n"
        << "   [15] kNumDispatchThreads      = " << config.num_dispatch_threads << "\n"
        << "   [16] kNumNonEpilogueThreads   = " << config.num_non_epilogue_threads << "\n"
        << "   [17] kNumEpilogueThreads      = " << config.num_epilogue_threads << "\n"
        << "   [18] kNumSMs                  = " << args.num_sms << "\n"
        << "   [19] kNumRanks                = " << args.num_ranks << "\n"
        << "   [20] kActivationClamp         = " << to_string(args.activation_clamp)
        << " (" << args.activation_clamp << ")\n"
        << "   [21] kFastMath                = " << (args.fast_math ? "true" : "false") << "\n"
        << "   [22] kLocalOnly               = " << (args.local_only ? "true" : "false") << "\n"
        << " > derived template defaults:\n"
        << "   L1_SHAPE_N                    = " << args.intermediate_hidden * 2
        << "  (kIntermediateHidden * 2)\n"
        << "   L1_SHAPE_K                    = " << args.hidden
        << "  (kHidden)\n"
        << "   L2_SHAPE_N                    = " << args.hidden
        << "  (kHidden)\n"
        << "   L2_SHAPE_K                    = " << args.intermediate_hidden
        << "  (kIntermediateHidden)\n"
        << "   kNumDispatchWarps             = " << config.num_dispatch_threads / 32 << "\n"
        << "   kNumMMANonEpilogueWarps       = " << config.num_non_epilogue_threads / 32 << "\n"
        << "   kNumEpilogueWarps             = " << config.num_epilogue_threads / 32 << "\n"
        << "   kNumEpilogueWarpgroups        = " << config.num_epilogue_threads / 128 << "\n"
        << "   kNumThreads                   = " << num_threads << "\n"
        << "   kNumTokensPerWarp             = " << 32 / args.num_topk << "\n"
        << "   kNumExpertsPerRank            = " << args.num_experts / args.num_ranks << "\n"
        << std::endl;
}

static MegaMoEConfig get_mega_moe_split_kernel2_config(
    MegaMoEConfig config
) {
    constexpr int kNumEpilogueStages = 2;
    constexpr int kNumMaxStages = 32;
    constexpr int kGranK = 32;

    const int load_block_m = config.block_m / 2;
    const int num_epilogue_warpgroups = config.num_epilogue_threads / 128;
    const int smem_cd_l2 = num_epilogue_warpgroups * config.store_block_m * config.block_n *
        static_cast<int>(sizeof(nv_bfloat16));
    const int smem_sfa_per_stage = config.sf_block_m * (config.block_k / kGranK);
    const int smem_sfb_per_stage = config.sf_block_n * (config.block_k / kGranK);
    const int smem_per_stage =
        load_block_m * config.block_k +
        config.block_n * config.block_k +
        smem_sfa_per_stage + smem_sfb_per_stage +
        2 * static_cast<int>(sizeof(uint64_t));
    const int smem_fixed =
        smem_cd_l2 +
        kNumEpilogueStages * 2 * static_cast<int>(sizeof(uint64_t)) +
        static_cast<int>(sizeof(uint32_t));
    const int num_stages = std::min(
        (SM100ArchSpec::smem_capacity - smem_fixed) / smem_per_stage,
        kNumMaxStages);
    DG_HOST_ASSERT(num_stages >= 2);

    config.num_stages = num_stages;
    config.smem_size = smem_fixed + num_stages * smem_per_stage;
    return config;
}

#if CUDART_VERSION >= 13010

class SM100FP8FP4MegaMoESplitGraph final {
private:
    struct BufferViews {
        torch::Tensor l1_acts;
        torch::Tensor l1_acts_sf;
        torch::Tensor l2_acts;
        torch::Tensor l2_acts_sf;
    };

    struct Kernel1NodeArgs {
        int* cumulative_local_expert_recv_stats;
        uint32_t num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        CUtensorMap tensor_map_l1_weights_sf;
        CUtensorMap tensor_map_l1_output;

        Kernel1NodeArgs() = default;
    };

    struct Kernel2NodeArgs {
        void* state;
        uint32_t num_work_iters;
        layout::SymBuffer<> sym_buffer_ptrs;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        CUtensorMap tensor_map_l2_weights_sf;

        Kernel2NodeArgs() = default;
    };

    struct Kernel3NodeArgs {
        void* y;
        void* state;
        uint32_t num_tokens;
        layout::SymBuffer<> sym_buffer_ptrs;

        Kernel3NodeArgs() = default;

        Kernel3NodeArgs(
            void* y,
            void* state,
            const uint32_t& num_tokens,
            const layout::SymBuffer<>& sym_buffer_ptrs
        ) : y(y),
            state(state),
            num_tokens(num_tokens),
            sym_buffer_ptrs(sym_buffer_ptrs) {}
    };

    std::vector<torch::Tensor> states_;
    std::vector<torch::Tensor> ys_;
    std::vector<torch::Tensor> sym_buffers_;
    std::vector<std::vector<int64_t>> sym_buffer_ptrs_;
    std::vector<torch::Tensor> l1_weights_;
    std::vector<torch::Tensor> l1_weights_sf_;
    std::vector<torch::Tensor> l2_weights_;
    std::vector<torch::Tensor> l2_weights_sf_;
    std::vector<torch::Tensor> stats_;

    uint32_t rank_idx_ = 0;
    uint32_t num_max_tokens_per_rank_ = 0;
    uint32_t num_experts_ = 0;
    uint32_t num_experts_per_rank_ = 0;
    uint32_t num_topk_ = 0;
    uint32_t num_tokens_ = 0;
    uint32_t hidden_ = 0;
    uint32_t intermediate_hidden_ = 0;
    uint32_t kernel1_sms_ = 0;
    uint32_t kernel2_sms_ = 0;
    uint32_t reduce_sms_ = 0;
    uint32_t kernel2_work_iters_ = 0;
    uint32_t reduce_work_iters_ = 0;
    float activation_clamp_ = 0.0f;
    bool fast_math_ = true;
    MegaMoEConfig config_;
    MegaMoEConfig kernel2_config_;

    std::shared_ptr<KernelRuntime> kernel1_runtime_;
    std::shared_ptr<KernelRuntime> kernel2_runtime_;
    std::shared_ptr<KernelRuntime> kernel3_runtime_;
    KernelHandle kernel1_graph_kernel_ = nullptr;
    KernelHandle kernel2_graph_kernel_ = nullptr;
    KernelHandle kernel3_graph_kernel_ = nullptr;

    int device_idx_ = 0;
    cudaExecutionContext_t primary_context_ = nullptr;
    std::array<cudaExecutionContext_t, 2> green_contexts_ = {nullptr, nullptr};
    std::array<unsigned long long, 2> green_context_ids_ = {0, 0};

    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    std::vector<cudaGraphNode_t> kernel1_nodes_;
    std::vector<cudaGraphNode_t> kernel2_nodes_;
    std::vector<cudaGraphNode_t> reduce_nodes_;

    std::vector<Kernel1NodeArgs> kernel1_args_;
    std::vector<Kernel2NodeArgs> kernel2_args_;
    std::vector<Kernel3NodeArgs> kernel3_args_;
    std::vector<std::array<void*, 8>> kernel1_params_;
    std::vector<std::array<void*, 7>> kernel2_params_;
    std::vector<std::array<void*, 4>> kernel3_params_;

    static uint32_t checked_nonnegative_u32(const int& value) {
        DG_HOST_ASSERT(value >= 0);
        return static_cast<uint32_t>(value);
    }

    static uint32_t checked_positive_u32(const int& value) {
        DG_HOST_ASSERT(value > 0);
        return static_cast<uint32_t>(value);
    }

    static int get_num_padded_sf_pool_tokens(
        const int& num_max_pool_tokens
    ) {
        int num_padded_sf_pool_tokens = 0;
        for (int block_m: layout::kCandidateBlockM) {
            num_padded_sf_pool_tokens = std::max(
                num_padded_sf_pool_tokens,
                layout::get_num_padded_sf_pool_tokens(num_max_pool_tokens, block_m));
        }
        return num_padded_sf_pool_tokens;
    }

    BufferViews slice_buffer(const torch::Tensor& buffer) const {
        const auto workspace = layout::SplitWorkspace(
            nullptr, num_ranks(), num_experts_, num_max_tokens_per_rank_, num_topk_);
        const auto fp8_token_layout = layout::Data(hidden_);
        const auto fp8_intermediate_token_layout = layout::Data(intermediate_hidden_);
        const auto fp8_sf_layout = layout::Data(hidden_ / 32);
        const auto fp8_intermediate_sf_layout = layout::Data(intermediate_hidden_ / 32);
        const auto input_topk_idx_layout = layout::Data(num_topk_ * sizeof(int64_t), false);
        const auto input_topk_weights_layout = layout::Data(num_topk_ * sizeof(float), false);
        const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

        const auto input_token_buffer = layout::Buffer(
            fp8_token_layout, 1, num_max_tokens_per_rank_,
            workspace.get_end_ptr());
        const auto input_sf_buffer = layout::Buffer(
            fp8_sf_layout, 1, num_max_tokens_per_rank_,
            input_token_buffer.get_end_ptr());
        const auto input_topk_idx_buffer = layout::Buffer(
            input_topk_idx_layout, 1, num_max_tokens_per_rank_,
            input_sf_buffer.get_end_ptr());
        const auto input_topk_weights_buffer = layout::Buffer(
            input_topk_weights_layout, 1, num_max_tokens_per_rank_,
            input_topk_idx_buffer.get_end_ptr());
        const auto l1_token_buffer = layout::Buffer(
            fp8_token_layout, 1, config_.num_max_pool_tokens,
            input_topk_weights_buffer.get_end_ptr());
        const auto l1_sf_buffer = layout::Buffer(
            fp8_sf_layout, 1, config_.num_padded_sf_pool_tokens,
            l1_token_buffer.get_end_ptr());
        const auto l1_topk_weights_buffer = layout::Buffer(
            l1_topk_weights_layout, 1, config_.num_max_pool_tokens,
            l1_sf_buffer.get_end_ptr());
        const auto l2_token_buffer = layout::Buffer(
            fp8_intermediate_token_layout, 1, config_.num_max_pool_tokens,
            l1_topk_weights_buffer.get_end_ptr());
        const auto l2_sf_buffer = layout::Buffer(
            fp8_intermediate_sf_layout, 1, config_.num_padded_sf_pool_tokens,
            l2_token_buffer.get_end_ptr());

        return {
            torch::from_blob(
                math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l1_token_buffer.base)),
                {config_.num_max_pool_tokens, hidden_},
                torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device())),
            torch::from_blob(
                math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l1_sf_buffer.base)),
                {config_.num_padded_sf_pool_tokens, hidden_ / 128},
                {1, config_.num_padded_sf_pool_tokens},
                torch::TensorOptions().dtype(torch::kInt).device(buffer.device())),
            torch::from_blob(
                math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l2_token_buffer.base)),
                {config_.num_max_pool_tokens, intermediate_hidden_},
                torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device())),
            torch::from_blob(
                math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l2_sf_buffer.base)),
                {config_.num_padded_sf_pool_tokens, intermediate_hidden_ / 128},
                {1, config_.num_padded_sf_pool_tokens},
                torch::TensorOptions().dtype(torch::kInt).device(buffer.device()))
        };
    }

    uint32_t num_ranks() const {
        DG_HOST_ASSERT(not sym_buffer_ptrs_.empty());
        return static_cast<uint32_t>(sym_buffer_ptrs_[0].size());
    }

    void check_inputs() {
        DG_HOST_ASSERT(not states_.empty());
        DG_HOST_ASSERT(states_.size() == ys_.size());
        DG_HOST_ASSERT(states_.size() == sym_buffers_.size());
        DG_HOST_ASSERT(states_.size() == sym_buffer_ptrs_.size());
        DG_HOST_ASSERT(states_.size() == l1_weights_.size());
        DG_HOST_ASSERT(states_.size() == l1_weights_sf_.size());
        DG_HOST_ASSERT(states_.size() == l2_weights_.size());
        DG_HOST_ASSERT(states_.size() == l2_weights_sf_.size());
        DG_HOST_ASSERT(states_.size() == stats_.size());
        DG_HOST_ASSERT(num_experts_ % num_ranks() == 0);
        DG_HOST_ASSERT(num_experts_per_rank_ == num_experts_ / num_ranks());
        DG_HOST_ASSERT(num_tokens_ <= num_max_tokens_per_rank_);
        DG_HOST_ASSERT(hidden_ % 128 == 0 and intermediate_hidden_ % 128 == 0);
        DG_HOST_ASSERT(kernel1_sms_ % 2 == 0 and kernel2_sms_ % 2 == 0);

        for (uint32_t buffer_idx = 0; buffer_idx < states_.size(); ++buffer_idx) {
            check_split_skeleton_tensor(states_[buffer_idx], torch::kInt);
            DG_HOST_ASSERT(states_[buffer_idx].numel() >= 7);
            DG_HOST_ASSERT(ys_[buffer_idx].is_cuda());
            DG_HOST_ASSERT(ys_[buffer_idx].is_contiguous());
            DG_HOST_ASSERT(ys_[buffer_idx].nbytes() % 4 == 0);
            DG_HOST_ASSERT(sym_buffers_[buffer_idx].is_cuda());
            DG_HOST_ASSERT(sym_buffers_[buffer_idx].is_contiguous());
            DG_HOST_ASSERT(l1_weights_[buffer_idx].is_cuda());
            DG_HOST_ASSERT(l1_weights_[buffer_idx].is_contiguous());
            DG_HOST_ASSERT(l1_weights_sf_[buffer_idx].is_cuda());
            DG_HOST_ASSERT(l2_weights_[buffer_idx].is_cuda());
            DG_HOST_ASSERT(l2_weights_[buffer_idx].is_contiguous());
            DG_HOST_ASSERT(l2_weights_sf_[buffer_idx].is_cuda());
            DG_HOST_ASSERT(stats_[buffer_idx].is_cuda());
            DG_HOST_ASSERT(stats_[buffer_idx].is_contiguous());
            DG_HOST_ASSERT(stats_[buffer_idx].scalar_type() == torch::kInt);
            DG_HOST_ASSERT(stats_[buffer_idx].numel() == num_experts_per_rank_);
        }
    }

    void build_kernel_runtimes() {
        const SM100FP8FP4MegaMoESplitDispatchL1SwigluRuntime::Args kernel1_args = {
            .num_max_tokens_per_rank = static_cast<int>(num_max_tokens_per_rank_),
            .hidden = static_cast<int>(hidden_),
            .intermediate_hidden = static_cast<int>(intermediate_hidden_),
            .num_experts = static_cast<int>(num_experts_),
            .num_topk = static_cast<int>(num_topk_),
            .num_ranks = static_cast<int>(num_ranks()),
            .num_sms = static_cast<int>(kernel1_sms_),
            .activation_clamp = activation_clamp_,
            .fast_math = fast_math_,
            .local_only = false,
            .config = config_,
            .cumulative_local_expert_recv_stats = nullptr,
            .num_tokens = static_cast<int>(num_tokens_),
            .sym_buffer_ptrs = layout::SymBuffer<>(),
            .tensor_map_l1_acts = {},
            .tensor_map_l1_acts_sf = {},
            .tensor_map_l1_weights = {},
            .tensor_map_l1_weights_sf = {},
            .tensor_map_l1_output = {},
            .launch_args = LaunchArgs(static_cast<int>(kernel1_sms_),
                                      config_.num_dispatch_threads + config_.num_non_epilogue_threads + config_.num_epilogue_threads,
                                      config_.smem_size, 2)
        };
        const SM100FP8FP4MegaMoESplitL2CombineRuntime::Args real_kernel2_args = {
            .num_max_tokens_per_rank = static_cast<int>(num_max_tokens_per_rank_),
            .hidden = static_cast<int>(hidden_),
            .intermediate_hidden = static_cast<int>(intermediate_hidden_),
            .num_experts = static_cast<int>(num_experts_),
            .num_topk = static_cast<int>(num_topk_),
            .num_ranks = static_cast<int>(num_ranks()),
            .kernel1_sms = static_cast<int>(kernel1_sms_),
            .kernel2_sms = static_cast<int>(kernel2_sms_),
            .config = kernel2_config_,
            .state = nullptr,
            .num_work_iters = kernel2_work_iters_,
            .sym_buffer_ptrs = layout::SymBuffer<>(),
            .tensor_map_l2_acts = {},
            .tensor_map_l2_acts_sf = {},
            .tensor_map_l2_weights = {},
            .tensor_map_l2_weights_sf = {},
            .launch_args = LaunchArgs(
                static_cast<int>(kernel2_sms_),
                kernel2_config_.num_non_epilogue_threads + kernel2_config_.num_epilogue_threads,
                kernel2_config_.smem_size, 2)
        };
        const SM100FP8FP4MegaMoESplitCombineReduceRuntime::Args kernel3_args = {
            .num_max_tokens_per_rank = static_cast<int>(num_max_tokens_per_rank_),
            .hidden = static_cast<int>(hidden_),
            .intermediate_hidden = static_cast<int>(intermediate_hidden_),
            .num_experts = static_cast<int>(num_experts_),
            .num_topk = static_cast<int>(num_topk_),
            .num_ranks = static_cast<int>(num_ranks()),
            .config = config_,
            .y = nullptr,
            .state = nullptr,
            .num_tokens = static_cast<uint32_t>(num_tokens_),
            .sym_buffer_ptrs = layout::SymBuffer<>(),
            .launch_args = LaunchArgs(static_cast<int>(num_tokens_), 512, 0, 1, false)
        };

        print_mega_moe_split_kernel1_config(kernel1_args);
        const auto kernel1_code = SM100FP8FP4MegaMoESplitDispatchL1SwigluRuntime::generate(kernel1_args);
        const auto kernel2_code = SM100FP8FP4MegaMoESplitL2CombineRuntime::generate(real_kernel2_args);
        const auto kernel3_code = SM100FP8FP4MegaMoESplitCombineReduceRuntime::generate(kernel3_args);
        kernel1_runtime_ = compiler->build("sm100_fp8_fp4_mega_moe_split_dispatch_l1_swiglu", kernel1_code);
        kernel2_runtime_ = compiler->build("sm100_fp8_fp4_mega_moe_split_l2_combine", kernel2_code);
        kernel3_runtime_ = compiler->build("sm100_fp8_fp4_mega_moe_split_combine_reduce", kernel3_code);

        kernel1_graph_kernel_ = kernel1_runtime_->kernel;
        kernel2_graph_kernel_ = kernel2_runtime_->kernel;
        kernel3_graph_kernel_ = kernel3_runtime_->kernel;
    }

    void create_green_contexts() {
        DG_CUDA_RUNTIME_CHECK(cudaGetDevice(&device_idx_));
        DG_CUDA_RUNTIME_CHECK(cudaSetDevice(device_idx_));
        DG_CUDA_RUNTIME_CHECK(lazy_cudaDeviceGetExecutionCtx(&primary_context_, device_idx_));
        DG_HOST_ASSERT(primary_context_ != nullptr);

        cudaDevResource sm_resource = {};
        DG_CUDA_RUNTIME_CHECK(lazy_cudaDeviceGetDevResource(device_idx_, &sm_resource, cudaDevResourceTypeSm));
        DG_HOST_ASSERT(sm_resource.type == cudaDevResourceTypeSm);
        DG_HOST_ASSERT(kernel1_sms_ + kernel2_sms_ <= sm_resource.sm.smCount);

        cudaDevResource workqueue_resource = {};
        DG_CUDA_RUNTIME_CHECK(lazy_cudaDeviceGetDevResource(
            device_idx_, &workqueue_resource, cudaDevResourceTypeWorkqueueConfig));
        DG_HOST_ASSERT(workqueue_resource.type == cudaDevResourceTypeWorkqueueConfig);
        workqueue_resource.wqConfig.sharingScope = cudaDevWorkqueueConfigScopeGreenCtxBalanced;

        std::array<cudaDevResource, 2> split_resources = {};
        cudaDevResource remainder = {};
        std::array<cudaDevSmResourceGroupParams, 2> group_params = {};
        group_params[0].smCount = kernel1_sms_;
        group_params[0].coscheduledSmCount = 2;
        group_params[1].smCount = kernel2_sms_;
        group_params[1].coscheduledSmCount = 2;
        DG_CUDA_RUNTIME_CHECK(lazy_cudaDevSmResourceSplit(
            split_resources.data(), static_cast<unsigned int>(split_resources.size()),
            &sm_resource, &remainder, 0, group_params.data()));

        for (uint32_t context_idx = 0; context_idx < split_resources.size(); ++context_idx) {
            DG_HOST_ASSERT(split_resources[context_idx].type == cudaDevResourceTypeSm);
            std::array<cudaDevResource, 2> context_resources = {
                split_resources[context_idx],
                workqueue_resource
            };
            cudaDevResourceDesc_t resource_desc = nullptr;
            DG_CUDA_RUNTIME_CHECK(lazy_cudaDevResourceGenerateDesc(
                &resource_desc, context_resources.data(), context_resources.size()));
            DG_CUDA_RUNTIME_CHECK(lazy_cudaGreenCtxCreate(
                &green_contexts_[context_idx], resource_desc, device_idx_, 0));
            DG_CUDA_RUNTIME_CHECK(lazy_cudaExecutionCtxGetId(
                green_contexts_[context_idx], &green_context_ids_[context_idx]));
        }
    }

    void prepare_kernel_params() {
        const auto num_buffers = states_.size();
        kernel1_args_.resize(num_buffers);
        kernel2_args_.resize(num_buffers);
        kernel3_args_.resize(num_buffers);
        kernel1_params_.resize(num_buffers);
        kernel2_params_.resize(num_buffers);
        kernel3_params_.resize(num_buffers);

        constexpr int kGranK = 32;
        const int sf_smem_outer_dim = config_.block_k / (kGranK * 4);
        for (uint32_t buffer_idx = 0; buffer_idx < num_buffers; ++buffer_idx) {
            const auto views = slice_buffer(sym_buffers_[buffer_idx]);
            kernel1_args_[buffer_idx].cumulative_local_expert_recv_stats = stats_[buffer_idx].data_ptr<int>();
            kernel1_args_[buffer_idx].num_tokens = num_tokens_;
            kernel1_args_[buffer_idx].sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs_[buffer_idx], rank_idx_);
            kernel1_args_[buffer_idx].tensor_map_l1_acts = make_tma_2d_desc(
                views.l1_acts,
                hidden_, config_.num_max_pool_tokens,
                config_.block_k, config_.load_block_m,
                static_cast<int>(views.l1_acts.stride(-2)),
                config_.swizzle_acts_mode);
            kernel1_args_[buffer_idx].tensor_map_l1_acts_sf = make_tma_sf_desc(
                cute::UMMA::Major::MN, views.l1_acts_sf,
                config_.num_padded_sf_pool_tokens, hidden_,
                config_.sf_block_m, kGranK,
                1, 0, 0, false,
                sf_smem_outer_dim);
            kernel1_args_[buffer_idx].tensor_map_l1_weights = make_tma_2d_desc(
                l1_weights_[buffer_idx],
                hidden_, num_experts_per_rank_ * intermediate_hidden_ * 2,
                config_.block_k, config_.load_block_n,
                static_cast<int>(l1_weights_[buffer_idx].stride(-2)),
                config_.swizzle_weights_mode);
            kernel1_args_[buffer_idx].tensor_map_l1_weights_sf = make_tma_sf_desc(
                cute::UMMA::Major::MN, l1_weights_sf_[buffer_idx],
                intermediate_hidden_ * 2, hidden_,
                config_.block_n, kGranK,
                num_experts_per_rank_, 0, 0, false,
                sf_smem_outer_dim);
            kernel1_args_[buffer_idx].tensor_map_l1_output = make_tma_2d_desc(
                views.l2_acts,
                intermediate_hidden_, config_.num_max_pool_tokens,
                config_.block_n / 2, config_.store_block_m,
                static_cast<int>(views.l2_acts.stride(-2)),
                config_.swizzle_acts_mode / 2);

            kernel2_args_[buffer_idx].state = states_[buffer_idx].data_ptr();
            kernel2_args_[buffer_idx].num_work_iters = kernel2_work_iters_;
            kernel2_args_[buffer_idx].sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs_[buffer_idx], rank_idx_);
            kernel2_args_[buffer_idx].tensor_map_l2_acts = make_tma_2d_desc(
                views.l2_acts,
                intermediate_hidden_, config_.num_max_pool_tokens,
                config_.block_k, config_.load_block_m,
                static_cast<int>(views.l2_acts.stride(-2)),
                config_.swizzle_acts_mode);
            kernel2_args_[buffer_idx].tensor_map_l2_acts_sf = make_tma_sf_desc(
                cute::UMMA::Major::MN, views.l2_acts_sf,
                config_.num_padded_sf_pool_tokens, intermediate_hidden_,
                config_.sf_block_m, kGranK,
                1, 0, 0, false,
                sf_smem_outer_dim);
            kernel2_args_[buffer_idx].tensor_map_l2_weights = make_tma_2d_desc(
                l2_weights_[buffer_idx],
                intermediate_hidden_, num_experts_per_rank_ * hidden_,
                config_.block_k, config_.load_block_n,
                static_cast<int>(l2_weights_[buffer_idx].stride(-2)),
                config_.swizzle_weights_mode);
            kernel2_args_[buffer_idx].tensor_map_l2_weights_sf = make_tma_sf_desc(
                cute::UMMA::Major::MN, l2_weights_sf_[buffer_idx],
                hidden_, intermediate_hidden_,
                config_.block_n, kGranK,
                num_experts_per_rank_, 0, 0, false,
                sf_smem_outer_dim);
            kernel3_args_[buffer_idx] = Kernel3NodeArgs(
                ys_[buffer_idx].data_ptr(),
                states_[buffer_idx].data_ptr(),
                num_tokens_,
                layout::SymBuffer<>(sym_buffer_ptrs_[buffer_idx], rank_idx_));

            kernel1_params_[buffer_idx] = {
                &kernel1_args_[buffer_idx].cumulative_local_expert_recv_stats,
                &kernel1_args_[buffer_idx].num_tokens,
                &kernel1_args_[buffer_idx].sym_buffer_ptrs,
                &kernel1_args_[buffer_idx].tensor_map_l1_acts,
                &kernel1_args_[buffer_idx].tensor_map_l1_acts_sf,
                &kernel1_args_[buffer_idx].tensor_map_l1_weights,
                &kernel1_args_[buffer_idx].tensor_map_l1_weights_sf,
                &kernel1_args_[buffer_idx].tensor_map_l1_output
            };
            kernel2_params_[buffer_idx] = {
                &kernel2_args_[buffer_idx].state,
                &kernel2_args_[buffer_idx].num_work_iters,
                &kernel2_args_[buffer_idx].sym_buffer_ptrs,
                &kernel2_args_[buffer_idx].tensor_map_l2_acts,
                &kernel2_args_[buffer_idx].tensor_map_l2_acts_sf,
                &kernel2_args_[buffer_idx].tensor_map_l2_weights,
                &kernel2_args_[buffer_idx].tensor_map_l2_weights_sf
            };
            kernel3_params_[buffer_idx] = {
                &kernel3_args_[buffer_idx].y,
                &kernel3_args_[buffer_idx].state,
                &kernel3_args_[buffer_idx].num_tokens,
                &kernel3_args_[buffer_idx].sym_buffer_ptrs
            };
        }
    }

    cudaGraphNode_t add_kernel_node(
        const cudaExecutionContext_t& context,
        const KernelHandle& kernel,
        const uint32_t& num_blocks,
        const uint32_t& block_dim,
        const uint32_t& shared_mem_bytes,
        const uint32_t& cluster_dim,
        void** kernel_params,
        const std::vector<cudaGraphNode_t>& dependencies
    ) {
        if (shared_mem_bytes > 0) {
        #if defined(DG_JIT_USE_RUNTIME_API)
            DG_CUDA_RUNTIME_CHECK(cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_bytes));
        #else
            DG_CUDA_DRIVER_CHECK(lazy_cuFuncSetAttribute(
                kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, static_cast<int>(shared_mem_bytes)));
        #endif
        }

        cudaGraphNode_t node = nullptr;
        cudaGraphNodeParams node_params = {};
        node_params.type = cudaGraphNodeTypeKernel;
    #if defined(DG_JIT_USE_RUNTIME_API)
        node_params.kernel.kern = kernel;
        node_params.kernel.functionType = cudaKernelFunctionTypeKernel;
    #else
        node_params.kernel.cuFunc = kernel;
        node_params.kernel.functionType = cudaKernelFunctionTypeFunction;
    #endif
        node_params.kernel.gridDim = dim3(num_blocks, 1, 1);
        node_params.kernel.blockDim = dim3(block_dim, 1, 1);
        node_params.kernel.sharedMemBytes = shared_mem_bytes;
        node_params.kernel.kernelParams = kernel_params;
        node_params.kernel.extra = nullptr;
        node_params.kernel.ctx = context;

        const auto dependency_ptr = dependencies.empty() ? nullptr : dependencies.data();
        DG_CUDA_RUNTIME_CHECK(lazy_cudaGraphAddNode(
            &node, graph_, dependency_ptr, nullptr, dependencies.size(), &node_params));

        if (cluster_dim > 1) {
            cudaKernelNodeAttrValue attr = {};
            attr.clusterDim = {cluster_dim, 1, 1};
            DG_CUDA_RUNTIME_CHECK(lazy_cudaGraphKernelNodeSetAttribute(
                node, cudaKernelNodeAttributeClusterDimension, &attr));
        }
        return node;
    }

    void build_graph() {
        prepare_kernel_params();
        DG_CUDA_RUNTIME_CHECK(cudaGraphCreate(&graph_, 0));

        const auto num_buffers = states_.size();
        kernel1_nodes_.reserve(num_buffers);
        kernel2_nodes_.reserve(num_buffers);
        reduce_nodes_.reserve(num_buffers);

        const uint32_t kernel1_block_dim = static_cast<uint32_t>(
            config_.num_dispatch_threads + config_.num_non_epilogue_threads + config_.num_epilogue_threads);
        for (uint32_t buffer_idx = 0; buffer_idx < num_buffers; ++buffer_idx) {
            std::vector<cudaGraphNode_t> dependencies;
            if (not kernel1_nodes_.empty())
                dependencies.push_back(kernel1_nodes_.back());
            kernel1_nodes_.push_back(add_kernel_node(
                green_contexts_[0], kernel1_graph_kernel_, kernel1_sms_,
                kernel1_block_dim, static_cast<uint32_t>(config_.smem_size), 2,
                kernel1_params_[buffer_idx].data(), dependencies));
        }

        for (uint32_t buffer_idx = 0; buffer_idx < num_buffers; ++buffer_idx) {
            std::vector<cudaGraphNode_t> dependencies;
            if (not kernel2_nodes_.empty())
                dependencies.push_back(kernel2_nodes_.back());
            const uint32_t kernel2_block_dim = static_cast<uint32_t>(kernel2_config_.num_non_epilogue_threads + kernel2_config_.num_epilogue_threads);
            const uint32_t kernel2_shared_mem_bytes = static_cast<uint32_t>(kernel2_config_.smem_size);
            const uint32_t kernel2_cluster_dim = 2u;
            kernel2_nodes_.push_back(add_kernel_node(
                green_contexts_[1], kernel2_graph_kernel_, kernel2_sms_,
                kernel2_block_dim, kernel2_shared_mem_bytes, kernel2_cluster_dim,
                kernel2_params_[buffer_idx].data(), dependencies));
        }

        for (uint32_t buffer_idx = 0; buffer_idx < num_buffers; ++buffer_idx) {
            std::vector<cudaGraphNode_t> dependencies;
            if (reduce_nodes_.empty()) {
                dependencies.push_back(kernel1_nodes_.back());
                dependencies.push_back(kernel2_nodes_.back());
            } else {
                dependencies.push_back(reduce_nodes_.back());
            }
            reduce_nodes_.push_back(add_kernel_node(
                primary_context_, kernel3_graph_kernel_, num_tokens_,
                512, 0, 1,
                kernel3_params_[buffer_idx].data(), dependencies));
        }

        DG_CUDA_RUNTIME_CHECK(lazy_cudaGraphInstantiate(&graph_exec_, graph_, 0));
    }

    void destroy_noexcept() noexcept {
        if (graph_exec_ != nullptr) {
            (void) lazy_cudaGraphExecDestroy(graph_exec_);
            graph_exec_ = nullptr;
        }
        if (graph_ != nullptr) {
            (void) lazy_cudaGraphDestroy(graph_);
            graph_ = nullptr;
        }
        for (uint32_t context_idx = 0; context_idx < green_contexts_.size(); ++context_idx) {
            if (green_contexts_[context_idx] != nullptr) {
                (void) lazy_cudaExecutionCtxDestroy(green_contexts_[context_idx]);
                green_contexts_[context_idx] = nullptr;
            }
        }
    }

public:
    SM100FP8FP4MegaMoESplitGraph(
        std::vector<torch::Tensor> states,
        std::vector<torch::Tensor> ys,
        std::vector<torch::Tensor> sym_buffers,
        std::vector<std::vector<int64_t>> sym_buffer_ptrs,
        std::vector<torch::Tensor> l1_weights,
        std::vector<torch::Tensor> l1_weights_sf,
        std::vector<torch::Tensor> l2_weights,
        std::vector<torch::Tensor> l2_weights_sf,
        std::vector<torch::Tensor> stats,
        const int& rank_idx,
        const int& num_max_tokens_per_rank,
        const int& num_experts,
        const int& num_topk,
        const int& num_tokens,
        const int& hidden,
        const int& intermediate_hidden,
        const float& activation_clamp,
        const bool& fast_math,
        const int& kernel1_sms,
        const int& kernel2_sms,
        const int& reduce_sms,
        const int& kernel2_work_iters,
        const int& reduce_work_iters
    ) : states_(std::move(states)),
        ys_(std::move(ys)),
        sym_buffers_(std::move(sym_buffers)),
        sym_buffer_ptrs_(std::move(sym_buffer_ptrs)),
        l1_weights_(std::move(l1_weights)),
        l1_weights_sf_(std::move(l1_weights_sf)),
        l2_weights_(std::move(l2_weights)),
        l2_weights_sf_(std::move(l2_weights_sf)),
        stats_(std::move(stats)),
        rank_idx_(checked_nonnegative_u32(rank_idx)),
        num_max_tokens_per_rank_(checked_positive_u32(num_max_tokens_per_rank)),
        num_experts_(checked_positive_u32(num_experts)),
        num_experts_per_rank_(0),
        num_topk_(checked_positive_u32(num_topk)),
        num_tokens_(checked_nonnegative_u32(num_tokens)),
        hidden_(checked_positive_u32(hidden)),
        intermediate_hidden_(checked_positive_u32(intermediate_hidden)),
        kernel1_sms_(checked_positive_u32(kernel1_sms)),
        kernel2_sms_(checked_positive_u32(kernel2_sms)),
        reduce_sms_(checked_positive_u32(reduce_sms)),
        kernel2_work_iters_(checked_nonnegative_u32(kernel2_work_iters)),
        reduce_work_iters_(checked_nonnegative_u32(reduce_work_iters)),
        activation_clamp_(activation_clamp),
        fast_math_(fast_math) {
        DG_HOST_ASSERT(not sym_buffer_ptrs_.empty());
        num_experts_per_rank_ = num_experts_ / num_ranks();
        const int num_max_pool_tokens = layout::get_num_max_pool_tokens(
            static_cast<int>(num_ranks()), static_cast<int>(num_max_tokens_per_rank_),
            static_cast<int>(num_topk_), static_cast<int>(num_experts_per_rank_));
        const int num_padded_sf_pool_tokens = get_num_padded_sf_pool_tokens(num_max_pool_tokens);
        config_ = get_mega_moe_split_kernel1_config(
            static_cast<int>(num_ranks()), static_cast<int>(num_experts_), static_cast<int>(num_experts_per_rank_),
            static_cast<int>(num_max_tokens_per_rank_), static_cast<int>(num_tokens_), static_cast<int>(num_topk_),
            static_cast<int>(hidden_), static_cast<int>(intermediate_hidden_),
            num_padded_sf_pool_tokens, static_cast<int>(kernel1_sms_));
        kernel2_config_ = get_mega_moe_split_kernel2_config(config_);
        check_inputs();
        build_kernel_runtimes();
        create_green_contexts();
        build_graph();
    }

    SM100FP8FP4MegaMoESplitGraph(const SM100FP8FP4MegaMoESplitGraph&) = delete;
    SM100FP8FP4MegaMoESplitGraph& operator=(const SM100FP8FP4MegaMoESplitGraph&) = delete;

    ~SM100FP8FP4MegaMoESplitGraph() {
        destroy_noexcept();
    }

    void replay() {
        DG_HOST_ASSERT(graph_exec_ != nullptr);
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        DG_CUDA_RUNTIME_CHECK(lazy_cudaGraphLaunch(graph_exec_, stream));
    }

    std::tuple<unsigned long long, unsigned long long> get_green_context_ids() const {
        return {green_context_ids_[0], green_context_ids_[1]};
    }
};

#else

class SM100FP8FP4MegaMoESplitGraph final {
public:
    SM100FP8FP4MegaMoESplitGraph(
        std::vector<torch::Tensor>,
        std::vector<torch::Tensor>,
        std::vector<torch::Tensor>,
        std::vector<std::vector<int64_t>>,
        std::vector<torch::Tensor>,
        std::vector<torch::Tensor>,
        std::vector<torch::Tensor>,
        std::vector<torch::Tensor>,
        std::vector<torch::Tensor>,
        const int&,
        const int&,
        const int&,
        const int&,
        const int&,
        const int&,
        const int&,
        const float&,
        const bool&,
        const int&,
        const int&,
        const int&,
        const int&,
        const int&
    ) {
        DG_HOST_UNREACHABLE(
            "Native real K1 fake K2 graph requires CUDA Runtime 13.1+");
    }

    void replay() {
        DG_HOST_UNREACHABLE(
            "Native real K1 fake K2 graph requires CUDA Runtime 13.1+");
    }

    std::tuple<unsigned long long, unsigned long long> get_green_context_ids() const {
        DG_HOST_UNREACHABLE(
            "Native real K1 fake K2 graph requires CUDA Runtime 13.1+");
    }
};

#endif

} // namespace deep_gemm
