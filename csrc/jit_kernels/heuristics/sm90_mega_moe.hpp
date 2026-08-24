#pragma once

#include <algorithm>
#include <cstdint>

#include "../../utils/exception.hpp"

#include <deep_gemm/common/types.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sm90_mega_moe.cuh>

#include "../../utils/math.hpp"
#include "sm90.hpp"

namespace deep_gemm {

// The retained SM90 path is the compact Humming FP8 x MXFP4 persistent kernel.
// Its block/thread schedule is fixed so the host sizing logic and the kernel's
// two-CTA occupancy contract cannot drift independently.
struct MegaMoESM90Config {
    using Schedule = layout::Sm90HummingMoeSchedule;
    static constexpr int block_m = static_cast<int>(Schedule::block_m);
    static constexpr int block_n = static_cast<int>(Schedule::block_n);
    static constexpr int block_k = static_cast<int>(Schedule::block_k);
    static constexpr int num_stages = static_cast<int>(Schedule::num_stages);
    static constexpr int num_dispatch_threads =
        static_cast<int>(Schedule::num_dispatch_threads);
    static constexpr int num_non_epilogue_threads =
        static_cast<int>(Schedule::num_non_epilogue_threads);
    static constexpr int num_epilogue_threads =
        static_cast<int>(Schedule::num_epilogue_threads);

    int num_sms;
    int smem_size;
};

constexpr int kSM90MoeMaxLatencyOverlapTokens = 4096;

struct Sm90MoeHeuristicInput {
    int launch_num_sms;

    int num_experts;
    int num_tokens;
    int hidden, intermediate_hidden;
};

static int get_mxfp4_pipeline_smem_size_for_mega_moe_sm90(
    const int smem_capacity,
    const int num_experts,
    const int hidden,
    const bool double_buffer_mxfp4_expanded_b) {
    constexpr int kSmemAlignment = 1024;
    constexpr int block_m = MegaMoESM90Config::block_m;
    constexpr int block_n = MegaMoESM90Config::block_n;
    constexpr int block_k = MegaMoESM90Config::block_k;
    constexpr int num_stages = MegaMoESM90Config::num_stages;
    constexpr int num_dispatch_warps =
        MegaMoESM90Config::num_dispatch_threads / 32;
    constexpr int num_epilogue_warps =
        MegaMoESM90Config::num_epilogue_threads / 32;

    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(
            layout::Data(hidden), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const int smem_dispatch_size =
        smem_expert_count_size + smem_send_buffers_size;

    constexpr int smem_cd_l1 =
        block_m * (block_n / 2);
    constexpr int smem_cd_l2 =
        block_m * block_n * static_cast<int>(sizeof(nv_bfloat16));
    const int smem_cd = align(std::max(smem_cd_l1, smem_cd_l2),
                              kSmemAlignment);

    constexpr int smem_sfa_half_stride_bytes =
        ((block_m * static_cast<int>(sizeof(float)) + 127) / 128) * 128;
    constexpr int smem_sfa_per_stage =
        (block_k / 64) * smem_sfa_half_stride_bytes;
    constexpr int smem_sfb_per_stage = block_n * (block_k / 32);
    constexpr int smem_packed_b_per_stage = block_n * block_k / 2;
    constexpr int smem_a_per_stage = block_m * block_k;
    constexpr int smem_barriers_per_stage = 2 * 8;
    constexpr int smem_per_stage =
        smem_a_per_stage + smem_packed_b_per_stage +
        smem_sfa_per_stage + smem_sfb_per_stage +
        smem_barriers_per_stage;

    const int smem_expanded_b_scratch =
        block_n * block_k * (double_buffer_mxfp4_expanded_b ? 2 : 1);
    constexpr int smem_barriers_fixed =
        (num_dispatch_warps + 2 * num_epilogue_warps) * 8;
    const int smem_fixed = smem_dispatch_size + smem_cd +
                           smem_expanded_b_scratch + smem_barriers_fixed;
    const int smem_size = smem_fixed + num_stages * smem_per_stage;
    return smem_size <= smem_capacity ? smem_size : 0;
}

// Compact Hopper MXFP4 schedule. One math warpgroup owns fixed expanded-B
// scratch. Flash reserves a second ping-pong tile for both latency and
// throughput workloads so the next packed-B stage can be decoded while the
// current WGMMA group is in flight. A, packed-B, SFA, and SFB use a three-stage
// producer pipeline. Two logical worker CTAs are launched per physical H20 SM;
// the exact-kernel occupancy check at launch is the hard safety gate for grid
// barriers.
static MegaMoESM90Config select_mxfp4_mega_moe_sm90(
    const Sm90MoeHeuristicInput& input) {
    constexpr int block_m = MegaMoESM90Config::block_m;
    constexpr int block_n = MegaMoESM90Config::block_n;
    constexpr int block_k = MegaMoESM90Config::block_k;

    DG_HOST_ASSERT((2 * input.intermediate_hidden) % block_n == 0 and
                   input.hidden % block_n == 0);
    DG_HOST_ASSERT(input.hidden % block_k == 0 and
                   input.intermediate_hidden % block_k == 0);

    const int num_worker_ctas = 2 * input.launch_num_sms;
    const bool double_buffer_mxfp4_expanded_b =
        input.hidden == 4096;
    const int smem_size = get_mxfp4_pipeline_smem_size_for_mega_moe_sm90(
        SM90ArchSpec::smem_capacity,
        input.num_experts,
        input.hidden,
        double_buffer_mxfp4_expanded_b);
    // `smem_capacity` is the opt-in per-block limit. Reserve the remaining
    // 1 KiB/CTA implementation overhead in the two-CTA static precheck; the
    // exact JIT kernel still goes through the runtime occupancy hard gate.
    DG_HOST_ASSERT(smem_size > 0 and
                   2 * smem_size <= SM90ArchSpec::smem_capacity - 1024);
    return {
        num_worker_ctas,
        smem_size,
    };
}

} // namespace deep_gemm
