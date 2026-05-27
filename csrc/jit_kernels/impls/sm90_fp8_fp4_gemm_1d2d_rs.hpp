#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/layout.hpp"
#include "epilogue.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

// JIT runtime for the RS-mode SM90 FP8xFP4 1D2D kernel.
//
// S2 stage: identical wire-format to `SM90FP8FP4Gemm1D2DRuntime`, only the
// included header and instantiated kernel name differ. This gives the new
// kernel its own JIT artifact and entry point so subsequent steps (S3+) can
// safely diverge the device code without affecting the SS path or the
// contiguous host route.
class SM90FP8FP4Gemm1D2DRSRuntime final: public LaunchRuntime<SM90FP8FP4Gemm1D2DRSRuntime> {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        LaunchArgs launch_args;

        cute::UMMA::Major major_sfb;
        bool scale_b_direct_load;
        bool scale_b_pow2_promote;
        bool k32_quad_reduce;
        bool k32_quad_split_promote;
        bool k32_quad_scale_b_inline;
        bool k32_quad_scale_b_prefetch;
        bool k32_quad_scale_b_vec4;
        bool k32_quad_pair4x2_promote;
        bool small_m_simple_sched;
        bool compact_masked_sched;
        // Fused-decode：在 cache_sfb_k32 阶段把 e8m0 scale 编进 LUT，wgmma 后省一次 fmul。
        // 与 scale_b_direct_load / k32_quad_reduce / e8m0/bf16/pow2/compact_sched 互斥
        // （host 决策侧保证），路径走 path-B 通用 cache_sfb_k32 + LUT decode。
        bool fuse_scale_b_decode;
        // 配合 fuse_scale_b_decode：sfb 物理布局是 [groups, K/32/4, N]（MN-major + 4 个
        // e8m0 打包成 1 个 int32），体积 = fp32 的 1/4。需 sfb.scalar_type() == kInt。
        bool scale_b_packed_ue8m0;
        uint32_t scale_b_gran_k;
        // INT4-sym (signed [-8, 7] packed two nibbles/byte) variant for B.
        // Path-A: per-128 fp32 SFB, no fused-decode. See kernel header.
        bool b_is_int4_sym;
        // bf16 SFB（path-A k128 / path-B fast-path）：体积砍半，scale 用 __nv_bfloat16。
        bool scale_b_bf16;
        // E8M0 SFB（path-B fast-path 专用）：每元素 1B = fp32 的 8 位指数。
        // 解码 `__uint_as_float(uint32(e) << 23)` 零误差。仅 pow2 scale 适用。
        bool scale_b_e8m0;
        // compact_masked_sched 按 m_max 降序遍历 active group（实验性扩展）：
        // 仅在 compact_masked_sched 开启时生效，让 wave 0 优先吃重 group，
        // 减少 last-wave imbalance。host 默认关。
        bool reorder_masked_by_max_m;
        void *gmem_b_ptr;
        void *gmem_d_ptr;
        void *sfb;
        void *grouped_layout;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_d;
        CUtensorMap tensor_map_sfa;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_fp8_fp4_gemm_1d2d_rs.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_fp4_gemm_1d2d_rs_impl<
        {},
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {},
        {}, {},
        {}, {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}
    >);
}};
)",
        to_string(args.major_sfb),
        get_compiled_dim(args.gemm_desc.m, 'm', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.k, 'k', args.gemm_desc.compiled_dims),
        args.gemm_desc.num_groups,
        args.gemm_config.layout.block_m,
        args.gemm_config.layout.block_n,
        args.gemm_config.layout.block_k,
        args.gemm_config.storage_config.swizzle_a_mode,
        args.gemm_config.storage_config.swizzle_b_mode,
        args.gemm_config.storage_config.swizzle_cd_mode,
        args.gemm_config.pipeline_config.num_stages,
        args.gemm_config.launch_config.num_tma_threads,
        args.gemm_config.launch_config.num_math_threads,
        args.gemm_config.layout.get_cluster_size(),
        args.gemm_config.layout.cluster_n > 1,
        args.gemm_config.launch_config.num_sms,
        to_string(args.gemm_desc.gemm_type),
        get_default_epilogue_type(std::nullopt),
        "false",
        "false",
        "false",
        "false",
        "false",
        args.scale_b_pow2_promote ? "true" : "false",
        "false",
        "false",
        args.scale_b_direct_load ? "true" : "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        args.k32_quad_reduce ? "true" : "false",
        "false",
        args.k32_quad_reduce ? "true" : "false",
        args.k32_quad_split_promote ? "true" : "false",
        args.k32_quad_scale_b_inline ? "true" : "false",
        args.k32_quad_scale_b_prefetch ? "true" : "false",
        args.k32_quad_scale_b_vec4 ? "true" : "false",
        args.k32_quad_pair4x2_promote ? "true" : "false",
        "false",
        "false",
        "false",
        "false",
        args.small_m_simple_sched ? "true" : "false",
        args.compact_masked_sched ? "true" : "false",
        args.fuse_scale_b_decode ? "true" : "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        0,
        args.scale_b_packed_ue8m0 ? "true" : "false",
        args.scale_b_gran_k,
        1,
        1,
        0,
        args.b_is_int4_sym ? "true" : "false",
        args.scale_b_bf16 ? "true" : "false",
        args.scale_b_e8m0 ? "true" : "false",
        args.reorder_masked_by_max_m ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.gmem_b_ptr, args.sfb, args.grouped_layout, args.gmem_d_ptr,
            args.gemm_desc.m, args.gemm_desc.n, args.gemm_desc.k,
            args.tensor_map_a, args.tensor_map_b, args.tensor_map_d, args.tensor_map_sfa));
    }
};

} // namespace deep_gemm
