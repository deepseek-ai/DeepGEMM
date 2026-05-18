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
        bool k32_quad_reduce;
        bool small_m_simple_sched;
        uint32_t scale_b_gran_k;
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
        "false",
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
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        "false",
        args.small_m_simple_sched ? "true" : "false",
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
        0,
        "false",
        args.scale_b_gran_k,
        1,
        1,
        0);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.gmem_b_ptr, args.sfb, args.grouped_layout, args.gmem_d_ptr,
            args.gemm_desc.m, args.gemm_desc.n, args.gemm_desc.k,
            args.tensor_map_a, args.tensor_map_b, args.tensor_map_d, args.tensor_map_sfa));
    }
};

} // namespace deep_gemm
