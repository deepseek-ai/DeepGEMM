#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM120FP8BhrHdrBhdScalarRuntime final: public LaunchRuntime<SM120FP8BhrHdrBhdScalarRuntime> {
public:
    struct Args {
        int shape_b;
        int shape_h;
        int shape_d;
        int shape_r;

        int64_t a_stride_b;
        int64_t a_stride_h;
        int64_t a_stride_r;
        int64_t sfa_stride_b;
        int64_t sfa_stride_h;
        int64_t sfa_stride_r;
        int64_t b_stride_h;
        int64_t b_stride_d;
        int64_t b_stride_r;
        int64_t sfb_stride_h;
        int64_t sfb_stride_d;
        int64_t sfb_stride_r;
        int64_t d_stride_b;
        int64_t d_stride_h;
        int64_t d_stride_d;

        void* a;
        float* sfa;
        void* b;
        float* sfb;
        void* d;
        at::ScalarType output_dtype;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        constexpr int output_tile_d = 128;
        return fmt::format(R"(
#include <deep_gemm/impls/sm120_fp8_einsum.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_fp8_bhr_hdr_bhd_scalar<
        {}, {}
    >);
}};
)", to_string(args.output_dtype), output_tile_d);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.shape_b,
            args.shape_h,
            args.shape_d,
            args.shape_r,
            args.a_stride_b,
            args.a_stride_h,
            args.a_stride_r,
            args.sfa_stride_b,
            args.sfa_stride_h,
            args.sfa_stride_r,
            args.b_stride_h,
            args.b_stride_d,
            args.b_stride_r,
            args.sfb_stride_h,
            args.sfb_stride_d,
            args.sfb_stride_r,
            args.d_stride_b,
            args.d_stride_h,
            args.d_stride_d,
            args.a,
            args.sfa,
            args.b,
            args.sfb,
            args.d
        ));
    }
};

static void sm120_fp8_bhr_hdr_bhd_scalar(const torch::Tensor& a,
                                            const torch::Tensor& sfa,
                                            const torch::Tensor& b,
                                            const torch::Tensor& sfb,
                                            const torch::Tensor& d) {
    constexpr int gran_mn = 128;
    constexpr int gran_k = 128;
    constexpr int output_tile_d = 128;

    const auto [shape_b, shape_h, shape_r] = get_shape<3>(a);
    const auto [shape_h_, shape_d, shape_r_] = get_shape<3>(b);
    const auto [shape_b_, shape_h__, shape_d_] = get_shape<3>(d);

    DG_HOST_ASSERT(shape_b == shape_b_ and shape_h == shape_h_ and shape_h == shape_h__);
    DG_HOST_ASSERT(shape_r == shape_r_ and shape_d == shape_d_);
    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(sfa.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(sfb.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(a.stride(2) == 1 and b.stride(2) == 1 and d.stride(2) == 1);

    check_sf_layout(sfa, shape_h, shape_r, 1, gran_k, shape_b);
    check_sf_layout(sfb, shape_d, shape_r, gran_mn, gran_k, shape_h);

    const int num_d_tiles = ceil_div(shape_d, output_tile_d);
    const int grid_dim = shape_b * shape_h * num_d_tiles;
    const SM120FP8BhrHdrBhdScalarRuntime::Args args = {
        .shape_b = shape_b,
        .shape_h = shape_h,
        .shape_d = shape_d,
        .shape_r = shape_r,
        .a_stride_b = a.stride(0),
        .a_stride_h = a.stride(1),
        .a_stride_r = a.stride(2),
        .sfa_stride_b = sfa.stride(0),
        .sfa_stride_h = sfa.stride(1),
        .sfa_stride_r = sfa.stride(2),
        .b_stride_h = b.stride(0),
        .b_stride_d = b.stride(1),
        .b_stride_r = b.stride(2),
        .sfb_stride_h = sfb.stride(0),
        .sfb_stride_d = sfb.stride(1),
        .sfb_stride_r = sfb.stride(2),
        .d_stride_b = d.stride(0),
        .d_stride_h = d.stride(1),
        .d_stride_d = d.stride(2),
        .a = a.data_ptr(),
        .sfa = sfa.data_ptr<float>(),
        .b = b.data_ptr(),
        .sfb = sfb.data_ptr<float>(),
        .d = d.data_ptr(),
        .output_dtype = d.scalar_type(),
        .launch_args = LaunchArgs(grid_dim, output_tile_d)
    };
    const auto code = SM120FP8BhrHdrBhdScalarRuntime::generate(args);
    const auto runtime = compiler->build("sm120_fp8_bhr_hdr_bhd_scalar", code);
    SM120FP8BhrHdrBhdScalarRuntime::launch(runtime, args);
}

} // namespace deep_gemm
