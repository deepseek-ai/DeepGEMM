#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/sm120.hpp"

#include "runtime_utils.hpp"

namespace deep_gemm {

class SM120BF16GemmRuntime final: public LaunchRuntime<SM120BF16GemmRuntime> {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        LaunchArgs launch_args;
        const std::optional<std::string> epilogue_type;

        void* gmem_d;
        void* gmem_c;
        void* gmem_a_ptr;
        void* gmem_b_ptr;
        void* grouped_layout;
        void* tensor_map_buffer;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_cd;
    };

    static std::string generate_impl(const Args& args) {
        // kNWarps: BM=64 needs kNWarps=2 (kMWarps=4); BM=128 uses default
        // kNWarps=kNumMathWarps (all warps along M, original layout).
        const uint32_t block_m = args.gemm_config.layout.block_m;
        const uint32_t k_n_warps = (block_m % 64 == 0 and block_m < 128) ? 2u : 1u;
        return fmt::format(R"(
#include <deep_gemm/impls/sm120_bf16_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_bf16_gemm_impl<
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {},
        {}, {},
        {},
        {}, {},
        {},
        {},
        {},
        {}
    >);
}};
)",
        get_compiled_dim(args.gemm_desc.m, 'm', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.k, 'k', args.gemm_desc.compiled_dims),
        args.gemm_desc.num_groups,
        args.gemm_config.layout.block_m, args.gemm_config.layout.block_n, args.gemm_config.layout.block_k,
        args.gemm_config.storage_config.swizzle_a_mode, args.gemm_config.storage_config.swizzle_b_mode,
        args.gemm_config.storage_config.swizzle_cd_mode,
        args.gemm_config.pipeline_config.num_stages,
        args.gemm_config.launch_config.num_tma_threads, args.gemm_config.launch_config.num_math_threads,
        args.gemm_config.launch_config.num_sms,
        to_string(args.gemm_desc.gemm_type), args.gemm_desc.with_accumulation,
        to_string(args.gemm_desc.cd_dtype),
        get_default_epilogue_type(args.epilogue_type),
        args.gemm_desc.major_b == cute::UMMA::Major::K ? "true" : "false",
        k_n_warps);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.gmem_d, args.gmem_c,
            args.gmem_a_ptr, args.gmem_b_ptr,
            args.grouped_layout,
            args.tensor_map_buffer,
            args.gemm_desc.m, args.gemm_desc.n, args.gemm_desc.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_cd));
    }
};

static void sm120_bf16_gemm(const torch::Tensor& a,
                            const torch::Tensor& b,
                            const std::optional<torch::Tensor>& c,
                            const torch::Tensor& d,
                            const int& m, const int& n, const int& k,
                            const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                            const std::string& compiled_dims) {
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);

    const auto desc = GemmDesc {
        .gemm_type = GemmType::Normal,
        .kernel_type = KernelType::KernelNoSF,
        .m = m, .n = n, .k = k, .num_groups = 1,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = c.has_value(),
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(),
        .compiled_dims = compiled_dims
    };
    const auto config = get_best_config<SM120ArchSpec>(desc);

    const auto cd = c.value_or(d);
    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), 1,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.layout.block_m, config.layout.block_n,
                                                n, 1,
                                                config.storage_config.swizzle_cd_mode);

    const SM120BF16GemmRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  1),
        .epilogue_type = std::nullopt,
        .gmem_d = d.data_ptr(),
        .gmem_c = c.has_value() ? cd.data_ptr() : nullptr,
        .gmem_a_ptr = nullptr,
        .gmem_b_ptr = nullptr,
        .grouped_layout = nullptr,
        .tensor_map_buffer = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto code = SM120BF16GemmRuntime::generate(args);
    const auto runtime = compiler->build("sm120_bf16_gemm", code);
    SM120BF16GemmRuntime::launch(runtime, args);
}

static void sm120_m_grouped_bf16_gemm_contiguous(const torch::Tensor& a,
                                                 const torch::Tensor& b,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& grouped_layout,
                                                 const int& num_groups, const int& m, const int& n, const int& k,
                                                 const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                 const std::string& compiled_dims,
                                                 const bool& use_psum_layout,
                                                 const std::optional<int>& expected_m_for_psum_layout) {
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);

    const auto gemm_type = use_psum_layout ?
        GemmType::MGroupedContiguousWithPsumLayout : GemmType::MGroupedContiguous;

    if (expected_m_for_psum_layout)
        DG_HOST_ASSERT(use_psum_layout);

    const auto desc = GemmDesc {
        .gemm_type = gemm_type,
        .kernel_type = KernelType::KernelNoSF,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .expected_m = expected_m_for_psum_layout.value_or(m),
        .expected_n = n, .expected_k = k,
        .expected_num_groups = expected_m_for_psum_layout.has_value() ? num_groups : 1
    };
    const auto config = get_best_config<SM120ArchSpec>(desc);

    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.layout.block_m, config.layout.block_n,
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.storage_config.swizzle_cd_mode);

    const SM120BF16GemmRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  1),
        .epilogue_type = std::nullopt,
        .gmem_d = d.data_ptr(),
        .gmem_c = nullptr,
        .gmem_a_ptr = nullptr,
        .gmem_b_ptr = nullptr,
        .grouped_layout = grouped_layout.data_ptr(),
        .tensor_map_buffer = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto code = SM120BF16GemmRuntime::generate(args);
    const auto runtime = compiler->build("sm120_m_grouped_bf16_gemm_contiguous", code);
    SM120BF16GemmRuntime::launch(runtime, args);
}

static void sm120_m_grouped_bf16_gemm_masked(const torch::Tensor& a,
                                             const torch::Tensor& b,
                                             const torch::Tensor& d,
                                             const torch::Tensor& masked_m,
                                             const int& num_groups, const int& m, const int& n, const int& k,
                                             const int& expected_m,
                                             const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                             const std::string& compiled_dims) {
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);

    const auto desc = GemmDesc {
        .gemm_type = GemmType::MGroupedMasked,
        .kernel_type = KernelType::KernelNoSF,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .expected_m = expected_m, .expected_n = n, .expected_k = k, .expected_num_groups = num_groups
    };
    const auto config = get_best_config<SM120ArchSpec>(desc);

    const auto tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), num_groups,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), num_groups,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.layout.block_m, config.layout.block_n,
                                                static_cast<int>(d.stride(-2)), num_groups,
                                                config.storage_config.swizzle_cd_mode);

    const SM120BF16GemmRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  1),
        .epilogue_type = std::nullopt,
        .gmem_d = d.data_ptr(),
        .gmem_c = nullptr,
        .gmem_a_ptr = nullptr,
        .gmem_b_ptr = nullptr,
        .grouped_layout = masked_m.data_ptr(),
        .tensor_map_buffer = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto code = SM120BF16GemmRuntime::generate(args);
    const auto runtime = compiler->build("sm120_m_grouped_bf16_gemm_masked", code);
    SM120BF16GemmRuntime::launch(runtime, args);
}

static void sm120_bf16_k_grouped_gemm(const torch::Tensor& a,
                                      const torch::Tensor& b,
                                      const std::optional<torch::Tensor>& c,
                                      const torch::Tensor& d,
                                      const int& m, const int& n,
                                      const std::vector<int>& ks, const torch::Tensor& ks_tensor,
                                      const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                      const std::string& compiled_dims) {
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::MN and major_b == cute::UMMA::Major::MN);
    DG_HOST_ASSERT(c.has_value());

    int sum_k = 0;
    for (const auto k: ks) {
        sum_k += k;
        DG_HOST_ASSERT(k % 128 == 0);
    }

    // SM120 K-grouped kernel expects per-group compact K-major layout:
    // each group stored as [mn, Ki] flattened, groups concatenated.
    // API provides MN-major a[sum_k, m], b[sum_k, n] — transpose per group.
    auto a_km = torch::empty({static_cast<int64_t>(sum_k) * m}, a.options());
    auto b_km = torch::empty({static_cast<int64_t>(sum_k) * n}, b.options());
    int prefix = 0;
    for (const auto ki: ks) {
        if (ki == 0) continue;
        a_km.slice(0, static_cast<int64_t>(prefix) * m, static_cast<int64_t>(prefix + ki) * m)
            .copy_(a.slice(0, prefix, prefix + ki).t().contiguous().flatten());
        b_km.slice(0, static_cast<int64_t>(prefix) * n, static_cast<int64_t>(prefix + ki) * n)
            .copy_(b.slice(0, prefix, prefix + ki).t().contiguous().flatten());
        prefix += ki;
    }
    const auto num_groups = static_cast<int>(ks.size());
    const auto max_k = *std::max_element(ks.begin(), ks.end());

    const auto desc = GemmDesc {
        .gemm_type = GemmType::KGroupedContiguous,
        .kernel_type = KernelType::KernelNoSF,
        .m = m, .n = n, .k = sum_k, .num_groups = num_groups,
        .a_dtype = a.scalar_type(), .b_dtype = b.scalar_type(),
        .cd_dtype = d.scalar_type(),
        .major_a = cute::UMMA::Major::K, .major_b = cute::UMMA::Major::K,
        .with_accumulation = c.has_value(),
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims,
        .expected_m = m, .expected_n = n, .expected_k = max_k, .expected_num_groups = num_groups
    };
    const auto config = get_best_config<SM120ArchSpec>(desc);

    // Allocate tensor map buffer for dynamic replacement (A + B per SM)
    const auto num_sms = device_runtime->get_num_sms();
    const auto tensor_map_buffer = torch::empty(
        {num_sms * 2 * static_cast<int>(sizeof(CUtensorMap))},
        a.options().dtype(torch::kByte));

    // Use first non-zero K for initial TMA descriptors
    int first_k = 0;
    for (const auto k: ks)
        if (k != 0) { first_k = k; break; }

    // K-major TMA: inner=K, outer stride = first_k (not sum_k)
    const auto tensor_map_a = make_tma_a_desc(cute::UMMA::Major::K, a_km, m, first_k,
                                              config.storage_config.load_block_m,
                                              config.layout.block_k,
                                              first_k, 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(cute::UMMA::Major::K, b_km, n, first_k,
                                              config.storage_config.load_block_n,
                                              config.layout.block_k,
                                              first_k, 1,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_cd_desc(d, m, n,
                                                config.layout.block_m, config.layout.block_n,
                                                n, num_groups,
                                                config.storage_config.swizzle_cd_mode);
    const auto cd = c.value_or(d);
    const SM120BF16GemmRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size, 1),
        .epilogue_type = std::nullopt,
        .gmem_d = d.data_ptr(),
        .gmem_c = cd.data_ptr(),
        .gmem_a_ptr = a_km.data_ptr(),
        .gmem_b_ptr = b_km.data_ptr(),
        .grouped_layout = ks_tensor.data_ptr(),
        .tensor_map_buffer = tensor_map_buffer.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto code = SM120BF16GemmRuntime::generate(args);
    const auto runtime = compiler->build("sm120_bf16_k_grouped_gemm", code);
    SM120BF16GemmRuntime::launch(runtime, args);
}

static void sm120_bf16_bhr_hdr_bhd(const torch::Tensor& tensor_a,
                                   const torch::Tensor& tensor_b,
                                   const torch::Tensor& tensor_d,
                                   const int& b, const int& h, const int& r, const int& d,
                                   const std::string& compiled_dims = "nk") {
    const auto desc = GemmDesc {
        .gemm_type = GemmType::Batched,
        .kernel_type = KernelType::KernelNoSF,
        .m = b, .n = d, .k = r, .num_groups = h,
        .a_dtype = tensor_a.scalar_type(), .b_dtype = tensor_b.scalar_type(),
        .cd_dtype = tensor_d.scalar_type(),
        .major_a = cute::UMMA::Major::K, .major_b = cute::UMMA::Major::K,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims
    };
    const auto config = get_best_config<SM120ArchSpec>(desc);

    const auto tensor_map_a = make_tma_3d_desc(tensor_a, r, b, h,
                                               config.layout.block_k, config.storage_config.load_block_m, 1,
                                               tensor_a.stride(0), tensor_a.stride(1),
                                               config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_3d_desc(tensor_b, r, d, h,
                                               config.layout.block_k, config.storage_config.load_block_n, 1,
                                               tensor_b.stride(1), tensor_b.stride(0),
                                               config.storage_config.swizzle_b_mode);
    const auto tensor_map_cd = make_tma_3d_desc(tensor_d, d, b, h,
                                                config.layout.block_n, config.layout.block_m, 1,
                                                tensor_d.stride(0), tensor_d.stride(1),
                                                config.storage_config.swizzle_cd_mode);

    const SM120BF16GemmRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  1),
        .epilogue_type = std::nullopt,
        .gmem_d = tensor_d.data_ptr(),
        .gmem_c = nullptr,
        .gmem_a_ptr = nullptr,
        .gmem_b_ptr = nullptr,
        .grouped_layout = nullptr,
        .tensor_map_buffer = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto code = SM120BF16GemmRuntime::generate(args);
    const auto runtime = compiler->build("sm120_bf16_bhr_hdr_bhd", code);
    SM120BF16GemmRuntime::launch(runtime, args);
}

static void sm120_bf16_bhd_hdr_bhr(const torch::Tensor& tensor_a,
                                   const torch::Tensor& tensor_b,
                                   const torch::Tensor& tensor_d,
                                   const int& b, const int& h, const int& r, const int& d,
                                   const std::string& compiled_dims = "nk") {
    // bhd,hdr->bhr: batch=h, M=b, N=r, K=d
    // A=[b,h,d] K-major (d contiguous), B=[h,d,r] MN-major (r=N contiguous)
    const auto desc = GemmDesc {
        .gemm_type = GemmType::Batched,
        .kernel_type = KernelType::KernelNoSF,
        .m = b, .n = r, .k = d, .num_groups = h,
        .a_dtype = tensor_a.scalar_type(), .b_dtype = tensor_b.scalar_type(),
        .cd_dtype = tensor_d.scalar_type(),
        .major_a = cute::UMMA::Major::K, .major_b = cute::UMMA::Major::MN,
        .with_accumulation = false,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(), .compiled_dims = compiled_dims
    };
    const auto config = get_best_config<SM120ArchSpec>(desc);

    // A=[b,h,d]: inner=d(K), outer=b(M), batch=h
    const auto tensor_map_a = make_tma_3d_desc(tensor_a, d, b, h,
                                               config.layout.block_k, config.storage_config.load_block_m, 1,
                                               tensor_a.stride(0), tensor_a.stride(1),
                                               config.storage_config.swizzle_a_mode);
    // B=[h,d,r]: MN-major → inner=r(N), outer=d(K), batch=h
    const auto tensor_map_b = make_tma_3d_desc(tensor_b, r, d, h,
                                               config.storage_config.load_block_n, config.layout.block_k, 1,
                                               tensor_b.stride(1), tensor_b.stride(0),
                                               config.storage_config.swizzle_b_mode);
    // D=[b,h,r]: inner=r(N), outer=b(M), batch=h
    const auto tensor_map_cd = make_tma_3d_desc(tensor_d, r, b, h,
                                                config.layout.block_n, config.layout.block_m, 1,
                                                tensor_d.stride(0), tensor_d.stride(1),
                                                config.storage_config.swizzle_cd_mode);

    const SM120BF16GemmRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  1),
        .epilogue_type = std::nullopt,
        .gmem_d = tensor_d.data_ptr(),
        .gmem_c = nullptr,
        .gmem_a_ptr = nullptr,
        .gmem_b_ptr = nullptr,
        .grouped_layout = nullptr,
        .tensor_map_buffer = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_cd = tensor_map_cd,
    };
    const auto code = SM120BF16GemmRuntime::generate(args);
    const auto runtime = compiler->build("sm120_bf16_bhd_hdr_bhr", code);
    SM120BF16GemmRuntime::launch(runtime, args);
}

} // namespace deep_gemm
