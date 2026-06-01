#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"
#include "../heuristics/common.hpp"
#include "../heuristics/sm100.hpp"

#include "runtime_utils.hpp"

namespace deep_gemm {

// FP4xFP4 (MXFP4) GEMM via SM100_MMA_MXF4_SS instruction.
// Distinct from main's sm100_fp8_fp4_gemm_1d1d (MXF8F6F4 path) — this is the
// FP4-specialized hardware path, used when both A and B are kPackedFP4.
class SM100FP4Gemm1D1DRuntime final: public LaunchRuntime<SM100FP4Gemm1D1DRuntime> {
public:
    struct Args {
        GemmDesc gemm_desc;
        GemmConfig gemm_config;
        LaunchArgs launch_args;
        int num_last_stages;

        void* grouped_layout;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_sfa;
        CUtensorMap tensor_map_sfb;
        CUtensorMap tensor_map_c;
        CUtensorMap tensor_map_d;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/sm100_fp4_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp4_gemm_1d1d_impl<
        {}, {},
        {}, {}, {},
        {}, {}, {},
        {},
        {}, {}, {},
        {}, {},
        {}, {},
        {}, {},
        {},
        {},
        {}, {}, {}
    >);
}};
)",
        to_string(args.gemm_desc.major_a), to_string(args.gemm_desc.major_b),
        get_compiled_dim(args.gemm_desc.m, 'm', args.gemm_desc.compiled_dims),
        get_compiled_dim(args.gemm_desc.n, 'n', args.gemm_desc.compiled_dims),
        // FP4: my .cuh expects SHAPE_K in int32 count (= FP4_count / 8).
        // main's desc.k is FP4 logical count.
        get_compiled_dim(args.gemm_desc.k / 8, 'k', args.gemm_desc.compiled_dims),
        args.gemm_config.layout.block_m, args.gemm_config.layout.block_n,
        // FP4: BLOCK_K in int32 count (main's heuristic gives bytes for int8 pack).
        args.gemm_config.layout.block_k / 4,
        args.gemm_desc.num_groups,
        args.gemm_config.storage_config.swizzle_a_mode, args.gemm_config.storage_config.swizzle_b_mode, args.gemm_config.storage_config.swizzle_cd_mode,
        args.gemm_config.pipeline_config.num_stages, args.num_last_stages,
        args.gemm_config.launch_config.num_non_epilogue_threads, args.gemm_config.launch_config.num_epilogue_threads,
        args.gemm_config.layout.get_cluster_size(), args.gemm_config.layout.cluster_n > 1,
        args.gemm_config.launch_config.num_sms,
        args.gemm_config.layout.swap_ab,
        to_string(args.gemm_desc.gemm_type), args.gemm_desc.with_accumulation,
        to_string(args.gemm_desc.cd_dtype));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // FP4: my .cuh expects shape_k in int32 count (= FP4_count / 8) at runtime as well.
        // Note: must be non-const (launch_kernel takes &args, can't bind const* to void*).
        int shape_k_int32 = args.gemm_desc.k / 8;
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.grouped_layout, args.gemm_desc.m, args.gemm_desc.n, shape_k_int32,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_sfa, args.tensor_map_sfb,
            args.tensor_map_c, args.tensor_map_d));
    }
};

// Helper: compute num_last_stages from k / block_k / num_stages.
// desc.k is FP4 logical count; main's heuristic block_k is in bytes (int8 elem of packed FP4 tensor).
// 1 byte = 2 FP4, so K_per_block_fp4 = block_k_bytes * 2.
static int compute_num_last_stages_fp4(int k, int block_k_bytes, int num_stages) {
    const int num_k_blocks = ceil_div(k, block_k_bytes * 2);
    const int rem = num_k_blocks % num_stages;
    return rem == 0 ? num_stages : rem;
}

// My MXF4 kernel allocates 2x SF smem per stage (2 packed int32 per row to cover the
// full BLOCK_K_FP4=256 of 8 SFs) than main's heuristic assumes (1 packed int32 per
// row, covering only 128 FP4). Cap num_stages so total smem fits SM100's 232448-byte
// capacity. Returns {new_num_stages, new_smem_size}.
// Note: this is a v0 workaround. Phase 2: align main's heuristic SF computation with
// my kernel's actual usage so num_stages selection is correct upstream.
static std::pair<int, int> recompute_stages_for_fp4(const GemmConfig& config, int block_m, int block_n, int k_fp4) {
    constexpr int smem_capacity = 232448;
    constexpr int sf_block_align = 128;
    const int sf_block_m = (block_m + sf_block_align - 1) / sf_block_align * sf_block_align;
    const int sf_block_n = (block_n + sf_block_align - 1) / sf_block_align * sf_block_align;

    // Match fea-fp4 kernel's actual smem usage:
    //   A per stage: load_block_m * block_k_bytes
    //   B per stage: load_block_n * block_k_bytes
    //   SFA/B per stage: sf_block_mn * sf_packed_k_per_stage * 4
    // where sf_packed_k_per_stage = block_k_bytes / 64 (since BLOCK_K_FP4 / VS / 4 = block_k_int32 / 4 = block_k_bytes / 16 / 4 / 4 ... wait simpler: block_k_int32/4)
    // Kernel: BLOCK_K_FP4 = block_k_int32 * 8; SF_K_PER_STAGE = BLOCK_K_FP4 / 32;
    //         SF_PACKED_K_PER_STAGE = SF_K_PER_STAGE / 4 = block_k_int32 / 16.
    // In bytes: block_k_bytes / 64.
    const int sf_packed_k_per_stage = config.layout.block_k / 64;
    const int per_stage = config.storage_config.load_block_m * config.layout.block_k
                        + config.storage_config.load_block_n * config.layout.block_k
                        + sf_block_m * sf_packed_k_per_stage * 4
                        + sf_block_n * sf_packed_k_per_stage * 4;

    // Fixed extras (CD smem + barriers + tmem_ptr, conservative estimate)
    // CD smem (must match kernel's SMEM_CD_SIZE_PER_STAGE × kNumTMAStoreStages=2):
    //   swap_ab: STORE_BLOCK_M(=16) * STORE_BLOCK_N(=block_n) * sizeof(cd_dtype)
    //   non-swap: STORE_BLOCK_M(=block_m) * kSwizzleCDMode
    // cd_dtype size: assume fp32 (4 bytes); bf16 epilogue is TODO so we can't reach it here.
    int cd_size;
    if (config.layout.swap_ab) {
        constexpr int store_block_m_swap = 16;
        cd_size = store_block_m_swap * block_n * 4 * 2;  // fp32, 2 stages
    } else {
        cd_size = config.storage_config.store_block_m * config.storage_config.swizzle_cd_mode * 2;
    }
    const int barriers = 12 * 8 * 4 + 4 * 8 * 2 + 8;  // ~ 416 bytes max
    const int tmem_ptr = 4;
    const int fixed_extras = cd_size + barriers + tmem_ptr;

    const int max_stages_smem = (smem_capacity - fixed_extras) / per_stage;
    const int num_k_blocks = ceil_div(k_fp4, config.layout.block_k * 2);
    const int new_num_stages = std::min({12, max_stages_smem, num_k_blocks});
    const int new_smem_size = fixed_extras + new_num_stages * per_stage;
    return {new_num_stages, new_smem_size};
}

// Pick the FP4-optimal Layout (block_m=128 fixed, wave-aware block_n, B-multicast for
// large M, swap_ab for sparse m-grouped). Mirrors fea-fp4's `get_best_fp4_config` but
// returns main's nested Layout struct.
//
// Constants (FP4 MXF4 path):
//   block_m = 128 (UMMA_M for MXF4)
//   block_k = 128 bytes (= 32 int32 = 256 FP4 per K block)
//   sf_pk   = 2  (SF_PACKED_K_PER_STAGE; from block_k_int32/16)
//
// block_n is chosen by wave count + composite score = est_stages^2 * bn,
// with 2-epi-stage tiebreak for multi-wave cases. swap_ab is enabled for
// MGroupedContiguous when expected_m_per_group < BLOCK_M (sparse MoE).
//
// expected_m_per_group: useful per-group row count (m_indices >= 0 sum/G).
// Pass INT_MAX to disable swap_ab gating (default for Normal / non-grouped).
static Layout pick_fp4_layout(const GemmType& gemm_type,
                              const int& m, const int& n, const int& k,
                              const int& num_groups, const int& num_sms,
                              const int& expected_m_per_group = INT_MAX) {
    constexpr int block_m = 128;
    constexpr int block_k_bytes = 128;  // 32 int32; matches main's block_k convention
    constexpr int sf_pk = 2;             // block_k_int32 / 16 = 32/16
    constexpr int sf_block_m_cols = (128 / 32) * sf_pk;  // 8
    constexpr int smem_capacity = 232448;

    // BLOCK_N legality (mirrors fea-fp4's is_fp4_block_n_legal)
    auto is_legal = [&](int bn) {
        if (bn % 16 != 0 || bn < 16 || bn > 256) return false;
        const int sf_block_n = (bn + 127) / 128 * 128;
        const int sf_block_n_cols = (sf_block_n / 32) * sf_pk;
        // TMEM minimum: 1 epi stage must fit
        if ((1 * bn + sf_block_m_cols + sf_block_n_cols) > 512) return false;
        return true;
    };
    auto fits_2_epi = [&](int bn) {
        const int sf_block_n = (bn + 127) / 128 * 128;
        const int sf_block_n_cols = (sf_block_n / 32) * sf_pk;
        return (2 * bn + sf_block_m_cols + sf_block_n_cols) <= 512;
    };

    auto num_blocks = [&](int bn) { return ceil_div(m, block_m) * ceil_div(n, bn) * num_groups; };
    auto num_waves = [&](int bn) { return ceil_div(num_blocks(bn), num_sms); };

    int best_bn = 0, best_waves = 0, best_score = 0;
    bool best_fits_2epi = false;
    for (int bn = 16; bn <= 256; bn += 16) {
        if (!is_legal(bn)) continue;
        const int waves = num_waves(bn);

        // est_stages (conservative, no multicast): per_stage in bytes
        const int sf_block_n = (bn + 127) / 128 * 128;
        const int per_stage = block_m * block_k_bytes + bn * block_k_bytes
                            + 128 * sf_pk * 4 + sf_block_n * sf_pk * 4;
        const int avail = smem_capacity - 32768 - 200;
        const int est_stages = std::min(12, std::max(1, avail / per_stage));
        const int score = est_stages * est_stages * bn;
        const bool cur_fits_2epi = fits_2_epi(bn);
        const bool consider_epi = waves >= 2;  // 2-epi benefit only when SM processes >= 2 tiles

        bool success = false;
        if (best_bn == 0 || waves < best_waves) {
            success = true;
        } else if (waves == best_waves && bn <= n) {
            if (consider_epi && cur_fits_2epi && !best_fits_2epi) success = true;
            else if ((!consider_epi || cur_fits_2epi == best_fits_2epi) && score > best_score) success = true;
        }
        if (success) {
            best_bn = bn; best_waves = waves; best_score = score; best_fits_2epi = cur_fits_2epi;
        }
    }
    DG_HOST_ASSERT(best_bn > 0);

    // Env override (for benchmarking)
    if (const auto env_bn = get_env<int>("DG_FP4_BLOCK_N"); env_bn > 0) {
        DG_HOST_ASSERT(env_bn % 16 == 0 && env_bn <= 256);
        best_bn = env_bn;
    }

    // Multicast: B-multicast (cluster_m=2, cluster_n=1) when M >= 512 for Normal / KGrouped.
    // For m-grouped, fea-fp4 keeps cluster=1 (m_indices iteration breaks multi-CTA M-distribution).
    int cluster_m = 1, cluster_n = 1;
    const bool can_multicast = (m >= 512)
                           && (ceil_div(m, block_m) % 2 == 0)
                           && (num_sms % 2 == 0)
                           && (gemm_type == GemmType::Normal || gemm_type == GemmType::KGroupedContiguous);
    if (can_multicast) {
        cluster_m = 2;
    }

    // swap_ab for sparse m-grouped contiguous (forces block_n=128 + cluster=1)
    bool swap_ab = false;
    if (gemm_type == GemmType::MGroupedContiguous && expected_m_per_group < block_m) {
        swap_ab = true;
        best_bn = 128;
        cluster_m = cluster_n = 1;
    }

    return Layout{swap_ab, block_m, best_bn, block_k_bytes, cluster_m, cluster_n};
}

// Build a GemmDesc forcing a_dtype=b_dtype=kPackedFP4 (since this is the FP4xFP4 wrapper).
static GemmDesc make_fp4_desc(GemmType gemm_type, int m, int n, int k, int num_groups,
                              const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                              const at::ScalarType& cd_dtype, bool with_accumulation,
                              const std::string& compiled_dims,
                              int expected_m = 0, int expected_num_groups = 1) {
    return GemmDesc {
        .gemm_type = gemm_type,
        .kernel_type = KernelType::Kernel1D1D,
        .m = m, .n = n, .k = k, .num_groups = num_groups,
        .a_dtype = kPackedFP4, .b_dtype = kPackedFP4,
        .cd_dtype = cd_dtype,
        .major_a = major_a, .major_b = major_b,
        .with_accumulation = with_accumulation,
        .num_sms = device_runtime->get_num_sms(),
        .tc_util = device_runtime->get_tc_util(),
        .compiled_dims = compiled_dims,
        .expected_m = expected_m > 0 ? expected_m : m,
        .expected_n = n, .expected_k = k,
        .expected_num_groups = expected_num_groups
    };
}

static void sm100_fp4_gemm_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                const torch::Tensor& b, const torch::Tensor& sfb,
                                const std::optional<torch::Tensor>& c,
                                const torch::Tensor& d,
                                const int& m, const int& n, const int& k,
                                const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                const std::string& compiled_dims) {
    constexpr int gran_k = 32;
    const auto desc = make_fp4_desc(GemmType::Normal, m, n, k, 1,
                                    major_a, major_b, d.scalar_type(),
                                    c.has_value(), compiled_dims);
    const auto layout = pick_fp4_layout(GemmType::Normal, m, n, k, 1,
                                        device_runtime->get_num_sms());
    auto config = GemmConfig{
        .layout = layout,
        .storage_config = SM100ArchSpec::get_storage_config(desc, layout),
        .pipeline_config = {},
        .launch_config = SM100ArchSpec::get_launch_config(desc, layout),
    };
    config.pipeline_config = SM100ArchSpec::get_pipeline_config(desc, layout, config.storage_config);
    const auto [new_stages, new_smem] = recompute_stages_for_fp4(config, layout.block_m, layout.block_n, k);
    config.pipeline_config.num_stages = new_stages;
    config.pipeline_config.smem_size = new_smem;

    // View FP4-packed-int8 tensors as int32 (8 FP4 per int32). Same memory layout,
    // different dtype tag -- makes the TMA descriptor use INT32 (not 16U4_ALIGN16B
    // unpacked-smem), which matches my kernel's int32-packed smem expectation.
    const auto a_int32 = a.view(torch::kInt);
    const auto b_int32 = b.view(torch::kInt);
    const int k_int32 = k / 8;
    const int block_k_int32 = config.layout.block_k / 4;

    const auto cd = c.value_or(d);
    const auto tensor_map_a = make_tma_a_desc(major_a, a_int32, m, k_int32,
                                              config.storage_config.load_block_m,
                                              block_k_int32,
                                              static_cast<int>(a_int32.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b_int32, n, k_int32,
                                              config.storage_config.load_block_n,
                                              block_k_int32,
                                              static_cast<int>(b_int32.stride(get_non_contiguous_dim(major_b))), 1,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_d = make_tma_cd_desc(d, m, n,
                                               config.storage_config.store_block_m,
                                               config.storage_config.store_block_n,
                                               static_cast<int>(d.stride(-2)), 1,
                                               config.storage_config.swizzle_cd_mode);
    const auto tensor_map_c = make_tma_cd_desc(cd, m, n,
                                               config.storage_config.store_block_m,
                                               config.storage_config.store_block_n,
                                               static_cast<int>(cd.stride(-2)), 1,
                                               config.storage_config.swizzle_cd_mode);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, gran_k, 1, 0);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                 config.layout.block_n, gran_k, 1, 0);

    if (c.has_value()) {
        if (c->data_ptr() == d.data_ptr()) {
            DG_HOST_ASSERT(c->sizes() == d.sizes() and c->strides() == d.strides());
        } else {
            d.copy_(c.value());
        }
    }

    const SM100FP4Gemm1D1DRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .num_last_stages = compute_num_last_stages_fp4(k, config.layout.block_k, config.pipeline_config.num_stages),
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_c = tensor_map_c,
        .tensor_map_d = tensor_map_d
    };
    const auto code = SM100FP4Gemm1D1DRuntime::generate(args);
    const auto runtime = compiler->build("sm100_fp4_gemm_1d1d", code);
    SM100FP4Gemm1D1DRuntime::launch(runtime, args);
}

static void sm100_m_grouped_fp4_gemm_contiguous_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                     const torch::Tensor& b, const torch::Tensor& sfb,
                                                     const torch::Tensor& d,
                                                     const torch::Tensor& grouped_layout,
                                                     const int& num_groups, const int& m, const int& n, const int& k,
                                                     const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                     const std::string& compiled_dims) {
    constexpr int gran_k = 32;
    const auto desc = make_fp4_desc(GemmType::MGroupedContiguous, m, n, k, num_groups,
                                    major_a, major_b, d.scalar_type(),
                                    false, compiled_dims);
    // For m-grouped contiguous, derive useful-per-group row count from m_indices
    // (count of valid rows / num_groups). Drives swap_ab gating for sparse MoE.
    const auto useful_m = (grouped_layout >= 0).sum().item<int64_t>();
    const int useful_per_group = num_groups > 0 ? static_cast<int>(useful_m) / num_groups : 0;
    const auto layout = pick_fp4_layout(GemmType::MGroupedContiguous, m, n, k, num_groups,
                                        device_runtime->get_num_sms(), useful_per_group);
    auto config = GemmConfig{
        .layout = layout,
        .storage_config = SM100ArchSpec::get_storage_config(desc, layout),
        .pipeline_config = {},
        .launch_config = SM100ArchSpec::get_launch_config(desc, layout),
    };
    config.pipeline_config = SM100ArchSpec::get_pipeline_config(desc, layout, config.storage_config);
    const auto [new_stages_g, new_smem_g] = recompute_stages_for_fp4(config, layout.block_m, layout.block_n, k);
    config.pipeline_config.num_stages = new_stages_g;
    config.pipeline_config.smem_size = new_smem_g;

    const auto a_int32 = a.view(torch::kInt);
    const auto b_int32 = b.view(torch::kInt);
    const int k_int32 = k / 8;
    const int block_k_int32 = config.layout.block_k / 4;

    const auto tensor_map_a = make_tma_a_desc(major_a, a_int32, m, k_int32,
                                              config.storage_config.load_block_m,
                                              block_k_int32,
                                              static_cast<int>(a_int32.stride(get_non_contiguous_dim(major_a))), 1,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b_int32, n, k_int32,
                                              config.storage_config.load_block_n,
                                              block_k_int32,
                                              static_cast<int>(b_int32.stride(get_non_contiguous_dim(major_b))), num_groups,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_d = make_tma_cd_desc(d, m, n,
                                               config.storage_config.store_block_m,
                                               config.storage_config.store_block_n,
                                               static_cast<int>(d.stride(-2)), 1,
                                               config.storage_config.swizzle_cd_mode);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, gran_k, 1, 0);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                 config.layout.block_n, gran_k, num_groups, 0);

    const SM100FP4Gemm1D1DRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .num_last_stages = compute_num_last_stages_fp4(k, config.layout.block_k, config.pipeline_config.num_stages),
        .grouped_layout = grouped_layout.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_c = tensor_map_d,
        .tensor_map_d = tensor_map_d
    };
    const auto code = SM100FP4Gemm1D1DRuntime::generate(args);
    const auto runtime = compiler->build("sm100_m_grouped_fp4_gemm_contiguous_1d1d", code);
    SM100FP4Gemm1D1DRuntime::launch(runtime, args);
}

static void sm100_m_grouped_fp4_gemm_masked_1d1d(const torch::Tensor& a, const torch::Tensor& sfa,
                                                 const torch::Tensor& b, const torch::Tensor& sfb,
                                                 const torch::Tensor& d,
                                                 const torch::Tensor& masked_m,
                                                 const int& num_groups, const int& m, const int& n, const int& k,
                                                 const int& expected_m,
                                                 const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                                 const std::string& compiled_dims) {
    constexpr int gran_k = 32;
    const auto desc = make_fp4_desc(GemmType::MGroupedMasked, m, n, k, num_groups,
                                    major_a, major_b, d.scalar_type(),
                                    false, compiled_dims,
                                    /*expected_m=*/expected_m, /*expected_num_groups=*/num_groups);
    // m-grouped masked: pass expected_m as per-group hint (fea-fp4 invariant);
    // swap_ab heuristic itself only activates for MGroupedContiguous per fea-fp4 v0
    // (masked + swap_ab had NaN issues unresolved upstream).
    const auto layout = pick_fp4_layout(GemmType::MGroupedMasked, m, n, k, num_groups,
                                        device_runtime->get_num_sms(), expected_m);
    auto config = GemmConfig{
        .layout = layout,
        .storage_config = SM100ArchSpec::get_storage_config(desc, layout),
        .pipeline_config = {},
        .launch_config = SM100ArchSpec::get_launch_config(desc, layout),
    };
    config.pipeline_config = SM100ArchSpec::get_pipeline_config(desc, layout, config.storage_config);
    const auto [new_stages_mk, new_smem_mk] = recompute_stages_for_fp4(config, layout.block_m, layout.block_n, k);
    config.pipeline_config.num_stages = new_stages_mk;
    config.pipeline_config.smem_size = new_smem_mk;

    const auto a_int32 = a.view(torch::kInt);
    const auto b_int32 = b.view(torch::kInt);
    const int k_int32 = k / 8;
    const int block_k_int32 = config.layout.block_k / 4;

    const auto tensor_map_a = make_tma_a_desc(major_a, a_int32, m, k_int32,
                                              config.storage_config.load_block_m,
                                              block_k_int32,
                                              static_cast<int>(a_int32.stride(get_non_contiguous_dim(major_a))), num_groups,
                                              config.storage_config.swizzle_a_mode);
    const auto tensor_map_b = make_tma_b_desc(major_b, b_int32, n, k_int32,
                                              config.storage_config.load_block_n,
                                              block_k_int32,
                                              static_cast<int>(b_int32.stride(get_non_contiguous_dim(major_b))), num_groups,
                                              config.storage_config.swizzle_b_mode);
    const auto tensor_map_d = make_tma_cd_desc(d, m, n,
                                               config.storage_config.store_block_m,
                                               config.storage_config.store_block_n,
                                               static_cast<int>(d.stride(-2)), num_groups,
                                               config.storage_config.swizzle_cd_mode);
    const auto tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                 config.layout.block_m, gran_k, num_groups, 0);
    const auto tensor_map_sfb = make_tma_sf_desc(cute::UMMA::Major::MN, sfb, n, k,
                                                 config.layout.block_n, gran_k, num_groups, 0);

    const SM100FP4Gemm1D1DRuntime::Args args = {
        .gemm_desc = desc,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.launch_config.num_sms, config.launch_config.num_threads,
                                  config.pipeline_config.smem_size,
                                  config.layout.get_cluster_size()),
        .num_last_stages = compute_num_last_stages_fp4(k, config.layout.block_k, config.pipeline_config.num_stages),
        .grouped_layout = masked_m.data_ptr(),
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_sfa = tensor_map_sfa,
        .tensor_map_sfb = tensor_map_sfb,
        .tensor_map_c = tensor_map_d,
        .tensor_map_d = tensor_map_d
    };
    const auto code = SM100FP4Gemm1D1DRuntime::generate(args);
    const auto runtime = compiler->build("sm100_m_grouped_fp4_gemm_masked_1d1d", code);
    SM100FP4Gemm1D1DRuntime::launch(runtime, args);
}

} // namespace deep_gemm
