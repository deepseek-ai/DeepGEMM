#pragma once

#include <cute/arch/mma_sm100_desc.hpp>
// Reuse some types in the JIT modules
#include <deep_gemm/common/types.hpp>

#include "common.hpp"

namespace deep_gemm {

struct SM90ArchSpec {
    static constexpr int smem_capacity = 232448;

    static int get_ab_load_block_m(const MulticastConfig& multicast_config, const int& block_m) {
        return block_m;
    }

    static int get_ab_load_block_n(const MulticastConfig& multicast_config, const int& block_n) {
        return block_n;
    }

    static int get_cd_store_block_m(const int& block_m) {
        return block_m;
    }

    static int get_cd_store_block_n(const int& block_n) {
        return block_n;
    }

    static bool is_block_size_legal(const KernelType& kernel_type,
                                    const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
                                    const at::ScalarType& ab_dtype, const at::ScalarType& cd_dtype,
                                    const int& block_m, const int& block_n, const int& block_k) {
        // FP32 output does not support `block_m == 256`
        if (cd_dtype == at::kFloat and block_m == 256)
            return false;

        // TODO: more general block N selection
        // Must be some fixed block N selections
        if (block_n > 128 and kernel_type == KernelType::Kernel1D1D and (block_n != 136 and block_n != 152))
            return false;

        // Too many scaling factors in a single block: `block_n > block_k and std::gcd(block_n, block_k) != block_n - block_k`
        // Or too many register spills
        if (block_n > 128 and kernel_type == KernelType::Kernel1D2D and (block_n != 144 and block_n != 160 and block_n != 192))
            return false;

        // Avoid bank conflicts for FP32 output
        if (cd_dtype == torch::kFloat and block_n % 16 == 0)
            return false;

        // The block sizes cannot be too large (for enough registers), so at least one dim less than 128
        return block_m <= 128 or block_n <= 128;
    }

    static bool is_num_stages_legal(const at::ScalarType& ab_dtype, const at::ScalarType& cd_dtype,
                                    const int& num_stages,
                                    const int& block_m, const int& block_n, const int& block_k) {
        // Unrolling both stages and `num_former_iters` will cause large code size
        if (ab_dtype == torch::kFloat8_e4m3fn and block_k % block_n != 0 and block_k / std::gcd(block_n, block_k) <= 4)
            return num_stages <= 4;
        return true;
    }

    static bool should_minimize_num_sms() {
        return true;
    }

    static std::pair<bool, bool> get_multicast_legality(const GemmType& gemm_type,
                                                        const int& m, const int& n, const int& block_m, const int& block_n,
                                                        const int& num_sms) {
        return {
            is_multicast_legal(n, block_n, 2, num_sms, gemm_type == GemmType::MGroupedMasked),
            is_multicast_legal(m, block_m, 2, num_sms, false) and gemm_type != GemmType::MGroupedMasked,
        };
    }

    static ThreadConfig get_thread_config(const KernelType& kernel_type,
                                          const int& block_m, const int& block_n) {
        return ThreadConfig::sm90(128, (block_m == 64 ? 1 : 2) * 128);
    }

    static int get_smem_cd_size(const KernelType& kernel_type,
                                const int& block_m, const int& block_n,
                                const int& swizzle_cd_mode, const at::ScalarType& cd_dtype) {
        return block_m * block_n * static_cast<int>(c10::elementSize(cd_dtype));
    }

    static std::pair<int, int> get_sf_smem_size_per_stage(const KernelType& kernel_type,
                                                          const int& block_m, const int& block_n, const int& block_k,
                                                          const at::ScalarType& ab_dtype, const at::ScalarType& cd_dtype) {
        if (ab_dtype == torch::kBFloat16)
            return {0, 0};

        int smem_sfa_per_stage = block_m * static_cast<int>(sizeof(float));
        int smem_sfb_per_stage = 0;
        // TODO: figure out here
        if (kernel_type == KernelType::Kernel1D1D)
            smem_sfb_per_stage = align(block_n * 4, block_k);
        return {smem_sfa_per_stage, smem_sfb_per_stage};
    }

    static int get_extra_sfb_smem_size(const int& m, const int& n, const int& k,
                                       const int& block_m, const int& block_n, const int& block_k) {
        const auto& use_uniform_sfb = block_k % block_n == 0 ? 1 : 2;
        return align<int>(ceil_div(k, block_k) * static_cast<int>(sizeof(float)) * use_uniform_sfb, 8);
    }

    static int get_barrier_smem_size(const int& num_stages) {
        // For 1D1D kernels, there is an extra barrier for accumulation
        return (num_stages + 1) * 8 * 2;
    }

    static int get_tmem_ptr_smem_size() {
        return 0;
    }
};

} // namespace deep_gemm
