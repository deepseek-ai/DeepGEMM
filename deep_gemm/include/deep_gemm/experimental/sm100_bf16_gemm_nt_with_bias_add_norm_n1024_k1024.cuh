#pragma once

#include <deep_gemm/experimental/sm100_bf16_gemm_nt_with_add_norm_n1024_k2816.cuh>

namespace deep_gemm::experimental {

template <int kNumSMs, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t kSplitK = 1>
CUTLASS_GLOBAL void __launch_bounds__(256, 1)
sm100_bf16_gemm_nt_with_bias_add_norm_n1024_k1024_small_impl(
    uint32_t shape_m,
    const __grid_constant__ cute::TmaDescriptor tensor_map_a,
    const __grid_constant__ cute::TmaDescriptor tensor_map_b,
    void* __restrict__ residual,
    void* __restrict__ d,
    const void* __restrict__ bias,
    const void* __restrict__ gamma,
    float* __restrict__ partials,
    uint32_t* __restrict__ counters) {
    sm100_bf16_gemm_add_norm_small_device<1024, true, kNumSMs, BLOCK_M, BLOCK_N, BLOCK_K, kSplitK>(
        shape_m, tensor_map_a, tensor_map_b,
        residual, d, bias, gamma, partials, counters);
}

template <int kNumSMs, uint32_t BLOCK_M = 128, uint32_t BLOCK_N = 128,
          uint32_t kNumStages = 6, uint32_t kNumCDPages = 2,
          bool kUseResTmaStore = false,
          uint32_t kNumResPages = 0, uint32_t kNumBStages = kNumStages>
CUTLASS_GLOBAL void __launch_bounds__(512, 1)
sm100_bf16_gemm_nt_with_bias_add_norm_n1024_k1024_impl(
    uint32_t shape_m,
    const __grid_constant__ cute::TmaDescriptor tensor_map_a,
    const __grid_constant__ cute::TmaDescriptor tensor_map_b,
    const __grid_constant__ cute::TmaDescriptor tensor_map_residual,
    void* __restrict__ residual,
    const void* __restrict__ bias,
    const void* __restrict__ gamma,
    void* __restrict__ d,
    float* __restrict__ partials,
    uint32_t* __restrict__ counters) {
    sm100_bf16_gemm_add_norm_device<1024, true, kNumSMs, BLOCK_M, BLOCK_N,
                                    kNumStages, kNumCDPages, kUseResTmaStore,
                                    kNumResPages, kNumBStages>(
        shape_m, tensor_map_a, tensor_map_b, tensor_map_residual,
        residual, bias, gamma, d, partials, counters);
}

template <int kNumSMs, uint32_t BLOCK_M = 128, uint32_t kNumCDPagesPerGroup = 2,
          uint32_t kNumAStages = 4, uint32_t kNumBStages = 3, bool kUnalignedIO = false, uint32_t kNumSharedResPages = 0>
CUTLASS_GLOBAL void __launch_bounds__(384, 1)
sm100_bf16_gemm_nt_with_bias_add_norm_n1024_k1024_whole_row_impl(
    uint32_t shape_m,
    const __grid_constant__ cute::TmaDescriptor tensor_map_a,
    const __grid_constant__ cute::TmaDescriptor tensor_map_b,
    const __grid_constant__ cute::TmaDescriptor tensor_map_residual,
    const __grid_constant__ cute::TmaDescriptor tensor_map_d,
    void* __restrict__ residual,
    const void* __restrict__ bias,
    const void* __restrict__ gamma,
    void* __restrict__ d) {
    sm100_bf16_gemm_add_norm_whole_row_device<
        1024, true, kNumSMs, BLOCK_M, kNumCDPagesPerGroup, kNumAStages, kNumBStages, kUnalignedIO, kNumSharedResPages>(
        shape_m, tensor_map_a, tensor_map_b, tensor_map_residual, tensor_map_d,
        residual, bias, gamma, d);
}

} // namespace deep_gemm::experimental
