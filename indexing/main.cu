#include "deep_gemm/fp8_gemm.cuh"

using namespace deep_gemm;

int main() {
    int m = 128;
    constexpr int N = 4096;
    constexpr int K = 7168;

    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 128;
    constexpr int BLOCK_N_PADDING = 0;
    constexpr int kSwizzleDMode = 0;
    constexpr int kNumGroups = 1;
    constexpr int kNumStages = 5;
    constexpr int kNumTMAMulticast = 1;
    constexpr bool kIsTMAMulticastOnA = false;

    using gemm_t = Gemm<N, K, BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_N_PADDING, kSwizzleDMode, kNumGroups, kNumStages, kNumTMAMulticast, kIsTMAMulticastOnA, GemmType::Normal>;
    auto tma_a_desc = gemm_t::make_2d_tma_a_desc(reinterpret_cast<__nv_fp8_e4m3*>(0), m);
    auto tma_b_desc = gemm_t::make_2d_tma_b_desc(reinterpret_cast<__nv_fp8_e4m3*>(0));
    auto tma_d_desc = gemm_t::make_2d_tma_d_desc(reinterpret_cast<nv_bfloat16*>(0), m);
    auto tma_scales_a_desc = gemm_t::make_2d_tma_scales_a_desc(reinterpret_cast<float*>(0), m);
    gemm_t::run(nullptr, nullptr, nullptr,
                m,
                tma_a_desc, tma_b_desc, tma_scales_a_desc, tma_d_desc,
                nullptr, 132, 0);
    return 0;
}
