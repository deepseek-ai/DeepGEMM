#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm;

static void instantiate_k256_packet() {
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_mega_moe_impl<
        384,
        7168, 3072,
        384, 0,
        6, 16, 128,
        256,
        8, 128,
        128,
        6144,
        98304,
        5,
        3584, 128, 128,
        128, 152,
        4,
        0x1.4p+3f,
        true,
        true,
        true
    >);
}

static void instantiate_k128_packet() {
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_mega_moe_impl<
        2304,
        7168, 3072,
        384, 0,
        6, 192, 128,
        128,
        32, 256,
        128,
        12672,
        202752,
        6,
        3584, 128, 128,
        256, 152,
        4,
        0x1.4p+3f,
        true,
        false,
        true
    >);
}
