#pragma once

#include <deep_gemm/common/utils.cuh>

namespace deep_gemm::sm90 {

template <typename dtype_t>
struct SM90_U32x2_STSM_N {
    __device__ __forceinline__ static void
    copy(dtype_t src_0, dtype_t src_1, void* smem_dst) {
        const uint32_t src[2] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1)};
        asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n"
                     :: "l"(smem_dst), "r"(src[0]), "r"(src[1]));
    }
};

} // namespace `deep_gemm::sm90`
