#pragma once

#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

template <uint32_t kNumHops, uint32_t kNumChunksPerSM, uint32_t kNumChunkBytes>
CUTLASS_GLOBAL __launch_bounds__(32, 1)
void locality_domain_probe_chase(const uint32_t* buf, uint16_t* out) {
    if (threadIdx.x != 0)
        return;
    const auto sm_idx = ptx::get_sm_idx();
    for (uint32_t chunk_idx = 0; chunk_idx < kNumChunksPerSM; ++ chunk_idx) {
        const auto chunk = buf + static_cast<uint64_t>(sm_idx * kNumChunksPerSM + chunk_idx) * (kNumChunkBytes / 4);
        uint32_t idx = ptx::ld_global_cg(chunk);
        const auto start = clock64();
        #pragma unroll
        for (uint32_t i = 0; i < kNumHops; ++ i)
            idx = ptx::ld_global_cg(chunk + idx);
        const auto cycles = clock64() - start;
        out[sm_idx * kNumChunksPerSM + chunk_idx] = idx == 0xffffffffu ? 0u : static_cast<uint16_t>(cycles / kNumHops);
    }
}

}  // namespace deep_gemm
