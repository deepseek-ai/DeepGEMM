#pragma once

#include <deep_gemm/ptx/utils.cuh>

namespace deep_gemm {

// `sm_idx[block_idx]`: the SM that block `block_idx` of the launch runs on
CUTLASS_GLOBAL __launch_bounds__(32, 1)
void probe_cluster_placement(int* sm_idx) {
    if (threadIdx.x == 0)
        sm_idx[blockIdx.x] = static_cast<int>(ptx::get_sm_idx());
}

}  // namespace deep_gemm
