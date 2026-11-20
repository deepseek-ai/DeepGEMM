#pragma once

#include <format>
#include <torch/python.h>

#include "../../runtime/runtime.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

// The probe chases pointer chains through chunks of this size, one hop per line, so that every timed hop is a fill from HBM
static constexpr int kNumLocalityDomainProbeChunkBytes = 4096;
static constexpr int kNumLocalityDomainProbeLineBytes = 128;
static constexpr int kNumLocalityDomainProbeHops = kNumLocalityDomainProbeChunkBytes / kNumLocalityDomainProbeLineBytes;

// `latency[sm_idx][chunk_idx]`: cycles per hop through chunk `chunk_idx` of SM `sm_idx`'s chunks of `buf`, chased by that SM
static void sm100_locality_domain_probe_chase(const torch::Tensor& buf, const torch::Tensor& latency) {
    // Compile
    const auto kernel = jit->compile("sm100_locality_domain_probe_chase", std::format(R"(
#include <deep_gemm/impls/sm100_locality_domain.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&locality_domain_probe_chase<{}, {}, {}>);
}};
)", kNumLocalityDomainProbeHops, buf.size(1), kNumLocalityDomainProbeChunkBytes));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = SM100ArchSpec::smem_capacity,
            .grid_dim = dim3(jit->device.get_num_sms(), 1, 1),
            .block_dim = dim3(32, 1, 1),
        },
        buf.data_ptr(), latency.data_ptr()
    );
}

}  // namespace deep_gemm
