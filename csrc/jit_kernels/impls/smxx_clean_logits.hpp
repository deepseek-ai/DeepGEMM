#pragma once

#include <format>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

// Standalone logits cleaner. The SM100/SM90 MQA kernels fuse cleaning; this is
// kept for the SM120 path, whose kernels do not.
static void smxx_clean_logits(const torch::Tensor& logits,
                              const std::optional<torch::Tensor>& cu_seq_len_k_start,
                              const torch::Tensor& cu_seq_len_k_end,
                              const int& next_n,
                              const int& seq_len, const int& seq_len_kv,
                              const uint64_t &stride_logits) {
    const int block_kv = 8192;
    const int num_warps = 8;
    const int smem_size = block_kv * sizeof(float);

    // Compile
    const auto kernel = jit->compile("smxx_clean_logits", std::format(R"(
#include <deep_gemm/impls/smxx_clean_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&smxx_clean_logits<
        {}, {}, {}, {}
    >);
}};
)", next_n, block_kv, num_warps, to_string(logits.scalar_type())));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(runtime->get_num_sms(), 1, 1),
            .block_dim = dim3(num_warps * 32, 1, 1),
        },
        seq_len, seq_len_kv, static_cast<int64_t>(stride_logits),
        cu_seq_len_k_start.has_value() ? cu_seq_len_k_start.value().data_ptr<int>() : nullptr,
        cu_seq_len_k_end.data_ptr<int>(),
        logits.data_ptr()
    );
}

} // namespace deep_gemm
