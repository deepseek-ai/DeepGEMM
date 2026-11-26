#pragma once

#include <tuple>

#include "../../utils/exception.hpp"

namespace deep_gemm::mega_moe {

// Share the common byte configurations across training, inference and ring sizing.
static std::tuple<int, int> get_pull_config(const int& num_hidden_bytes) {
    DG_HOST_ASSERT(num_hidden_bytes > 0 and num_hidden_bytes <= 16384 and num_hidden_bytes % 128 == 0);
    if (num_hidden_bytes == 512) return std::make_tuple(512, 8);
    if (num_hidden_bytes == 768) return std::make_tuple(768, 5);
    if (num_hidden_bytes == 1024) return std::make_tuple(1024, 4);
    if (num_hidden_bytes == 1536) return std::make_tuple(1536, 3);
    if (num_hidden_bytes == 2048) return std::make_tuple(2048, 2);
    if (num_hidden_bytes == 2304) return std::make_tuple(2304, 2);
    if (num_hidden_bytes == 3072) return std::make_tuple(3072, 1);
    if (num_hidden_bytes == 3584) return std::make_tuple(3584, 1);
    if (num_hidden_bytes == 4096) return std::make_tuple(4096, 1);
    if (num_hidden_bytes == 4608) return std::make_tuple(4608, 1);
    if (num_hidden_bytes == 5120) return std::make_tuple(5120, 1);
    if (num_hidden_bytes == 6144) return std::make_tuple(3072, 1);
    if (num_hidden_bytes == 7168) return std::make_tuple(3584, 1);
    if (num_hidden_bytes == 8192) return std::make_tuple(4096, 1);
    if (num_hidden_bytes == 10240) return std::make_tuple(2560, 1);
    if (num_hidden_bytes == 14336) return std::make_tuple(3584, 1);

    // Keep fallback chunks near 4 KiB without losing 16-byte alignment.
    if (num_hidden_bytes <= 4096)
        return std::make_tuple(num_hidden_bytes, 4096 / num_hidden_bytes);
    int num_bytes_per_pull = num_hidden_bytes;
    while (num_bytes_per_pull > 4096 and num_bytes_per_pull % 32 == 0)
        num_bytes_per_pull /= 2;
    return std::make_tuple(num_bytes_per_pull, 1);
}

} // namespace deep_gemm::mega_moe
