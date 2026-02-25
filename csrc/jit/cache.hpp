#pragma once

#include <filesystem>
#include <memory>
#include <unordered_map>

#include "kernel_runtime.hpp"

namespace deep_gemm {

class KernelRuntimeCache {
    std::unordered_map<std::string, std::shared_ptr<KernelRuntime>> cache;

public:
    // TODO: consider cache capacity
    KernelRuntimeCache() = default;

    std::shared_ptr<KernelRuntime> get(const std::filesystem::path& dir_path) {
        // Hit the runtime cache
        const auto& dir_path_str = dir_path.string();
        if (const auto& iterator = cache.find(dir_path_str); iterator != cache.end())
            return iterator->second;

        if (KernelRuntime::check_validity(dir_path))
            return cache[dir_path_str] = std::make_shared<KernelRuntime>(dir_path);
        return nullptr;
    }
};

static auto kernel_runtime_cache = std::make_shared<KernelRuntimeCache>();

} // namespace deep_gemm
