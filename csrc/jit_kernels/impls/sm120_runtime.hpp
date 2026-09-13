#pragma once

#include <format>
#include <memory>

#include "../../runtime/runtime.hpp"
#include "epilogue.hpp"

namespace deep_gemm {

struct SM120LaunchArgs {
    int blocks, threads, smem, cluster;

    SM120LaunchArgs(int blocks, int threads, int smem, int cluster)
        : blocks(blocks), threads(threads), smem(smem), cluster(cluster) {}
};

template <typename Impl>
struct SM120LaunchRuntime {
    template <typename Args>
    static std::string generate(const Args& args) {
        return Impl::generate_impl(args);
    }

    template <typename Kernel, typename Args>
    static void launch(const Kernel& kernel, const Args& args) {
        Impl::launch_impl(kernel, args);
    }
};

template <typename Kernel, typename... Args>
static void sm120_launch_kernel(const Kernel& kernel, const SM120LaunchArgs& config, Args... args) {
    jit->launch(kernel, {
        .num_smem_bytes = config.smem,
        .grid_dim = dim3(config.blocks, 1, 1),
        .block_dim = dim3(config.threads, 1, 1),
        .cluster_dim = dim3(config.cluster > 0 ? config.cluster : 1, 1, 1),
    }, args...);
}

} // namespace deep_gemm
