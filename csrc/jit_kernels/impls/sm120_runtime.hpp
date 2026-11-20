#pragma once

#include <format>
#include <memory>

#include "../../runtime/runtime.hpp"
#include "epilogue_class.hpp"

namespace deep_gemm {

// SM120 GEMMs instantiate the identity or the alpha-scaling epilogue operator only
static std::string get_sm120_epilogue_type(const std::optional<std::string>& epilogue_type) {
    return epilogue_type.value_or(IdentityEpilogue().get_epilogue_operator_type());
}

// The operator type to instantiate the kernel with, and its runtime arguments to launch it with
struct SM120EpilogueInput {
    std::string type;
    EpilogueOperatorArgs args;
};

static SM120EpilogueInput make_sm120_epilogue_input(const int& m, const int& n,
                                                    const std::optional<std::string>& epilogue_type = std::nullopt,
                                                    const std::optional<float>& alpha = std::nullopt) {
    if (alpha.has_value()) {
        DG_HOST_ASSERT(not epilogue_type.has_value());
        const AlphaEpilogue epilogue(alpha.value());
        return {epilogue.get_epilogue_operator_type(), epilogue.make_epilogue_operator_args(m, n)};
    }
    return {get_sm120_epilogue_type(epilogue_type), {}};
}

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
