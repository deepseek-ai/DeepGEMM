#pragma once

#include <cuda_runtime.h>
#include <filesystem>

#include "../utils/exception.hpp"
#include "../utils/format.hpp"
#include "../utils/system.hpp"
#include "device_runtime.hpp"

namespace deep_gemm {

struct LaunchArgs {
    std::pair<int, int> grid_dim;
    int num_threads;
    int smem_size;
    int cluster_dim;

    LaunchArgs(const int& grid_dim_x, const int& num_threads, const int& smem_size = 0, const int& cluster_dim = 1):
        grid_dim({grid_dim_x, 1}), num_threads(num_threads), smem_size(smem_size), cluster_dim(cluster_dim) {}

    LaunchArgs(const std::pair<int, int>& grid_dim, const int& num_threads, const int& smem_size = 0, const int& cluster_dim = 1):
        grid_dim(grid_dim), num_threads(num_threads), smem_size(smem_size), cluster_dim(cluster_dim) {}
};

template <typename T>
concept HasLaunchArgs = requires (const T& t) {
    { t.launch_args } -> std::convertible_to<decltype(t.launch_args)>;
};

class KernelRuntime final {
public:
    static std::filesystem::path cuda_home;

    cudaLibrary_t library;
    cudaKernel_t kernel;

    explicit KernelRuntime(const std::filesystem::path& dir_path) {
        // NOLINT(*-pro-type-member-init)
        const auto& cuobjdump_path = cuda_home / "bin" / "cuobjdump";
        const auto& cubin_path = dir_path / "kernel.cubin";
        if (get_env<int>("DG_JIT_DEBUG"))
            printf("Loading CUBIN: %s\n", cubin_path.c_str());

        // Find the only symbol
        // TODO: use kernel enumeration for newer drivers
        const std::vector<std::string> illegal_names = {"vprintf", "__instantiate_kernel", "__internal", "__assertfail"};
        const auto& [exit_code, symbols] = call_external_command(fmt::format("{} -symbols {}", cuobjdump_path.c_str(), cubin_path.c_str()));
        DG_HOST_ASSERT(exit_code == 0);
        std::istringstream iss(symbols);
        std::vector<std::string> symbol_names;
        for (std::string line; std::getline(iss, line); ) {
            if (line.find("STT_FUNC") == 0 and std::ranges::none_of(illegal_names, [&](const auto& name) { return line.find(name) != std::string::npos; })) {
                const auto& last_space = line.rfind(' ');
                symbol_names.push_back(line.substr(last_space + 1));
            }
        }
        if (get_env<int>("DG_JIT_DEBUG")) {
            printf("Symbol names: ");
            for (const auto& symbol: symbol_names)
                printf("%s, ", symbol.c_str());
            printf("\n");
        }

        // Load from the library
        DG_HOST_ASSERT(symbol_names.size() == 1);
        DG_CUDA_RUNTIME_CHECK(cudaLibraryLoadFromFile(&library, cubin_path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
        DG_CUDA_RUNTIME_CHECK(cudaLibraryGetKernel(&kernel, library, symbol_names[0].c_str()));
    }

    static void set_cuda_home(const std::string& cuda_home_path_by_torch) {
        cuda_home = cuda_home_path_by_torch;
    }

    static bool check_validity(const std::filesystem::path& dir_path) {
        return std::filesystem::exists(dir_path / "kernel.cu") and
               std::filesystem::exists(dir_path / "kernel.cubin");
    }

    ~KernelRuntime() noexcept(false) {
        const auto& error = cudaLibraryUnload(library);
        DG_HOST_ASSERT(error == cudaSuccess or error == cudaErrorCudartUnloading);
    }
};

// Declare after defining
decltype(KernelRuntime::cuda_home) KernelRuntime::cuda_home;

template <typename Derived>
class LaunchRuntime {
public:
    template <typename Args> requires HasLaunchArgs<Args>
    static std::string generate(const Args& args) {
        const auto& code = Derived::generate_impl(args);
        if (get_env<int>("DG_JIT_DEBUG", 0))
            printf("Generated kernel code: %s\n", code.c_str());
        return code;
    }

    template <typename Args> requires HasLaunchArgs<Args>
    static void launch(const std::shared_ptr<KernelRuntime>& kernel_runtime, const Args& args) {
        const auto& kernel = kernel_runtime->kernel;
        const auto& stream = at::cuda::getCurrentCUDAStream();
        const LaunchArgs& launch_args = args.launch_args;

        // Set dynamic shared memory size
        if (launch_args.smem_size > 0)
            DG_CUDA_RUNTIME_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, launch_args.smem_size));

        // Launch config
        cudaLaunchConfig_t config;
        config.gridDim = {static_cast<unsigned>(launch_args.grid_dim.first),
                          static_cast<unsigned>(launch_args.grid_dim.second),
                          1};
        config.blockDim = {static_cast<unsigned>(launch_args.num_threads), 1, 1};
        config.dynamicSmemBytes = launch_args.smem_size;
        config.stream = stream;
        config.numAttrs = 0;

        // Clusters
        cudaLaunchAttribute attr;
        if (launch_args.cluster_dim > 1) {
            attr.id = cudaLaunchAttributeClusterDimension;
            attr.val.clusterDim = {static_cast<unsigned>(launch_args.cluster_dim), 1, 1};
            config.attrs = &attr;
            config.numAttrs = 1;
        }

        // Launch in the derived class
        if (get_env<int>("DG_JIT_DEBUG")) {
            printf("Launch kernel with {%d, %d} x %d, shared memory: %d bytes, cluster: %d, stream: %ld\n",
                   launch_args.grid_dim.first, launch_args.grid_dim.second, launch_args.num_threads,
                   launch_args.smem_size, launch_args.cluster_dim, stream.id());
        }
        Derived::launch_impl(kernel, config, args);
    }
};

} // namespace deep_gemm
