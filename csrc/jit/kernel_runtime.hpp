#pragma once

#include <chrono>

#include "../utils/exception.hpp"
#include "../utils/format.hpp"
#include "../utils/system.hpp"
#include "device_runtime.hpp"
#include "handle.hpp"
#include "include_parser.hpp"

namespace deep_gemm {

struct LaunchArgs {
    std::pair<int, int> grid_dim;
    int num_threads;
    int smem_size;
    int cluster_dim;
    bool enable_pdl;

    LaunchArgs(const int& grid_dim_x, const int& num_threads, const int& smem_size = 0, const int& cluster_dim = 1, const bool& enable_pdl = true):
        grid_dim({grid_dim_x, 1}), num_threads(num_threads), smem_size(smem_size), cluster_dim(cluster_dim), enable_pdl(enable_pdl) {}

    LaunchArgs(const std::pair<int, int>& grid_dim, const int& num_threads, const int& smem_size = 0, const int& cluster_dim = 1, const bool& enable_pdl = true):
        grid_dim(grid_dim), num_threads(num_threads), smem_size(smem_size), cluster_dim(cluster_dim), enable_pdl(enable_pdl) {}
};

class KernelRuntime final {
public:
    static std::filesystem::path cuda_home;

    LibraryHandle library;
    KernelHandle kernel;

    explicit KernelRuntime(const std::filesystem::path& dir_path) {
        // Check `prepare_init`
        DG_HOST_ASSERT(not cuda_home.empty());

        // NOLINT(*-pro-type-member-init)
        const auto cuobjdump_path = cuda_home / "bin" / "cuobjdump";
        const auto cubin_path = dir_path / "kernel.cubin";
        if (get_env<int>("DG_JIT_DEBUG"))
            printf("Loading CUBIN: %s\n", cubin_path.c_str());

        // Record start time
        std::chrono::high_resolution_clock::time_point start_time;
        if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_JIT_PRINT_LOAD_TIME"))
            start_time = std::chrono::high_resolution_clock::now();

#ifdef DG_JIT_USE_LIBRARY_ENUM_KERNELS
        // Load from the library
        kernel = load_kernel(cubin_path, {}, &library);
#else
        // Find the only symbol
        // TODO: use kernel enumeration for newer drivers
        const std::vector<std::string> illegal_names = {"vprintf", "__instantiate_kernel", "__internal", "__assertfail"};
        const auto [exit_code, symbols] = call_external_command(fmt::format("{} -symbols {}", cuobjdump_path.c_str(), cubin_path.c_str()));
        DG_HOST_ASSERT(exit_code == 0);
        std::istringstream iss(symbols);
        std::vector<std::string> symbol_names;
        for (std::string line; std::getline(iss, line); ) {
            if (line.find("STT_FUNC") == 0 and line.find("STO_ENTRY") != std::string::npos and
                std::none_of(illegal_names.begin(), illegal_names.end(),
                [&](const auto name) { return line.find(name) != std::string::npos; })) {
                const auto last_space = line.rfind(' ');
                symbol_names.push_back(line.substr(last_space + 1));
            }
        }

        // Print symbols
        if (symbol_names.size() != 1 or get_env<int>("DG_JIT_DEBUG")) {
            printf("Symbols: ");
            printf(" > CUBIN: %s\n", cubin_path.c_str());
            printf(" > Raw symbols: %s\n", symbols.c_str());
            printf(" > Parsed symbols:\n");
            for (const auto& symbol: symbol_names)
                printf("   > %s, ", symbol.c_str());
        }
        DG_HOST_ASSERT(symbol_names.size() == 1);

        // Load from the library
        kernel = load_kernel(cubin_path, symbol_names[0], &library);
#endif

        // Print load time
        if (get_env<int>("DG_JIT_DEBUG") or get_env<int>("DG_JIT_PRINT_LOAD_TIME")) {
            std::chrono::duration<double, std::milli> load_time = std::chrono::high_resolution_clock::now() - start_time;
            printf("Load time (%s): %.2lf ms\n", dir_path.c_str(), load_time.count());
        }
    }

    static void prepare_init(const std::string& cuda_home_path_by_python) {
        cuda_home = cuda_home_path_by_python;
    }

    static bool check_validity(const std::filesystem::path& dir_path) {
        if (not std::filesystem::exists(dir_path))
            return false;

        // NOTES: if the directory exists, `kernel.cu` and `kernel.cubin` must both exist,
        // because the directory is created atomically via rename
        if (not std::filesystem::exists(dir_path / "kernel.cu") or
            not std::filesystem::exists(dir_path / "kernel.cubin")) {
            printf("Corrupted JIT cache directory (missing kernel.cu or kernel.cubin): %s, "
                   "please run `rm -rf %s` and restart your task.\n",
                   dir_path.c_str(), dir_path.c_str());
            DG_HOST_ASSERT(false and "Corrupted JIT cache directory");
        }
        return true;
    }

    ~KernelRuntime() noexcept(false) {
        unload_library(library);
    }
};

DG_DECLARE_STATIC_VAR_IN_CLASS(KernelRuntime, cuda_home);

template <typename Derived>
class LaunchRuntime {
public:
    template <typename Args>
    static std::string generate(const Args& args) {
        auto code = Derived::generate_impl(args);

        // NOTES: we require that `generate_impl`'s includes never change
        static std::string include_hash;
        if (include_hash.empty())
            include_hash = include_parser->get_hash_value(code);

        // TODO: optimize string concat performance
        code = fmt::format("// Includes' hash value: {}\n{}", include_hash, code);
        if (get_env<int>("DG_JIT_DEBUG"))
            printf("Generated kernel code:\n%s\n", code.c_str());
        return code;
    }

    template <typename Args>
    static void launch(const std::shared_ptr<KernelRuntime>& kernel_runtime, const Args& args) {
        const auto kernel = kernel_runtime->kernel;
        const auto stream = at::cuda::getCurrentCUDAStream();
        LaunchArgs launch_args = args.launch_args;

        // Allow runtime override from Python.
        // NOTES: the default is enabled.
        launch_args.enable_pdl = device_runtime->get_pdl();

        const dim3 grid_dim = {static_cast<unsigned>(launch_args.grid_dim.first),
                               static_cast<unsigned>(launch_args.grid_dim.second),
                               1};
        const dim3 block_dim = {static_cast<unsigned>(launch_args.num_threads), 1, 1};
        auto config = construct_launch_config(kernel, stream, launch_args.smem_size,
                                              grid_dim, block_dim, launch_args.cluster_dim, launch_args.enable_pdl);

        // Launch in the derived class
        if (get_env<int>("DG_JIT_DEBUG")) {
            printf("Launch kernel with {%d, %d} x %d, shared memory: %d bytes, cluster: %d, pdl: %d, stream: %ld\n",
                   launch_args.grid_dim.first, launch_args.grid_dim.second, launch_args.num_threads,
                   launch_args.smem_size, launch_args.cluster_dim, launch_args.enable_pdl, stream.id());
        }
        Derived::launch_impl(kernel, config, args);
    }
};

} // namespace deep_gemm
