#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>

#include "../utils/exception.hpp"
#include "../utils/format.hpp"
#include "../utils/hash.hpp"
#include "../utils/system.hpp"
#include "cache.hpp"
#include "device_runtime.hpp"

namespace deep_gemm {

class Compiler {
    std::string library_version;
    std::filesystem::path library_root_path;

    static void collect_files(const std::filesystem::path& dir,
                        std::vector<std::filesystem::path>& out) {
        // We met ABI breakage using std::filesystem::recursive_directory_iterator
        // Use std::filesystem::directory_iterator instead
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            if (entry.is_directory()) {
                collect_files(entry.path(), out);
            } else if (entry.is_regular_file() && entry.path().extension()==".cuh") {
                out.emplace_back(entry.path());
            }
        }
    }

    std::string get_library_version() const {
        std::vector<std::filesystem::path> files;
        collect_files(library_include_path / "deep_gemm", files);
        std::sort(files.begin(), files.end());

        std::stringstream ss;
        for (const auto& f : files) {
            std::ifstream in(f, std::ios::binary);
            ss << in.rdbuf();
        }
        return get_hex_digest(ss.str());
    }

public:
    std::string signature, flags;
    std::filesystem::path library_include_path;
    std::filesystem::path cache_dir_path;

    explicit Compiler(const std::filesystem::path& library_root_path) {
        // Static library paths
        this->library_root_path = library_root_path;
        this->library_include_path = library_root_path / "include";
        this->library_version = get_library_version();

        // Cache settings
        cache_dir_path = std::filesystem::path(get_env<std::string>("HOME")) / ".deep_gemm";
        if (const auto& env_cache_dir_path = get_env<std::string>("DG_JIT_CACHE_DIR"); not env_cache_dir_path.empty())
            cache_dir_path = env_cache_dir_path;

        // The compiler flags applied to all derived compilers
        signature = "unknown-compiler";
        std::string ptxas_flags = "--ptxas-options=--register-usage-level=10";
        if (get_env<int>("DG_JIT_PTXAS_VERBOSE", 0))
            ptxas_flags += ",--verbose";
        flags = fmt::format("-std=c++20 --diag-suppress=39,161,174,177,186,940 {}", ptxas_flags);
    }

    virtual ~Compiler() = default;

    std::filesystem::path make_tmp_dir() const {
        return make_dirs(cache_dir_path / "tmp");
    }

    std::filesystem::path get_tmp_file_path() const {
        return make_tmp_dir() / get_uuid();
    }

    void put(const std::filesystem::path& path, const std::string& data) const {
        const auto tmp_file_path = get_tmp_file_path();

        // Write into the temporary file
        std::ofstream out(tmp_file_path, std::ios::binary);
        DG_HOST_ASSERT(out.write(data.data(), data.size()));
        out.close();

        // Atomically replace
        std::filesystem::rename(tmp_file_path, path);
    }

    std::shared_ptr<KernelRuntime> build(const std::string& name, const std::string& code) const {
        const auto kernel_signature = fmt::format("{}$${}$${}$${}$${}", name, library_version, signature, flags, code);
        const auto dir_path = cache_dir_path / "cache" / fmt::format("kernel.{}.{}", name, get_hex_digest(kernel_signature));

        // Hit the runtime cache
        if (const auto& runtime = kernel_runtime_cache->get(dir_path); runtime != nullptr)
            return runtime;

        // Create the kernel directory
        make_dirs(dir_path);

        // Compile into a temporary CUBIN
        const auto tmp_cubin_path = get_tmp_file_path();
        compile(code, dir_path, tmp_cubin_path);

        // Replace into the cache directory
        make_dirs(dir_path);
        std::filesystem::rename(tmp_cubin_path, dir_path / "kernel.cubin");

        // Put into the runtime cache
        const auto& runtime = kernel_runtime_cache->get(dir_path);
        DG_HOST_ASSERT(runtime != nullptr);
        return runtime;
    }

    virtual void compile(const std::string &code, const std::filesystem::path& dir_path, const std::filesystem::path &cubin_path) const = 0;
};

class NVCCCompiler final: public Compiler {
    std::filesystem::path nvcc_path;

    std::pair<int, int> get_nvcc_version() const {
        DG_HOST_ASSERT(std::filesystem::exists(nvcc_path));

        // Call the version command
        const auto& command = std::string(nvcc_path) + " --version";
        const auto& [return_code, output] = call_external_command(command);
        DG_HOST_ASSERT(return_code == 0);

        // The version should be at least 12.3, for the best performance with 12.9
        int major, minor;
        std::smatch match;
        DG_HOST_ASSERT(std::regex_search(output, match, std::regex(R"(release (\d+\.\d+))")));
        std::sscanf(match[1].str().c_str(), "%d.%d", &major, &minor);
        DG_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVCC version should be >= 12.3");
        if (major < 12 or (major == 12 and minor < 9))
            printf("Warning: please use at least NVCC 12.9 for the best DeepGEMM performance");
        return {major, minor};
    }

public:
    NVCCCompiler(const std::filesystem::path& library_root_path,
                 const std::filesystem::path& cuda_home_path_by_torch):
            Compiler(library_root_path) {
        // Override the compiler signature
        nvcc_path = cuda_home_path_by_torch / "bin" / "nvcc";
        if (const auto& env_nvcc_path = get_env<std::string>("DG_JIT_NVCC_COMPILER"); not env_nvcc_path.empty())
            nvcc_path = env_nvcc_path;
        const auto& [nvcc_major, nvcc_minor] = get_nvcc_version();
        signature = fmt::format("NVCC{}.{}", nvcc_major, nvcc_minor);

        // The override the compiler flags
        flags = fmt::format("{} -I{} --gpu-architecture=sm_{}a "
                            "--compiler-options=-fPIC,-O3,-fconcepts,-Wno-deprecated-declarations,-Wno-abi "
                            "-cubin -O3 --expt-relaxed-constexpr --expt-extended-lambda",
                            flags, library_include_path.c_str(), device_runtime->get_arch());
    }

    void compile(const std::string &code, const std::filesystem::path& dir_path, const std::filesystem::path &cubin_path) const override {
        // Write the code into the cache directory
        const auto& code_path = dir_path / "kernel.cu";
        put(code_path, code);

        // Compile
        const auto& command = fmt::format("{} {} -o {} {}", nvcc_path.c_str(), code_path.c_str(), cubin_path.c_str(), flags);
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PRINT_COMPILER_COMMAND", 0))
            printf("Running NVCC command: %s", command.c_str());
        const auto& [return_code, output] = call_external_command(command);
        if (return_code != 0) {
            printf("NVCC compilation failed: %s", output.c_str());
            DG_HOST_ASSERT(false and "NVCC compilation failed");
        }

        // Print PTXAS log
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PTXAS_VERBOSE", 0))
            printf("%s", output.c_str());
    }
};

static std::shared_ptr<Compiler> compiler = nullptr;

} // namespace deep_gemm
