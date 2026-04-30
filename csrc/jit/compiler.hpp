#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <nvrtc.h>
#include <regex>
#include <string>

#include "../utils/exception.hpp"
#include "../utils/format.hpp"
#include "../utils/hash.hpp"
#include "../utils/lazy_init.hpp"
#include "../utils/system.hpp"
#include "cache.hpp"
#include "device_runtime.hpp"
#include "include_parser.hpp"

namespace deep_gemm {

class Compiler {
public:
    static std::filesystem::path library_root_path;
    static std::filesystem::path library_include_path;
    static std::filesystem::path cuda_home;
    static std::filesystem::path cuobjdump_path;

    static void prepare_init(const std::string& library_root_path,
                             const std::string& cuda_home_path_by_python) {
        Compiler::library_root_path = library_root_path;
        Compiler::library_include_path = Compiler::library_root_path / "include";
        Compiler::cuda_home = cuda_home_path_by_python;
        Compiler::cuobjdump_path = Compiler::cuda_home / "bin" / "cuobjdump";
    }

    std::string signature, flags;
    std::filesystem::path cache_dir_path;

    Compiler() {
        // Check `prepare_init`
        DG_HOST_ASSERT(not library_root_path.empty());
        DG_HOST_ASSERT(not library_include_path.empty());
        DG_HOST_ASSERT(not cuda_home.empty());
        DG_HOST_ASSERT(not cuobjdump_path.empty());

        // Cache settings
        cache_dir_path = std::filesystem::path(get_env<std::string>("HOME")) / ".deep_gemm";
        if (const auto env_cache_dir_path = get_env<std::string>("DG_JIT_CACHE_DIR"); not env_cache_dir_path.empty())
            cache_dir_path = env_cache_dir_path;

        // The compiler flags applied to all derived compilers
        signature = "unknown-compiler";
        flags = fmt::format("-std=c++{} --diag-suppress=39,161,174,177,186,940 "
                            "--ptxas-options=--register-usage-level=10",
                            get_env<int>("DG_JIT_CPP_STANDARD", 20));
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PTXAS_VERBOSE", 0) or get_env("DG_JIT_PTXAS_CHECK", 0))
            flags += " --ptxas-options=--verbose,--warn-on-local-memory-usage";
        if (get_env("DG_JIT_WITH_LINEINFO", 0))
            flags += " -Xcompiler -rdynamic -lineinfo";
        // NOTES: `--device-debug` (-G) emits full device DWARF so that cuda-gdb
        // can resolve `__device__` global variables / line numbers in JIT
        // kernels. It DISABLES device-side optimization and will tank perf, so
        // it is gated behind an explicit env var. Enable when you need to
        // inspect debug-only globals such as `dg_dbg_prod_iter` after a hang.
        if (get_env("DG_JIT_WITH_DEVICE_DEBUG", 0))
            flags += " --device-debug";
    }

    virtual ~Compiler() = default;

    std::filesystem::path make_tmp_dir() const {
        return make_dirs(cache_dir_path / "tmp");
    }

    static void fsync_path(const std::filesystem::path& path) {
        const auto fd = ::open(path.c_str(), O_RDONLY);
        if (fd >= 0) {
            ::fsync(fd);
            ::close(fd);
        }
    }

    // Recursively fsync a directory: files and subdirectories first (bottom-up), then the directory itself
    // NOTES: ensures data and directory entries are visible on other nodes in distributed filesystems
    static void fsync_dir(const std::filesystem::path& dir_path) { // NOLINT(*-no-recursion)
        for (const auto& entry: std::filesystem::directory_iterator(dir_path)) {
            if (entry.is_directory())
                fsync_dir(entry.path());
            else if (entry.is_regular_file())
                fsync_path(entry.path());
        }
        fsync_path(dir_path);
    }

    static void put(const std::filesystem::path& path, const std::string& data) {
        std::ofstream out(path, std::ios::binary);
        DG_HOST_ASSERT(out.write(data.data(), data.size()));
        out.close();

        // NOTES: fsync to ensure the data is visible to other processes (e.g., NVCC)
        // on distributed filesystems, where `close()` alone does not guarantee persistence
        fsync_path(path);
    }

    std::shared_ptr<KernelRuntime> build(const std::string& name, const std::string& code) const {
        const auto kernel_signature = fmt::format("{}$${}$${}$${}", name, signature, flags, code);
        const auto dir_path = cache_dir_path / "cache" / fmt::format("kernel.{}.{}", name, get_hex_digest(kernel_signature));

        // Hit the runtime cache
        if (const auto runtime = kernel_runtime_cache->get(dir_path); runtime != nullptr)
            return runtime;

        // Compile into a temporary directory, then atomically rename the whole directory
        // NOTES: renaming a directory is atomic on both local and distributed filesystems,
        // avoiding the stale inode issue that occurs when renaming individual files
        const auto tmp_dir_path = make_tmp_dir() / get_uuid();
        make_dirs(tmp_dir_path);

        // Compile into the temporary directory
        const auto tmp_cubin_path = tmp_dir_path / "kernel.cubin";
        if (get_env<int>("DG_JIT_DUMP_ASM") or get_env<int>("DG_JIT_DUMP_PTX")) {
            const auto tmp_ptx_path = tmp_dir_path / "kernel.ptx";
            compile(code, tmp_dir_path, tmp_cubin_path, tmp_ptx_path);
        } else {
            compile(code, tmp_dir_path, tmp_cubin_path);
        }

        // Disassemble if needed
        if (get_env<int>("DG_JIT_DUMP_ASM") or get_env<int>("DG_JIT_DUMP_SASS")) {
            const auto tmp_sass_path = tmp_dir_path / "kernel.sass";
            disassemble(tmp_cubin_path, tmp_sass_path);
        }

        // Fsync before rename to ensure visibility on distributed filesystems
        fsync_dir(tmp_dir_path);

        // Atomically rename the temporary directory to the final cache path
        // NOTES: if another rank already created dir_path, rename will fail — that's fine
        make_dirs(dir_path.parent_path());
        std::error_code error_code;
        std::filesystem::rename(tmp_dir_path, dir_path, error_code);
        if (error_code) {
            // Another rank beat us, then clean up our dir and use the existing one
            // NOTES: avoid `std::filesystem::remove_all` here — it can segfault on
            // distributed filesystems, when concurrent processes operate
            // on the same parent directory, causing stale directory entries
            safe_remove_all(tmp_dir_path);
        }

        // Put into the runtime cache
        const auto runtime = kernel_runtime_cache->get(dir_path);
        DG_HOST_ASSERT(runtime != nullptr);
        return runtime;
    }

    static void disassemble(const std::filesystem::path &cubin_path, const std::filesystem::path &sass_path) {
        // Disassemble the CUBIN file to SASS
        const auto command = fmt::format("{} --dump-sass {} > {}", cuobjdump_path.c_str(), cubin_path.c_str(), sass_path.c_str());
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PRINT_COMPILER_COMMAND", 0))
            printf("Running cuobjdump command: %s\n", command.c_str());
        const auto [return_code, output] = call_external_command(command);
        if (return_code != 0) {
            printf("cuobjdump failed: %s\n", output.c_str());
            DG_HOST_ASSERT(false and "cuobjdump failed");
        }
    }

    virtual void compile(const std::string &code, const std::filesystem::path& dir_path, const std::filesystem::path &cubin_path, const std::optional<std::filesystem::path> &ptx_path = std::nullopt) const = 0;
};

DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_root_path);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_include_path);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, cuda_home);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, cuobjdump_path);

class NVCCCompiler final: public Compiler {
    std::filesystem::path nvcc_path;

    std::pair<int, int> get_nvcc_version() const {
        DG_HOST_ASSERT(std::filesystem::exists(nvcc_path));

        // Call the version command
        const auto command = std::string(nvcc_path) + " --version";
        const auto [return_code, output] = call_external_command(command);
        DG_HOST_ASSERT(return_code == 0);

        // The version should be at least 12.3, for the best performance with 12.9
        int major, minor;
        std::smatch match;
        DG_HOST_ASSERT(std::regex_search(output, match, std::regex(R"(release (\d+\.\d+))")));
        std::sscanf(match[1].str().c_str(), "%d.%d", &major, &minor);
        DG_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVCC version should be >= 12.3");
        if (major == 12 and minor < 9)
            printf("Warning: please use at least NVCC 12.9 for the best DeepGEMM performance\n");
        return {major, minor};
    }

public:
    NVCCCompiler() {
        // Override the compiler signature
        nvcc_path = cuda_home / "bin" / "nvcc";
        if (const auto env_nvcc_path = get_env<std::string>("DG_JIT_NVCC_COMPILER"); not env_nvcc_path.empty())
            nvcc_path = env_nvcc_path;
        const auto [nvcc_major, nvcc_minor] = get_nvcc_version();
        signature = fmt::format("NVCC{}.{}", nvcc_major, nvcc_minor);

        // The override the compiler flags
        // Only NVCC >= 12.9 supports arch-specific family suffix
        const auto arch = device_runtime->get_arch(false, nvcc_major > 12 or nvcc_minor >= 9);
        flags = fmt::format("{} -I{} --gpu-architecture=sm_{} "
                            "--compiler-options=-fPIC,-O3,-fconcepts,-Wno-deprecated-declarations,-Wno-abi "
                            "-O3 --expt-relaxed-constexpr --expt-extended-lambda",
                            flags, library_include_path.c_str(), arch);
    }

    void compile(const std::string &code, const std::filesystem::path& dir_path,
                 const std::filesystem::path &cubin_path,
                 const std::optional<std::filesystem::path> &ptx_path) const override {
        // Write the code into the cache directory
        const auto code_path = dir_path / "kernel.cu";
        put(code_path, code);

        // Compile
        // Avoid cwd files shadowing C++ standard library headers
        const auto compile_dir = make_tmp_dir();
        const auto command = fmt::format("cd {} && {} {} -cubin -o {} {}",
            compile_dir.c_str(), nvcc_path.c_str(), code_path.c_str(), cubin_path.c_str(), flags);
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PRINT_COMPILER_COMMAND", 0))
            printf("Running NVCC command: %s\n", command.c_str());
        const auto [return_code, output] = call_external_command(command);
        if (return_code != 0) {
            printf("NVCC compilation failed: %s\n", output.c_str());
            DG_HOST_ASSERT(false and "NVCC compilation failed");
        }

        // Compile to PTX if needed
        if (ptx_path.has_value()) {
            const auto ptx_command = fmt::format("cd {} && {} {} -ptx -o {} {}",
                compile_dir.c_str(), nvcc_path.c_str(), code_path.c_str(), ptx_path->c_str(), flags);
            if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PRINT_COMPILER_COMMAND", 0))
                printf("Running NVCC PTX command: %s\n", ptx_command.c_str());
            const auto [ptx_return_code, ptx_output] = call_external_command(ptx_command);
            if (ptx_return_code != 0) {
                printf("NVCC PTX compilation failed: %s\n", ptx_output.c_str());
                DG_HOST_ASSERT(false and "NVCC PTX compilation failed");
            }
        }

        // Check local memory usage
        if (get_env("DG_JIT_PTXAS_CHECK", 0))
            DG_HOST_ASSERT(not std::regex_search(output, std::regex(R"(Local memory used)")));

        // Print PTXAS log
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PTXAS_VERBOSE", 0))
            printf("%s", output.c_str());
    }
};

class NVRTCCompiler final: public Compiler {
public:
    NVRTCCompiler() {
        // Override the compiler signature
        int major, minor;
        DG_NVRTC_CHECK(nvrtcVersion(&major, &minor));
        signature = fmt::format("NVRTC{}.{}", major, minor);
        DG_HOST_ASSERT((major > 12 or (major == 12 and minor >= 3)) and "NVRTC version should be >= 12.3");

        // Build include directories list
        std::string include_dirs;
        include_dirs += fmt::format("-I{} ", library_include_path.string());
        include_dirs += fmt::format("-I{} ", (cuda_home / "include").string());

        // Add PCH support for version 12.8 and above
        // NOTES: PCH is vital for compilation speed
        std::string pch_flags;
        if (major > 12 or minor >= 8) {
            pch_flags = "--pch ";
            if (get_env<int>("DG_JIT_DEBUG", 0))
                pch_flags += "--pch-verbose=true ";
        }

        // Override the compiler flags
        // Only NVRTC >= 12.9 supports arch-specific family suffix
        const auto arch = device_runtime->get_arch(false, major > 12 or minor >= 9);
        flags = fmt::format("{} {}--gpu-architecture=sm_{} -default-device {} --device-int128",
                            flags, include_dirs, arch, pch_flags);
    }

    void compile(const std::string &code, const std::filesystem::path& dir_path,
                 const std::filesystem::path &cubin_path,
                 const std::optional<std::filesystem::path> &ptx_path) const override {
        // Write the code into the cache directory
        const auto code_path = dir_path / "kernel.cu";
        put(code_path, code);

        // Parse compilation options
        std::istringstream iss(flags);
        std::vector<std::string> options;
        std::string option;
        while (iss >> option)
            options.push_back(option);

        // Convert to C-style string array for NVRTC
        std::vector<const char*> option_cstrs;
        for (const auto& opt: options)
            option_cstrs.push_back(opt.c_str());

        // Print compiler command if requested
        if (get_env<int>("DG_JIT_DEBUG", 0) or get_env<int>("DG_JIT_PRINT_COMPILER_COMMAND", 0)) {
            printf("Compiling JIT runtime with NVRTC options: ");
            for (const auto& opt: options)
                printf("%s ", opt.c_str());
            printf("\n");
        }

        // Create NVRTC program and compile
        nvrtcProgram program;
        DG_NVRTC_CHECK(nvrtcCreateProgram(&program, code.c_str(), "kernel.cu", 0, nullptr, nullptr));
        const auto compile_result = nvrtcCompileProgram(program, static_cast<int>(option_cstrs.size()), option_cstrs.data());

        // Get and print compiler log
        size_t log_size;
        DG_NVRTC_CHECK(nvrtcGetProgramLogSize(program, &log_size));
        if (get_env<int>("DG_JIT_DEBUG", 0) or compile_result != NVRTC_SUCCESS) {
            if (compile_result != NVRTC_SUCCESS)
                DG_HOST_ASSERT(log_size > 1);
            if (log_size > 1) {
                std::string compilation_log(log_size, '\0');
                DG_NVRTC_CHECK(nvrtcGetProgramLog(program, compilation_log.data()));
                printf("NVRTC log: %s\n", compilation_log.c_str());
            }
        }

        if (ptx_path.has_value()) {
            // Get PTX size and data if needed
            size_t ptx_size;
            DG_NVRTC_CHECK(nvrtcGetPTXSize(program, &ptx_size));
            std::string ptx_data(ptx_size, '\0');
            DG_NVRTC_CHECK(nvrtcGetPTX(program, ptx_data.data()));

            // Write into the file system
            put(ptx_path.value(), ptx_data);
        }

        // Get CUBIN size and data
        size_t cubin_size;
        DG_NVRTC_CHECK(nvrtcGetCUBINSize(program, &cubin_size));
        std::string cubin_data(cubin_size, '\0');
        DG_NVRTC_CHECK(nvrtcGetCUBIN(program, cubin_data.data()));

        // Write into the file system
        put(cubin_path, cubin_data);

        // Cleanup
        DG_NVRTC_CHECK(nvrtcDestroyProgram(&program));
    }
};

static auto compiler = LazyInit<Compiler>([]() -> std::shared_ptr<Compiler> {
    if (get_env<int>("DG_JIT_USE_NVRTC", 0)) {
        return std::make_shared<NVRTCCompiler>();
    } else {
        return std::make_shared<NVCCCompiler>();
    }
});

} // namespace deep_gemm
