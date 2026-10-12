#pragma once

#include <hggc_runtime_api.h>
#include <hggc.h>
#include <chrono>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <hgrtc.h>
#include <regex>
#include <string>

#include "../utils/exception.hpp"
#include "../utils/format.hpp"
#include "../utils/hash.hpp"
#include "../utils/lazy_init.hpp"
#include "../utils/system.hpp"
#include "../utils/utils.hpp"
#include "cache.hpp"
#include "device_runtime.hpp"
#include "include_parser.hpp"
#include "acarch.h"

namespace deep_gemm {

class Compiler {
public:
    static std::filesystem::path library_root_path;
    static std::filesystem::path library_include_path;
    static std::filesystem::path sdk_home;
    mutable int blocks_per_cu = 1;

    // Single source of truth for the actlize include search paths, shared by the offline (HGCC)
    // and runtime (HGRTC) compilers.
    static std::vector<std::string> actlize_include_paths() {
        const std::string inc = library_include_path.string();
        return {inc + "/actlize_v1.0.0", inc};
    }

    static void prepare_init(const std::string& library_root_path,
                             const std::string& sdk_home_path) {
        Compiler::library_root_path = library_root_path;
        Compiler::library_include_path = Compiler::library_root_path / "include";
        Compiler::sdk_home = sdk_home_path;
    }

    std::string signature, flags;
    std::filesystem::path cache_dir_path;

    Compiler() {
        // Check `prepare_init`
        DG_HOST_ASSERT(not library_root_path.empty());
        DG_HOST_ASSERT(not library_include_path.empty());
        DG_HOST_ASSERT(not sdk_home.empty());

        // Cache settings
        cache_dir_path = std::filesystem::path(get_env<std::string>("HOME")) / ".deep_gemm";
        if (const auto& env_cache_dir_path = get_env<std::string>("DG_JIT_CACHE_DIR"); not env_cache_dir_path.empty())
            cache_dir_path = env_cache_dir_path;

        // The compiler flags applied to all derived compilers.
        // NOTE: keep only options this compiler actually accepts here -- it rejects both
        // --diag-suppress=... and --ptxas-options=..., so neither is seeded. Everything
        // seeded below is inherited by the derived compilers.
        signature = "unknown-compiler";
        flags = fmt::format("-std=c++{} ",
                            //"--ptxas-options=--register-usage-level=10",
                            get_env<int>("DG_CPP_STANDARD", 17));
        if (get_env("DG_JIT_WITH_LINEINFO", 0))
            flags += " -Xcompiler -rdynamic -lineinfo";
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
        if (const auto& runtime = kernel_runtime_cache->get(dir_path); runtime != nullptr)
            return runtime;

        // Compile into a temporary directory, then atomically rename the whole directory
        // NOTES: renaming a directory is atomic on both local and distributed filesystems,
        // avoiding the stale inode issue that occurs when renaming individual files
        const auto tmp_dir_path = make_tmp_dir() / get_uuid();
        make_dirs(tmp_dir_path);

        // Compile into a temporary HGBIN
        const auto tmp_hgbin_path = tmp_dir_path / "kernel.hgbin";
        compile(code, tmp_dir_path, tmp_hgbin_path, name);

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

    virtual void compile(const std::string &code, const std::filesystem::path& dir_path, const std::filesystem::path &hgbin_path, const std::string& name) const = 0;
};

DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_root_path);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, library_include_path);
DG_DECLARE_STATIC_VAR_IN_CLASS(Compiler, sdk_home);

class HGCCCompiler final: public Compiler {
    std::filesystem::path hgcc_path;

    // >>> PORTING LANDMARK: get_hgcc_version() — the offline compat script rewrites this whole
    //     method (matched by its signature below, NOT by this note). Interior edits are safe.
    // Query the hgcc driver version (best-effort; never fatal on format mismatch)
    std::string get_hgcc_version() const {
        DG_HOST_ASSERT(std::filesystem::exists(hgcc_path) and "hgcc compiler not found");

        // Call the version command
        const auto command = std::string(hgcc_path) + " --version";
        const auto [return_code, output] = call_external_command(command);
        DG_HOST_ASSERT(return_code == 0 and "Failed to query hgcc --version");

        std::smatch match;
        if (std::regex_search(output, match, std::regex(R"(version (\d+\.\d+(?:\.\d+)?))")))
            return match[1].str();
        return "unknown";
    }
    // <<< PORTING LANDMARK: end of get_hgcc_version().

public:
    HGCCCompiler() {
        // Locate the offline compiler binary
        hgcc_path = sdk_home / "bin" / "hgcc";
        if (const auto& env_hgcc_path = get_env<std::string>("DG_JIT_HGCC_COMPILER"); not env_hgcc_path.empty())
            hgcc_path = env_hgcc_path;
        signature = fmt::format("HGCC{}", get_hgcc_version());

        // Flags are APPENDED to the ones seeded by the base Compiler ctor (-std=c++NN plus
        // the opt-in -lineinfo addition), so those are preserved here too.
        // All configurable paths come from environment variables or SDK detection.

        // --- Language & defines ---
        // NOTE: leading space is required -- the base flags do not always end with one
        // (the optional debug/lineinfo appends have no trailing space).
        flags += " -DUSE_HGGC -DUSE_CLANG -DUSE_ACWRAPPER ";

        // --- Architecture ---
        if (is_ppu1v5_device()) {
            flags += "-arch=ppu_15 ";
        } else {
            flags += "-arch=ppu_10 ";
        }

        // --- Include paths ---
        // Resolved in `compile` for both offline and runtime compilers.

        // --- Output format & optimization ---
        flags += "-hgbin -ftemplate-depth=8192 -O3 -DNDEBUG ";

        // --- Host compiler flags (passed via -Xcompiler) ---
        flags += "-Xcompiler -fPIC ";
        flags += "-Xcompiler -Wno-deprecated-declarations -Xcompiler -Wno-abi ";

        // NOTE: --ptxas-options=--register-usage-level=10 is intentionally not passed here

        // >>> PORTING LANDMARK: source-language flag — dropped by the offline compat script
        //     (matched by the code below, NOT by this note); unused by the ported build.
        // --- Source language ---
        flags += " -x hg ";
        // <<< PORTING LANDMARK: end of dropped source-language flag.
    }

    void compile(const std::string &code, const std::filesystem::path& dir_path, const std::filesystem::path &hgbin_path, const std::string& name) const override {
        // Write the code into the cache directory
        const auto code_path = dir_path / "kernel.cu";
        put(code_path, code);

        // Per-kernel flags: warp-interleaving kernels (gemm_fp8, mqa_logits) use -Xllvm flags,
        // others only need -ppu-simt-branch=false (aligned with compiler.py logic)
        std::string per_kernel_flags;
        if (is_ppu1v5_device()) {
            per_kernel_flags = " -Xllvm -ppu-patch-fence-ppu=false -Xllvm -wno-loop-miss-transform"
                               " -Xllvm -ppu-cg-to-kp1=true -Xllvm -ppu-fix-uninit=true";
            const bool use_warp_interleaving = (name.find("fp8_grouped_deep_gemm") != std::string::npos) ||
                                               (name.find("fp8_deep_gemm") != std::string::npos) ||
                                               (name.find("mqa_logits") != std::string::npos &&
                                                name.find("paged") == std::string::npos);
            if (!use_warp_interleaving) {
                per_kernel_flags += " -Xllvm -ppu-simt-branch=false";
                const bool is_bf16_cute_free = name.find("bf16_deep_gemm_cute_free") != std::string::npos;
                if (is_bf16_cute_free) {
                    per_kernel_flags += " -Xllvm -ppu-pref-mma-reuse=true";
                }
            } else {
                per_kernel_flags += " -Xllvm -ppu-blksync-nb-schedule-boundary=true"
                                    " -Xllvm -ppu-simt-branch=false"
                                    " -Xllvm -ppu-adjust-tsm-valu-war=13"
                                    " -Xllvm -ppu-reassign-subregs=true"
                                    " -Xllvm -ppu-pref-fma-reuse=true"
                                    " -Xllvm -ppu-pref-mma-reuse=true";
            }
            if (name.find("w4a16") != std::string::npos) {
                per_kernel_flags += " -Xllvm -sort-copy-before-coalesce";
            }
        }
        // Include paths for actlize and DeepGEMM headers
        std::string include_flags;
        for (const auto& path: actlize_include_paths())
            include_flags += fmt::format("-I{} ", path);

        // Compile
        // Avoid cwd files shadowing C++ standard library headers
        const auto compile_dir = make_tmp_dir();
        const auto command = fmt::format("cd {} && {} {} -o {} {}{}{}",
            compile_dir.c_str(), hgcc_path.c_str(), code_path.c_str(), hgbin_path.c_str(),
            include_flags, flags, per_kernel_flags);
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PRINT_COMPILER_COMMAND", 0))
            printf("Running HGCC command: %s\n", command.c_str());
        const auto [return_code, output] = call_external_command(command);
        if (return_code != 0) {
            printf("HGCC compilation failed: %s\n", output.c_str());
            DG_HOST_ASSERT(false and "HGCC compilation failed");
        }

        // Check local memory usage
        if (get_env("DG_JIT_PTXAS_CHECK", 0))
            DG_HOST_ASSERT(not std::regex_search(output, std::regex(R"(Local memory used)")));

        // Print PTXAS log
        if (get_env("DG_JIT_DEBUG", 0) or get_env("DG_JIT_PTXAS_VERBOSE", 0))
            printf("%s", output.c_str());
    }
};

class HGRTCCompiler final: public Compiler {
public:
    class RtcOptions {
        public:
        RtcOptions() = delete;
        RtcOptions(acArch_t arch, const std::string& name = "") {
            opts = {
            "--device-as-default-execution-space",
            "-DHGGC_COMPILER_WRAPPER_MODE",
            // "-U__linux__",
            "-DUSE_HGGC",
            };
            auto opts_insert = [this](const std::vector<std::string> & vb) {
                opts.insert(opts.end(), vb.begin(), vb.end());
            };
            auto includes_insert = [this](const std::vector<std::string> & paths) {
            for (auto & path: paths) {
                opts.emplace_back("--include-path=" + path);
            }
            };
            if (getenv("PPU_HOME") == nullptr) {
              printf("No PPU_HOME exist\n");
            }
            std::string sdk_include = std::string(getenv("PPU_HOME")) + "/include";

        #if defined(HGGC_VERSION) && HGGC_VERSION >= 13000
            std::string sdk_include_cccl = std::string(getenv("PPU_HOME")) + "/include/cccl";
            includes_insert({sdk_include, sdk_include_cccl});
        #else
            includes_insert({sdk_include});
        #endif
            // Include paths for actlize and DeepGEMM headers
            includes_insert(Compiler::actlize_include_paths());
        // #else
            opts_insert({
                "-DNDEBUG",
                "-DUSE_CLANG",
                "-no-cache",
                "--diag-suppress=39,174,177,940",
                // "--ptxas-options=--register-usage-level=10", // not supported
            });

#ifdef DG_HGGC_SUPPORT_PCH
            if (!get_env<int>("DG_JIT_DISABLE_PCH", 0)) {
                opts_insert({"-pch"});
            }
#endif

            if (arch == AC_PPU0010) {
                opts_insert({
                    "--ppu-arch=ppu001",
                });
            } else if (arch == AC_PPU0015) {
                opts_insert({
                    "--ppu-arch=ppu0015",
                });
            }
            if (is_ppu1v5_device()) {
                opts_insert({
                    "--ppu-tuning-options=-ppu-patch-fence-ppu=false",
                    "--ppu-tuning-options=-wno-loop-miss-transform",
                    "--ppu-tuning-options=-ppu-cg-to-kp1=true",
                    "--ppu-tuning-options=-ppu-fix-uninit=true",
                });

                const bool use_warp_interleaving =
                    (name.find("fp8_grouped_deep_gemm") != std::string::npos) ||
                    (name.find("fp8_deep_gemm")         != std::string::npos) ||
                    (name.find("mqa_logits")            != std::string::npos &&
                     name.find("paged")                 == std::string::npos);
                if (!use_warp_interleaving) {
                    opts_insert({
                        "--ppu-tuning-options=-ppu-simt-branch=false",
                    });
                } else {
                    opts_insert({
                        "--ppu-tuning-options=-ppu-blksync-nb-schedule-boundary=true",
                        "--ppu-tuning-options=-ppu-simt-branch=false",
                        "--ppu-tuning-options=-ppu-adjust-tsm-valu-war=13",
                        "--ppu-tuning-options=-ppu-reassign-subregs=true",
                        "--ppu-tuning-options=-ppu-pref-fma-reuse=true",
                        "--ppu-tuning-options=-ppu-pref-mma-reuse=true",
                    });
                }
                if (name.find("w4a16") != std::string::npos) {
                    opts_insert({
                        "--ppu-tuning-options=-sort-copy-before-coalesce",
                    });
                }
            }
        // #endif
            if (get_env<int>("DG_JIT_DEBUG", 0) || get_env<int>("DG_JIT_PTXAS_VERBOSE", 0) || get_env<int>("DG_JIT_PTXAS_CHECK", 0)) {
                opts_insert({"--ptxas-options=--verbose,--warn-on-local-memory-usage"});
            }
            if (get_env<int>("DG_JIT_WITH_LINEINFO", 0)) {
                opts_insert({"--generate-line-info"});
            }
            std::string standard = "--std=c++17";
            opts.emplace_back(standard);

            opts_char.resize(opts.size());
            std::transform(opts.begin(), opts.end(), opts_char.begin(), [](const std::string& s) { return s.c_str(); });
        }

        size_t size() {
            return opts_char.size();
        }

        const char* const* data() {
            return opts_char.data();
        }
        private:
        std::vector<std::string> opts;
        std::vector<const char*> opts_char;
    };

    HGRTCCompiler() {
        // Override the compiler signature
        int major, minor;
        DG_HGRTC_CHECK(hgrtcVersion(&major, &minor));
        signature = fmt::format("HGRTC{}.{}", major, minor);

    }

    void compile(const std::string &code, const std::filesystem::path& dir_path, const std::filesystem::path &hgbin_path, const std::string& name) const override {
        // Write the code into the cache directory
        const auto& code_path = dir_path / "kernel.cu";
        put(code_path, code);
        acArch_t arch = is_ppu1v5_device() ? AC_PPU0015 : AC_PPU0010;
        RtcOptions opts(arch, name);
        // Print HGRTC compile options when DG_JIT_DEBUG is enabled
        if (get_env<int>("DG_JIT_DEBUG", 0)) {
            printf("HGRTC compile options (%zu):\n", opts.size());
            for (size_t i = 0; i < opts.size(); ++i) {
                printf("  [%zu] %s\n", i, opts.data()[i]);
            }
        }
        // Create HGRTC program and compile
        hgrtcProgram program;
        DG_HGRTC_CHECK(hgrtcCreateProgram(&program, code.c_str(), "kernel.cu", 0, nullptr, nullptr));
        const auto& compile_result = hgrtcCompileProgram(program, opts.size(), opts.data());

        // Get and print compiler log
        size_t log_size;
        DG_HGRTC_CHECK(hgrtcGetProgramLogSize(program, &log_size));
        if (get_env<int>("DG_JIT_DEBUG", 0) or compile_result != HGRTC_SUCCESS) {
            if (compile_result != HGRTC_SUCCESS)
                DG_HOST_ASSERT(log_size > 1);
            if (log_size > 1) {
                std::string compilation_log(log_size, '\0');
                DG_HGRTC_CHECK(hgrtcGetProgramLog(program, compilation_log.data()));
                printf("HGGCRTC log: %s\n", compilation_log.c_str());
            }
        }

        // Get HGBIN size and data
        size_t hgbin_size;
        DG_HGRTC_CHECK(hgrtcGetHGBINSize(program, &hgbin_size));
        std::string hgbin_data(hgbin_size, '\0');
        DG_HGRTC_CHECK(hgrtcGetHGBIN(program, hgbin_data.data()));

        // Write into the file system
        put(hgbin_path, hgbin_data);
        // Cleanup
        DG_HGRTC_CHECK(hgrtcDestroyProgram(&program));
    }
};

static auto compiler = LazyInit<Compiler>([]() -> std::shared_ptr<Compiler> {
    const int default_use_hgrtc = is_ppu1v5_device() ? 0 : 0;
    if (get_env<int>("DG_JIT_USE_HGRTC", default_use_hgrtc)) {
        return std::make_shared<HGRTCCompiler>();
    } else {
        return std::make_shared<HGCCCompiler>();
    }
});

} // namespace deep_gemm
