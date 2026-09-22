#pragma once

#include <filesystem>
#include <format>
#include <memory>
#include <string>

#include <cutlass/version.h>
#include <deep_jit/backend/cuda/backend.hpp>
#include <deep_jit/utils/env.hpp>

namespace deep_gemm {

inline deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>> jit(nullptr);

inline void init_jit(const std::string& library_root_path) {
    const auto library_root = std::filesystem::absolute(library_root_path);
    const auto include_dir = library_root / "include";
    const auto config = deep_jit::Config(
        library_root,
        "DG",
        "cutlass-" + std::to_string(CUTLASS_VERSION),
        {include_dir},
        {"deep_gemm/"});

    jit = deep_jit::LazyInit<deep_jit::Runtime<deep_jit::CUDA>>([config] {
        auto runtime = std::make_shared<deep_jit::Runtime<deep_jit::CUDA>>(config);
        runtime->default_compiler_options.nvcc_flags->
            emplace_back("--diag-suppress=39,161,174,177,186,940");
        runtime->default_compiler_options.nvcc_flags->
            emplace_back("--compiler-options=-Wno-deprecated-declarations,-Wno-abi");
        // Allow overriding the in-kernel barrier timeout, see `comm/barrier.cuh`
        // NOTES: the flags are part of the kernel cache key, so changing this value
        // naturally invalidates the JIT cache
        if (const auto barrier_timeout_seconds = deep_jit::get_env<int>("DG_JIT_BARRIER_TIMEOUT_SECONDS", 0);
            barrier_timeout_seconds > 0) {
            runtime->default_compiler_options.nvcc_flags->
                emplace_back(std::format("-DDG_BARRIER_TIMEOUT_SECONDS={}", barrier_timeout_seconds));
        }
        return runtime;
    });
}

}  // namespace deep_gemm
