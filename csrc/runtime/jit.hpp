#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include <cutlass/version.h>
#include <deep_jit/backend/cuda/backend.hpp>

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
        return runtime;
    });
}

}  // namespace deep_gemm
