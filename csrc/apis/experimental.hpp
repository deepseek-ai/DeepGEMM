#pragma once

#include <pybind11/pybind11.h>

#include "../experimental/sm100_bf16_gemm_nt_with_add_norm_n1024_k2816.hpp"
#include "../experimental/sm100_bf16_gemm_nt_with_bias_add_norm_n1024_k1024.hpp"
#include "../experimental/sm100_bf16_gemm_nt_with_swiglu_n5632_k1024.hpp"
#include "../experimental/sm100_bf16_newton_schulz.hpp"
#include "../experimental/sm100_fp8_gemm_nt_segmented_k256.hpp"

#include "../experimental/v41f_qkv_lora.hpp"
#include "../experimental/v41f_wv_o_proj.hpp"

namespace deep_gemm::experimental {

static void register_apis(pybind11::module_& m) {
    auto submodule = m.def_submodule("experimental", "Experimental kernels without stability guarantees");

    // FP8 GEMM with segmented A [M, S, 256] and prepacked per-32 UE8M0 scales.
    submodule.def("fp8_gemm_nt_segmented_k256", &fp8_gemm_nt_segmented_k256,
                  pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("d"));

    // torch: d = (F.silu(a @ b[:2816].T) * (a @ b[2816:].T)).bfloat16()
    submodule.def("bf16_gemm_nt_with_swiglu_n5632_k1024", &bf16_gemm_nt_with_swiglu_n5632_k1024,
                  pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("d"));

    // torch: residual += a @ b.T (bf16); d = (residual.float() * rsqrt(residual.float().pow(2).mean(-1, keepdim=True) + 1e-6) * gamma).bfloat16()
    submodule.def("bf16_gemm_nt_with_add_norm_n1024_k2816", &bf16_gemm_nt_with_add_norm_n1024_k2816,
                  pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("gamma"),
                  pybind11::arg("residual"), pybind11::arg("d"));

    // torch: residual += a @ b.T + bias (bf16); d = (residual.float() * rsqrt(residual.float().pow(2).mean(-1, keepdim=True) + 1e-6) * gamma).bfloat16()
    submodule.def("bf16_gemm_nt_with_bias_add_norm_n1024_k1024", &bf16_gemm_nt_with_bias_add_norm_n1024_k1024,
                  pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("bias"), pybind11::arg("gamma"),
                  pybind11::arg("residual"), pybind11::arg("d"));

    submodule.def("batched_bf16_newton_schulz", &batched_bf16_newton_schulz,
                  pybind11::arg("x"), pybind11::arg("out"),
                  pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"));

    submodule.def("v41f_qkv_lora", &v41f_qkv_lora,
                  pybind11::arg("x"), pybind11::arg("qkv_weight"), pybind11::arg("q_b_weight"),
                  pybind11::arg("q_norm_weight"), pybind11::arg("kv_norm_weight"),
                  pybind11::arg("positions"), pybind11::arg("cos_sin"), pybind11::arg("slots"),
                  pybind11::arg("q"), pybind11::arg("cache"), pybind11::arg("eps"));
    submodule.def("v41f_wv_o_proj", &v41f_wv_o_proj,
                  pybind11::arg("x"), pybind11::arg("wv_weight"), pybind11::arg("o_weight"),
                  pybind11::arg("y"));
}

} // namespace deep_gemm::experimental
