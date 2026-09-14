#pragma once

// Register selected DeepGEMM host APIs as torch dispatcher ops so external
// C++/Python callers can invoke them via `torch.ops.deep_gemm.*` without
// including DeepGEMM headers or linking against DeepGEMM. The schemas are
// compiled into DeepGEMM's `_C` extension next to the pybind bindings, so the
// ops become available as soon as `import deep_gemm` loads `_C`.
//
// Threading: these dispatcher entries may be called from pure C++ threads that
// do not hold the GIL, unlike the pybind bindings. They ultimately touch the
// shared JIT state (`jit->compile(...)`), so concurrent first-time compilation
// of new shapes relies on DeepJIT's own thread-safety. Callers that compile new
// shapes concurrently from multiple threads should warm the cache up first.
//
// This header must be included by exactly one translation unit
// (`csrc/python_api.cpp`). Schema registration uses TORCH_LIBRARY_FRAGMENT so
// additional headers can register more `deep_gemm` ops in the same TU without a
// duplicate-namespace error.

#include <cstdint>
#include <limits>
#include <optional>

#include <torch/library.h>

#include "hyperconnection.hpp"  // deep_gemm::hyperconnection::tf32_hc_prenorm_gemm

namespace deep_gemm::torch_ops {

// Thin forwarder matching a torch-op-friendly signature: `d` and `sqr_sum` are
// written in place, `num_splits` is optional.
inline void tf32_hc_prenorm_gemm(const torch::Tensor& a, const torch::Tensor& b,
                                 torch::Tensor& d, torch::Tensor& sqr_sum,
                                 std::optional<int64_t> num_splits) {
    std::optional<int> ns = std::nullopt;
    if (num_splits.has_value()) {
        TORCH_CHECK(num_splits.value() >= 1 &&
                        num_splits.value() <= std::numeric_limits<int>::max(),
                    "num_splits out of range: ", num_splits.value());
        ns = static_cast<int>(num_splits.value());
    }
    deep_gemm::hyperconnection::tf32_hc_prenorm_gemm(a, b, d, sqr_sum, ns);
}

}  // namespace deep_gemm::torch_ops

// NOTE: this op has only a CUDA implementation (no Meta/fake kernel), so it is
// eager-only: tracing it through torch.compile / make_fx will fail with
// "not implemented for Meta".
//
// NOTE: use the handle name `lib` (not `m`) so the .pyi stub generator, which
// scans csrc for pybind-style `m.def(...)`, skips these dispatcher schemas.
TORCH_LIBRARY_FRAGMENT(deep_gemm, lib) {
    lib.def("tf32_hc_prenorm_gemm(Tensor a, Tensor b, Tensor(d!) d, "
            "Tensor(s!) sqr_sum, int? num_splits=None) -> ()");
}

TORCH_LIBRARY_IMPL(deep_gemm, CUDA, lib) {
    lib.impl("tf32_hc_prenorm_gemm", &deep_gemm::torch_ops::tf32_hc_prenorm_gemm);
}
