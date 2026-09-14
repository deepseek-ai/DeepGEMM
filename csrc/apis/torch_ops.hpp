#pragma once

// Register selected DeepGEMM host APIs as torch dispatcher ops so external
// C++/Python callers can invoke them via `torch.ops.deep_gemm.*` without
// including DeepGEMM headers or linking against DeepGEMM. The schemas are
// compiled into DeepGEMM's `_C` extension next to the pybind bindings, so the
// ops become available as soon as `import deep_gemm` loads `_C`.

#include <torch/library.h>

#include "hyperconnection.hpp"  // deep_gemm::hyperconnection::tf32_hc_prenorm_gemm

namespace deep_gemm::torch_ops {

// Thin forwarder matching a torch-op-friendly signature: `d` and `sqr_sum` are
// written in place, `num_splits` is optional.
inline void tf32_hc_prenorm_gemm(const at::Tensor& a, const at::Tensor& b,
                                 at::Tensor& d, at::Tensor& sqr_sum,
                                 std::optional<int64_t> num_splits) {
    std::optional<int> ns = num_splits.has_value()
        ? std::optional<int>(static_cast<int>(num_splits.value()))
        : std::nullopt;
    deep_gemm::hyperconnection::tf32_hc_prenorm_gemm(a, b, d, sqr_sum, ns);
}

}  // namespace deep_gemm::torch_ops

// NOTE: use the handle name `lib` (not `m`) so the .pyi stub generator, which
// scans csrc for pybind-style `m.def(...)`, skips these dispatcher schemas.
TORCH_LIBRARY(deep_gemm, lib) {
    lib.def("tf32_hc_prenorm_gemm(Tensor a, Tensor b, Tensor(d!) d, "
            "Tensor(s!) sqr_sum, int? num_splits=None) -> ()");
}

TORCH_LIBRARY_IMPL(deep_gemm, CUDA, lib) {
    lib.impl("tf32_hc_prenorm_gemm", &deep_gemm::torch_ops::tf32_hc_prenorm_gemm);
}
