#pragma once

#include <memory>

#include <pybind11/pybind11.h>

#include "../jit_kernels/impls/epilogue_class.hpp"

namespace deep_gemm::epilogue_class {

// User-facing epilogue classes for the `epilogue` argument of the GEMM APIs, as `deep_gemm.epilogue.X`
// NOTES: `alpha` and the FP8 `(d, sfd)` output pair are the shorthand for `Alpha` and `FP8Quantization`
static void register_apis(pybind11::module_& m) {
    const auto submodule = m.def_submodule("epilogue", "GEMM epilogue classes");
    pybind11::class_<EpilogueClass, std::shared_ptr<EpilogueClass>>(submodule, "Epilogue");
    pybind11::class_<IdentityEpilogue, EpilogueClass, std::shared_ptr<IdentityEpilogue>>(submodule, "Identity")
        .def(pybind11::init<>());
    pybind11::class_<AlphaEpilogue, EpilogueClass, std::shared_ptr<AlphaEpilogue>>(submodule, "Alpha")
        .def(pybind11::init<const float&>(), pybind11::arg("alpha"));
    pybind11::class_<FP8QuantizationEpilogue, EpilogueClass, std::shared_ptr<FP8QuantizationEpilogue>>(submodule, "FP8Quantization")
        .def(pybind11::init<const torch::Tensor&>(), pybind11::arg("sfd"));
    pybind11::class_<BF16StochasticRoundingEpilogue, EpilogueClass,
                     std::shared_ptr<BF16StochasticRoundingEpilogue>>(submodule, "BF16StochasticRounding")
        .def(pybind11::init<>());
}

} // namespace deep_gemm::epilogue_class
