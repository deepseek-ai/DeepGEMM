#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "apis/gemm.hpp"
#include "apis/layout.hpp"
#include "apis/runtime.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME deep_gemm_cpp
#endif

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepGEMM C++ library";

    deep_gemm::gemm::register_apis(m);
    deep_gemm::layout::register_apis(m);
    deep_gemm::runtime::register_apis(m);
}
