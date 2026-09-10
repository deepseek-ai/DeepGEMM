#include <pybind11/pybind11.h>
#include <torch/python.h>

#include <deep_jit/backend/cuda/backend.hpp>
#include <deep_jit/python_api.hpp>

#include "apis/config.hpp"
#include "apis/attention.hpp"
#include "apis/einsum.hpp"
#include "apis/hyperconnection.hpp"
#include "apis/gemm.hpp"
#include "apis/layout.hpp"
#include "apis/mega_moe.hpp"
#include "apis/mega_mhc.hpp"
#include "apis/mega_gate.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

// ReSharper disable once CppParameterMayBeConstPtrOrRef
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepGEMM C++ library";

    // Register JIT objects
    deep_jit::register_python_api(m, deep_gemm::jit);

    // Register config APIs
    deep_gemm::config::register_apis(m);

    // Register kernels
    // TODO: make SM80 incompatible issues raise errors
    deep_gemm::attention::register_apis(m);
    deep_gemm::einsum::register_apis(m);
    deep_gemm::hyperconnection::register_apis(m);
    deep_gemm::gemm::register_apis(m);
    deep_gemm::layout::register_apis(m);
    deep_gemm::mega::register_apis(m);
    deep_gemm::mega_mhc::register_apis(m);
    deep_gemm::mega_gate::register_apis(m);
}
