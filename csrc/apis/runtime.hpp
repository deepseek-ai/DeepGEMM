#pragma once

#if DG_TENSORMAP_COMPATIBLE
#include "../jit/compiler.hpp"
#include "../jit/kernel_runtime.hpp"
#endif
#include "../jit/device_runtime.hpp"
#include "../jit_kernels/heuristics/runtime.hpp"

#include <torch/library.h>

namespace deep_gemm::torch_registration {

static void set_num_sms(const int64_t& new_num_sms) {
    device_runtime->set_num_sms(static_cast<int>(new_num_sms));
}

static int64_t get_num_sms() {
    return device_runtime->get_num_sms();
}

static void set_tc_util(const int64_t& new_tc_util) {
    device_runtime->set_tc_util(static_cast<int>(new_tc_util));
}

static int64_t get_tc_util() {
    return device_runtime->get_tc_util();
}

static void set_pdl(const bool& new_enable_pdl) {
    device_runtime->set_pdl(new_enable_pdl);
}

static bool get_pdl() {
    return device_runtime->get_pdl();
}

static void set_ignore_compile_dims(const bool& new_value) {
    heuristics_runtime->set_ignore_compile_dims(new_value);
}

static void set_block_size_multiple_of(const std::vector<int64_t>& value) {
    if (value.size() == 1) {
        const int v = static_cast<int>(value[0]);
        heuristics_runtime->set_block_size_multiple_of(v, v);
    } else {
        DG_HOST_ASSERT(value.size() == 2);
        heuristics_runtime->set_block_size_multiple_of(
            static_cast<int>(value[0]), static_cast<int>(value[1]));
    }
}

static void init(const std::string& library_root_path,
                 const std::string& cuda_home_path_by_python) {
#if DG_TENSORMAP_COMPATIBLE
        Compiler::prepare_init(library_root_path, cuda_home_path_by_python);
        KernelRuntime::prepare_init(cuda_home_path_by_python);
        IncludeParser::prepare_init(library_root_path);
#endif
}

}  // namespace deep_gemm::torch_registration

TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
    m.def("set_num_sms(int new_num_sms) -> ()", TORCH_FN(deep_gemm::torch_registration::set_num_sms));
    m.def("get_num_sms() -> int", TORCH_FN(deep_gemm::torch_registration::get_num_sms));
    m.def("set_tc_util(int new_tc_util) -> ()", TORCH_FN(deep_gemm::torch_registration::set_tc_util));
    m.def("get_tc_util() -> int", TORCH_FN(deep_gemm::torch_registration::get_tc_util));
    m.def("set_pdl(bool new_enable_pdl) -> ()", TORCH_FN(deep_gemm::torch_registration::set_pdl));
    m.def("get_pdl() -> bool", TORCH_FN(deep_gemm::torch_registration::get_pdl));
    m.def("set_ignore_compile_dims(bool new_value) -> ()", TORCH_FN(deep_gemm::torch_registration::set_ignore_compile_dims));
    m.def("set_block_size_multiple_of(int[] value) -> ()", TORCH_FN(deep_gemm::torch_registration::set_block_size_multiple_of));
    m.def("init(str library_root_path, str cuda_home_path_by_python) -> ()", TORCH_FN(deep_gemm::torch_registration::init));
}
