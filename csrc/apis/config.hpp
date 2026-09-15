#pragma once

#include "../jit_kernels/heuristics/runtime.hpp"
#include "../runtime/runtime.hpp"

namespace deep_gemm::config {

static void register_apis(pybind11::module_& m) {
    m.def("init", [&](const std::string& library_root_path) {
        init_jit(library_root_path);
    });
    m.def("set_num_sms", [&](const int& new_num_sms) {
        runtime->set_num_sms(new_num_sms);
    });
    m.def("get_num_sms", [&]() {
       return runtime->get_num_sms();
    });
    m.def("set_tc_util", [&](const int& new_tc_util) {
        runtime->set_tc_util(new_tc_util);
    });
    m.def("get_tc_util", [&]() {
        return runtime->get_tc_util();
    });
    m.def("set_pdl", [](const bool& new_enable_pdl) {
        jit->default_launch_options.enable_pdl = new_enable_pdl;
    });
    m.def("get_pdl", []() {
        return *jit->default_launch_options.enable_pdl;
    });
    m.def("use_deterministic_algorithms", [&](const bool enabled) {
        heuristics_runtime->use_deterministic_algorithms(enabled);
    });
    m.def("set_ignore_compile_dims", [&](const bool& new_value) {
        heuristics_runtime->set_ignore_compile_dims(new_value);
    });
    m.def("set_block_size_multiple_of", [&](const std::variant<int, std::tuple<int, int>>& new_value) {
        if (std::holds_alternative<int>(new_value)) {
            auto x = std::get<int>(new_value);
            heuristics_runtime->set_block_size_multiple_of(x, x);
        } else {
            auto [x, y] = std::get<std::tuple<int, int>>(new_value);
            heuristics_runtime->set_block_size_multiple_of(x, y);
        }
    });
}

} // namespace deep_gemm::config
