#pragma once

#include <string>
#include <torch/version.h>
#include <hggc_runtime_api.h>
#include "math.hpp"
#include "system.hpp"
#include "../jit/device_runtime.hpp"

namespace deep_gemm {

// The SM budget is owned by DeviceRuntime (jit/device_runtime.hpp) so that the pybind-exposed
// set_num_sms() actually drives the kernels. This free function is kept as a thin alias because
// every GEMM impl calls get_num_sms() unqualified; it now routes to that single source of truth.
int get_num_sms() {
    return device_runtime->get_num_sms();
}

bool is_ppu1v5_device() {
    hggcDeviceProp device_prop;
    hggcGetDeviceProperties(&device_prop, 0);
    if (device_prop.major == 8 && device_prop.minor == 9) {
        return true;
    } else {
        return false;
    }
}

int get_sm_count() {
    hggcDeviceProp device_prop;
    hggcGetDeviceProperties(&device_prop, 0);
    return device_prop.multiProcessorCount;
}

std::unordered_map<std::string, int> get_extra_info(int m = 0, int n = 0, int k = 0, int dtype = 1,
                                                    const std::string& api_type = "dense") {
    std::unordered_map<std::string, int> extra_info;
    extra_info["use_multistage_on_N"] = get_env<int>("DG_USE_MULTISTAGE_ON_N", 0);
    extra_info["use_moe_dynamic_tile"] = get_env<int>("DG_USE_MOE_DYNAMIC_TILE", 0);

    return extra_info;
}

static dim3 get_grid_shape(int sm_count) {
    return dim3(sm_count, 1, 1);
}

int32_t next_power_of_two(uint32_t n) {
    if (n == 0)
        return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

} // namespace deep_gemm
