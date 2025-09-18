#pragma once

#include <ATen/cuda/CUDAContext.h>

#include "../utils/exception.hpp"
#include "../utils/lazy_init.hpp"

namespace deep_gemm {

class DeviceRuntime {
    int num_sms = 0, tc_util = 0;
    std::shared_ptr<cudaDeviceProp> cached_prop;

public:
    explicit DeviceRuntime() = default;

    std::shared_ptr<cudaDeviceProp> get_prop() {
        if (cached_prop == nullptr)
            cached_prop = std::make_shared<cudaDeviceProp>(*at::cuda::getCurrentDeviceProperties());
        return cached_prop;
    }

    std::pair<int, int> get_arch_pair() {
        const auto prop = get_prop();
        return {prop->major, prop->minor};
    }

    std::string get_arch(const bool& number_only = false,
                         const bool& support_arch_family = false) {
        const auto& [major, minor] = get_arch_pair();
        if (major == 10 and minor != 1) {
            if (number_only)
                return "100";
            return support_arch_family ? "100f" : "100a";
        }
        return std::to_string(major * 10 + minor) + (number_only ? "" : "a");
    }

    int get_arch_major() {
        return get_arch_pair().first;
    }

    void set_num_sms(const int& new_num_sms) {
        DG_HOST_ASSERT(0 <= new_num_sms and new_num_sms <= get_prop()->multiProcessorCount);
        num_sms = new_num_sms;
    }

    int get_num_sms() {
        if (num_sms == 0)
            num_sms = get_prop()->multiProcessorCount;
        return num_sms;
    }

    void set_tc_util(const int& new_tc_util) {
        DG_HOST_ASSERT(0 <= new_tc_util and new_tc_util <= 100);
        tc_util = new_tc_util;
    }

    int get_tc_util() const {
        return tc_util == 0 ? 100 : tc_util;
    }
};

static auto device_runtime = LazyInit<DeviceRuntime>([](){ return std::make_shared<DeviceRuntime>(); });

} // namespace deep_gemm
