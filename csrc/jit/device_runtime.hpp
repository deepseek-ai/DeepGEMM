#pragma once

#include <ATen/cuda/CUDAContext.h>

#include "../utils/exception.hpp"

namespace deep_gemm {

class DeviceRuntime {
    int num_sms = 0;
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

    int get_arch() {
        const auto& [major, minor] = get_arch_pair();
        return major * 10 + minor;
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
};

static auto device_runtime = std::make_shared<DeviceRuntime>();

} // namespace deep_gemm
