#pragma once

#include <map>

#include <cuda.h>
#include <c10/cuda/CUDAFunctions.h>

#include <deep_gemm/common/types.cuh>
#include <deep_jit/backend/cuda/driver.hpp>

#include "../utils/exception.hpp"
#include "mlopart.hpp"

namespace deep_gemm {

// Compatability layer for locality domain APIs in CUDA 13.4
class LocalityDomainAllocator {
    struct Mapping {
        size_t num_bytes;
        int domain_idx;
        CUmemGenericAllocationHandle handle;
    };

    std::map<CUdeviceptr, size_t> reservations;
    std::map<CUdeviceptr, Mapping> mappings;

    static CUmemLocation get_device_location() {
        CUmemLocation location{};
        location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        location.id = c10::cuda::current_device();
        return location;
    }

    CUmemGenericAllocationHandle create(const size_t& num_bytes, const int& domain_idx) {
        // TODO: `cuMemCreate` on the location `{CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN, {device, domain_idx}}` with CUDA 13.4
        DG_HOST_ASSERT(0 <= domain_idx and domain_idx < get_num_locality_domains());
        return mlopart::create_memory(num_bytes, domain_idx);
    }

public:
    LocalityDomainAllocator() = default;
    LocalityDomainAllocator(const LocalityDomainAllocator&) = delete;
    LocalityDomainAllocator& operator=(const LocalityDomainAllocator&) = delete;

    // Outstanding reservations go with the allocator
    ~LocalityDomainAllocator() {
        while (not reservations.empty())
            release(reinterpret_cast<void*>(reservations.begin()->first));
    }

    static int get_num_locality_domains() {
        // TODO: `CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT` with CUDA 13.4
        return kNumDeviceLocalityDomains;
    }

    static size_t get_granularity() {
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location = get_device_location();
        size_t granularity = 0;
        DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        return granularity;
    }

    void* reserve(const size_t& num_bytes) {
        CUdeviceptr address = 0;
        DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemAddressReserve(&address, num_bytes, get_granularity(), 0, 0));
        reservations.emplace(address, num_bytes);
        return reinterpret_cast<void*>(address);
    }

    void map(void* ptr, const size_t& num_bytes, const int& domain_idx) {
        const auto address = reinterpret_cast<CUdeviceptr>(ptr);
        const auto handle = create(num_bytes, domain_idx);
        DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemMap(address, num_bytes, 0, handle, 0));
        CUmemAccessDesc access{};
        access.location = get_device_location();
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemSetAccess(address, num_bytes, &access, 1));
        mappings.emplace(address, Mapping(num_bytes, domain_idx, handle));
    }

    void release(void* ptr) {
        const auto address = reinterpret_cast<CUdeviceptr>(ptr);
        const auto num_bytes = reservations.at(address);
        reservations.erase(address);
        try {
            DG_CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());
            for (auto it = mappings.lower_bound(address); it != mappings.end() and it->first < address + num_bytes; it = mappings.erase(it)) {
                DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemUnmap(it->first, it->second.num_bytes));
                DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemRelease(it->second.handle));
            }
            DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemAddressFree(address, num_bytes));
        } catch (...) {
        }
    }

    int get_locality_domain(const void* ptr, const size_t& num_bytes) const {
        const auto address = reinterpret_cast<CUdeviceptr>(ptr);
        const auto it = mappings.upper_bound(address);
        if (it == mappings.begin())
            return -1;
        const auto& [start, mapping] = *std::prev(it);
        return address + num_bytes <= start + mapping.num_bytes ? mapping.domain_idx : -1;
    }
};

inline LocalityDomainAllocator locality_domain_allocator;

}  // namespace deep_gemm
