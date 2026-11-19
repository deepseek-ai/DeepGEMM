#pragma once

#include <unistd.h>

#include <cuda.h>
#include <pybind11/pybind11.h>
#include <deep_jit/backend/cuda/driver.hpp>

#include "jit.hpp"

namespace deep_gemm::mlopart {

// We have to use IPC to allocate localized memory via MLOPart prior to CUDA 13.4
// For convenience this is done in Python
static pybind11::module_ get_module() {
    return pybind11::module_::import("deep_gemm.utils.mlopart");
}

static pybind11::bytes get_device_uuid() {
    const auto& uuid = jit->device.get_prop().uuid.bytes;
    return pybind11::bytes(uuid, sizeof(uuid));
}

static bool is_available() {
    return get_module().attr("is_available")(get_device_uuid()).cast<bool>();
}

static CUmemGenericAllocationHandle create_memory(const size_t& num_bytes, const int& domain_idx) {
    const auto fd = get_module().attr("create_memory")(get_device_uuid(), num_bytes, domain_idx).cast<int>();
    CUmemGenericAllocationHandle handle = 0;
    DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemImportFromShareableHandle(
        &handle, reinterpret_cast<void*>(static_cast<intptr_t>(fd)), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    close(fd);
    return handle;
}

static void release() {
    get_module().attr("release")();
}

}  // namespace deep_gemm::mlopart
