#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <filesystem>

#include "../utils/exception.hpp"
#include "../utils/compatibility.hpp"

namespace deep_gemm {

// Lazy loading all driver symbols
static void* get_driver_handle() {
    static void* handle = nullptr;
    if (handle == nullptr) {
        handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        DG_HOST_ASSERT(handle != nullptr and "Failed to load CUDA driver `libcuda.so.1`");
    }
    return handle;
}

// Macro to define wrapper functions named `lazy_cu{API name}`
#define DECL_LAZY_CUDA_DRIVER_FUNCTION(name) \
template <typename... Args> \
static auto lazy_##name(Args&&... args) -> decltype(name(args...)) { \
    using FuncType = decltype(&(name)); \
    static FuncType func = nullptr; \
    if (func == nullptr) { \
        func = reinterpret_cast<FuncType>(dlsym(get_driver_handle(), #name)); \
        DG_HOST_ASSERT(func != nullptr and "Failed to load CUDA driver API"); \
    } \
    return func(std::forward<decltype(args)>(args)...); \
}

DECL_LAZY_CUDA_DRIVER_FUNCTION(cuGetErrorName);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuGetErrorString);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuFuncSetAttribute);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleLoad);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleUnload);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuModuleGetFunction);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLibraryLoadFromFile);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLibraryUnload);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuKernelGetFunction);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLaunchKernelEx);
DECL_LAZY_CUDA_DRIVER_FUNCTION(cuTensorMapEncodeTiled);

#if CUDART_VERSION >= 12080 and defined(DG_JIT_USE_RUNTIME_API)

// Use CUDA runtime API
using LibraryHandle = cudaLibrary_t;
using KernelHandle = cudaKernel_t;
using LaunchConfigHandle = cudaLaunchConfig_t;
using LaunchAttrHandle = cudaLaunchAttribute;

#define DG_CUDA_UNIFIED_CHECK DG_CUDA_RUNTIME_CHECK

static KernelHandle load_kernel(const std::filesystem::path& cubin_path, const std::string& func_name,
                                LibraryHandle *library_opt = nullptr) {
    LibraryHandle library;
    KernelHandle kernel{};
    DG_CUDA_RUNTIME_CHECK(cudaLibraryLoadFromFile(&library, cubin_path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
    DG_CUDA_RUNTIME_CHECK(cudaLibraryGetKernel(&kernel, library, func_name.c_str()));

    if (library_opt != nullptr)
        *library_opt = library;
    return kernel;
}

static void unload_library(const LibraryHandle& library) {
    const auto error = cudaLibraryUnload(library);
    DG_HOST_ASSERT(error == cudaSuccess or error == cudaErrorCudartUnloading);
}

static LaunchConfigHandle construct_launch_config(const KernelHandle& kernel,
                                                  const cudaStream_t& stream, const int& smem_size,
                                                  const dim3& grid_dim, const dim3& block_dim, const int& cluster_dim, const bool& enable_pdl) {
    if (smem_size > 0)
        DG_CUDA_RUNTIME_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    LaunchConfigHandle config;
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;

    // Create attributes
    // NOTES: must use `static` or the `attr` will be deconstructed
    static LaunchAttrHandle attrs[2];
    config.numAttrs = 0;
    config.attrs = attrs;

    // Cluster size
    if (cluster_dim > 1) {
        auto& attr = attrs[config.numAttrs ++];
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim = {static_cast<unsigned>(cluster_dim), 1, 1};
    }

    // Dependent kernel launch
    if (enable_pdl) {
        auto& attr = attrs[config.numAttrs ++];
        attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;
    }

    return config;
}

template<typename... ActTypes>
static auto launch_kernel(const KernelHandle& kernel, const LaunchConfigHandle& config, ActTypes&&... args) {
    void *ptr_args[] = { &args... };
    return cudaLaunchKernelExC(&config, kernel, ptr_args);
}

#else

// Use CUDA driver API
using KernelHandle = CUfunction;
using LaunchConfigHandle = CUlaunchConfig;
using LaunchAttrHandle = CUlaunchAttribute;

// `cuLibraryEnumerateKernels` is supported since CUDA Driver API 12.4
#if CUDA_VERSION >= 12040
    #define DG_JIT_USE_LIBRARY_ENUM_KERNELS
    DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLibraryGetKernelCount);
    DECL_LAZY_CUDA_DRIVER_FUNCTION(cuLibraryEnumerateKernels);
    using LibraryHandle = CUlibrary;
#else
    using LibraryHandle = CUmodule;
#endif

#define DG_CUDA_UNIFIED_CHECK DG_CUDA_DRIVER_CHECK

static KernelHandle load_kernel(const std::filesystem::path& cubin_path, const std::string& func_name,
                                LibraryHandle *library_opt = nullptr) {
    LibraryHandle library;
    KernelHandle kernel;

#ifdef DG_JIT_USE_LIBRARY_ENUM_KERNELS
    DG_CUDA_DRIVER_CHECK(lazy_cuLibraryLoadFromFile(&library, cubin_path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0));
    unsigned int num_kernels;
    DG_CUDA_DRIVER_CHECK(lazy_cuLibraryGetKernelCount(&num_kernels, library));
    if (num_kernels != 1) {
        const auto dir_path = cubin_path.parent_path();
        printf("Corrupted JIT cache directory (expected 1 kernel, found %u): %s, "
               "please run `rm -rf %s` and restart your task.\n",
               num_kernels, dir_path.c_str(), dir_path.c_str());
        DG_HOST_ASSERT(false and "Corrupted JIT cache directory");
    }

    CUkernel cu_kernel;
    DG_CUDA_DRIVER_CHECK(lazy_cuLibraryEnumerateKernels(&cu_kernel, 1, library));
    DG_CUDA_DRIVER_CHECK(lazy_cuKernelGetFunction(&kernel, cu_kernel));
#else
    DG_CUDA_DRIVER_CHECK(lazy_cuModuleLoad(&library, cubin_path.c_str()));
    DG_CUDA_DRIVER_CHECK(lazy_cuModuleGetFunction(&kernel, library, func_name.c_str()));
#endif

    if (library_opt != nullptr)
        *library_opt = library;
    return kernel;
}

static void unload_library(const LibraryHandle& library) {
#ifdef DG_JIT_USE_LIBRARY_ENUM_KERNELS
    const auto error = lazy_cuLibraryUnload(library);
#else
    const auto error = lazy_cuModuleUnload(library);
#endif
    DG_HOST_ASSERT(error == CUDA_SUCCESS or error == CUDA_ERROR_DEINITIALIZED);
}

static LaunchConfigHandle construct_launch_config(const KernelHandle& kernel,
                                                 const cudaStream_t& stream, const int& smem_size,
                                                 const dim3& grid_dim, const dim3& block_dim, const int& cluster_dim, const bool& enable_pdl) {
    if (smem_size > 0)
        DG_CUDA_DRIVER_CHECK(lazy_cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size));

    LaunchConfigHandle config;
    config.gridDimX = grid_dim.x;
    config.gridDimY = grid_dim.y;
    config.gridDimZ = grid_dim.z;
    config.blockDimX = block_dim.x;
    config.blockDimY = block_dim.y;
    config.blockDimZ = block_dim.z;
    config.sharedMemBytes = smem_size;
    config.hStream = stream;
    
    // Create attributes
    // NOTES: must use `static` or the `attr` will be deconstructed
    static LaunchAttrHandle attrs[2];
    config.numAttrs = 0;
    config.attrs = attrs;

    // Cluster size
    if (cluster_dim > 1) {
        auto& attr = attrs[config.numAttrs ++];
        attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        attr.value.clusterDim.x = static_cast<unsigned>(cluster_dim);
        attr.value.clusterDim.y = 1;
        attr.value.clusterDim.z = 1;
    }

    // Dependent kernel launch
    if (enable_pdl) {
        auto& attr = attrs[config.numAttrs ++];
        attr.id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        attr.value.programmaticStreamSerializationAllowed = 1;
    }

    return config;
}

template<typename... ActTypes>
static auto launch_kernel(const KernelHandle& kernel, const LaunchConfigHandle& config, ActTypes&&... args) {
    void *ptr_args[] = { &args... };
    return lazy_cuLaunchKernelEx(&config, kernel, ptr_args, nullptr);
}
#endif

} // namespace deep_gemm
