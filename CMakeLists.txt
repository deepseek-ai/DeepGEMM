# NOTES: current just for CMake-based IDE (e.g. CLion) indexing, the real compilation is done via JIT
cmake_minimum_required(VERSION 3.15)
project(deep_gemm LANGUAGES CXX CUDA)
set(CMAKE_VERBOSE_MAKEFILE ON)

option(DG_BUILD_PYTHON "Build the PyTorch/pybind Python extension" ON)
option(DG_BUILD_NATIVE_CORE "Build the standalone native-core shared library" OFF)
set(DG_NATIVE_INSTALL_DIR "deep_gemm/lib" CACHE STRING "Native-core install directory")
set(DG_NATIVE_CUDA_ARCH "" CACHE STRING "CUDA architecture override for native-core builds")

if(DG_NATIVE_CUDA_ARCH)
    set(CUDA_ARCH_LIST "${DG_NATIVE_CUDA_ARCH}" CACHE STRING
        "List of CUDA architectures to compile" FORCE)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -Wno-psabi -Wno-deprecated-declarations")
set(CUDA_SEPARABLE_COMPILATION ON)
list(APPEND CUDA_NVCC_FLAGS "-DENABLE_FAST_DEBUG")
list(APPEND CUDA_NVCC_FLAGS "-O3")
list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage")

set(USE_SYSTEM_NVTX on)
set(CUDA_ARCH_LIST "9.0" CACHE STRING "List of CUDA architectures to compile")
set(TORCH_CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")

find_package(CUDAToolkit REQUIRED)
if(DG_BUILD_PYTHON)
    find_package(pybind11 REQUIRED)
    find_package(Torch REQUIRED)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

include_directories(deep_gemm/include third-party/cutlass/include third-party/cutlass/tools/util/include third-party/fmt/include)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include/cccl)
if(DG_BUILD_PYTHON)
    include_directories(${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
    link_directories(${TORCH_INSTALL_PREFIX}/lib)
endif()
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)

if(DG_BUILD_PYTHON)
    # The main Python API entrance
    pybind11_add_module(_C csrc/python_api.cpp)
    target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} torch_python)
endif()

if(DG_BUILD_NATIVE_CORE)
    add_library(deep_gemm_native SHARED csrc/native_api.cpp)
    target_compile_features(deep_gemm_native PRIVATE cxx_std_17)
    target_include_directories(deep_gemm_native PRIVATE
        "${CUDA_TOOLKIT_ROOT_DIR}/include")
    target_link_libraries(deep_gemm_native PRIVATE
        CUDA::cudart CUDA::cublas CUDA::cublasLt)
    set_target_properties(deep_gemm_native PROPERTIES
        OUTPUT_NAME "deep_gemm_native"
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN")
    install(TARGETS deep_gemm_native
        LIBRARY DESTINATION "${DG_NATIVE_INSTALL_DIR}")
    install(FILES csrc/native_api.h
        DESTINATION "${DG_NATIVE_INSTALL_DIR}/include")
endif()

# Enable kernel code indexing with CMake-based IDEs. It follows the Python
# frontend because the indexed headers use torch::Tensor types.
if(DG_BUILD_PYTHON)
    cuda_add_library(deep_gemm_indexing_cuda STATIC csrc/indexing/main.cu)
endif()
