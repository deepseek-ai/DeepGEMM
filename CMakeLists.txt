# NOTES: current just for CMake-based IDE (e.g. CLion) indexing, the real compilation is done via JIT
# TODO: add CUDA utils' library via CMake
cmake_minimum_required(VERSION 3.10)
project(deep_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)

file(WRITE ${CMAKE_BINARY_DIR}/test_cuda.cu "extern \"C\" __global__ void testKernel() { }")
execute_process(
        COMMAND ${CUDA_NVCC_EXECUTABLE} ${CMAKE_CUDA_FLAGS} -gencode arch=compute_90a,code=sm_90a -gencode arch=compute_120,code=sm_120 -o ${CMAKE_BINARY_DIR}/test_cuda.o -c ${CMAKE_BINARY_DIR}/test_cuda.cu
        RESULT_VARIABLE NVCC_RESULT
        OUTPUT_VARIABLE NVCC_OUTPUT
        ERROR_VARIABLE NVCC_ERROR_OUTPUT
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

if (NVCC_RESULT EQUAL "0")
    set(NVCC_SUPPORTS_SM90 TRUE)
    message(STATUS "NVCC supports SM90")
else()
    message(STATUS "NVCC does not support SM90")
endif()

if (NVCC_SUPPORTS_SM90)
    set(TORCH_CUDA_ARCH_LIST "8.6;9.0a" CACHE STRING "Add arch tag 90a to NVCC" FORCE) # TODO: Check if 9.0a is correct for sm_90a, it might be just 9.0
    list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_90a,code=sm_90a")
    # Add Blackwell support if NVCC supports it (determined by the test_cuda.cu compilation)
    list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_120,code=sm_120")
    set(TORCH_CUDA_ARCH_LIST "${TORCH_CUDA_ARCH_LIST};12.0" CACHE STRING "Add arch tag 120 to NVCC" FORCE) # TODO: Check if 12.0 is correct for sm_120
endif()
find_package(Torch REQUIRED)

include_directories(deep_gemm/include third-party/cutlass/include third-party/cutlass/tools/util/include)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
link_directories(${TORCH_INSTALL_PREFIX}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -fPIC -DNDEBUG")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -std=c++17 -DNDEBUG --ptxas-options=--register-usage-level=10")

cuda_add_library(example_gemm STATIC indexing/main.cu)
