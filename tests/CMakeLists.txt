cmake_minimum_required(VERSION 3.18)
project(TMA_Performance_Tests)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Enable CUDA language
enable_language(CUDA)

# Set CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 90)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/cutlass/tools/util/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_90,code=sm_90")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")


# CUTLASS TMA Performance Test (using actual CUTLASS TMA operations)

# Add the CUTLASS TMA performance test
add_executable(cutlass_tma_performance_test cutlass_tma_performance_test.cu)

# Set CUDA architecture for SM90+ (required for TMA)
set_property(TARGET cutlass_tma_performance_test PROPERTY CUDA_ARCHITECTURES 90)

# Set CUDA standard
set_property(TARGET cutlass_tma_performance_test PROPERTY CUDA_STANDARD 17)

# Add compile definitions for TMA support
target_compile_definitions(cutlass_tma_performance_test PRIVATE
    CUDA_12_0_SM90_FEATURES_SUPPORTED=1
)

# Add include directories
target_include_directories(cutlass_tma_performance_test PRIVATE
    ${CMAKE_SOURCE_DIR}/third-party/cutlass/include
    ${CMAKE_SOURCE_DIR}/third-party/cutlass/tools/util/include
)

# Link CUDA libraries (CUTLASS is header-only, no linking needed)
target_link_libraries(cutlass_tma_performance_test PRIVATE 
    CUDA::cudart
    CUDA::cuda_driver
)

# Compiler flags for optimal performance
target_compile_options(cutlass_tma_performance_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --extended-lambda
        --use_fast_math
        -Xptxas -v
    >
)