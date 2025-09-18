#!/bin/bash

echo "CUTLASS TMA Performance Test Suite"
echo "=================================="

# Check if we're in the correct directory
if [ ! -f "cutlass_tma_performance_test.cu" ]; then
    echo "Error: Please run this script from the tests directory"
    exit 1
fi

# Check for CUDA and GPU
echo "Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA compiler (nvcc) not found"
    exit 1
fi

echo "Checking GPU capabilities..."
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits

# Build the test
echo ""
echo "Building CUTLASS TMA performance test..."
cd ..
mkdir -p build
cd build

# Configure with CMake
cmake -DCMAKE_BUILD_TYPE=Release ../tests

# Build the specific test
make cutlass_tma_performance_test -j$(nproc)

if [ $? -ne 0 ]; then
    echo "Error: Build failed"
    exit 1
fi

echo ""
echo "Build successful!"
echo ""

# Run the test
echo "Running CUTLASS TMA performance test..."
echo "======================================="
./cutlass_tma_performance_test

echo ""
echo "Test completed!"