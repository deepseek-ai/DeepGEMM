#!/bin/bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Change current directory into project root
original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

# Cleanup function
cleanup() {
    cd "$original_dir"
}
trap cleanup EXIT

echo -e "${GREEN}Setting up DeepGEMM development environment...${NC}"

# Check for required tools
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python is not installed or not in PATH${NC}" >&2
    exit 1
fi

# Check for CUDA
if [ -z "${CUDA_HOME:-}" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
        echo -e "${YELLOW}Warning: CUDA_HOME not set, using /usr/local/cuda${NC}"
    else
        echo -e "${RED}Error: CUDA_HOME is not set and /usr/local/cuda not found${NC}" >&2
        exit 1
    fi
fi

# Check for submodules
if [ ! -d "third-party/cutlass/include" ]; then
    echo -e "${YELLOW}Submodules not initialized. Initializing...${NC}"
    git submodule update --init --recursive
fi

# Link CUTLASS includes
echo "Linking CUTLASS includes..."
ln -sf "$script_dir/third-party/cutlass/include/cutlass" deep_gemm/include
ln -sf "$script_dir/third-party/cutlass/include/cute" deep_gemm/include

# Remove old dist file, build files, and build
echo "Cleaning previous build artifacts..."
rm -rf build dist
rm -rf *.egg-info

echo "Building DeepGEMM..."
python setup.py build

# Find the .so file in build directory and create symlink in current directory
so_file=$(find build -name "*.so" -type f | head -n 1)
if [ -n "$so_file" ]; then
    ln -sf "../$so_file" deep_gemm/
    echo -e "${GREEN}Successfully linked: $so_file${NC}"
else
    echo -e "${RED}Error: No .so file found in build directory${NC}" >&2
    exit 1
fi

echo -e "${GREEN}Development environment setup complete!${NC}"
echo ""
echo "You can now run tests with:"
echo "  python tests/test_core.py"
echo "  python tests/test_fp8.py"
echo "  python tests/test_bf16.py"
