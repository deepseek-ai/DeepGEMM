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

echo -e "${GREEN}Installing DeepGEMM...${NC}"

# Check for required tools
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python is not installed or not in PATH${NC}" >&2
    exit 1
fi

if ! command -v pip &> /dev/null; then
    echo -e "${RED}Error: pip is not installed or not in PATH${NC}" >&2
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

# Remove old dist file, build files
echo "Cleaning previous build artifacts..."
rm -rf build dist
rm -rf *.egg-info

# Build wheel
echo "Building wheel..."
python setup.py bdist_wheel

# Find and install wheel
wheel_file=$(find dist -name "*.whl" -type f | head -n 1)
if [ -n "$wheel_file" ]; then
    echo "Installing $wheel_file..."
    pip install "$wheel_file" --force-reinstall
    echo -e "${GREEN}DeepGEMM installed successfully!${NC}"
else
    echo -e "${RED}Error: No wheel file found in dist directory${NC}" >&2
    exit 1
fi

echo ""
echo "You can verify the installation with:"
echo "  python -c \"import deep_gemm; print(deep_gemm.__path__)\""
