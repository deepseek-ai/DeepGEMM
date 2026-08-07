# Change current directory into project root
original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

# Link CUTLASS includes
ln -sf $script_dir/third-party/cutlass/include/cutlass deep_gemm/include
ln -sf $script_dir/third-party/cutlass/include/cute deep_gemm/include

# Remove old dist file, build files, and build
rm -rf build dist
rm -rf *.egg-info
python setup.py build

# Find the Python extension specifically; the build also contains the
# Torch-independent native runtime library.
so_file=$(find build -name "_C*.so" -type f -print -quit)
if [ -n "$so_file" ]; then
    ln -sf "../$so_file" deep_gemm/
else
    echo "Error: No SO file found in build directory" >&2
    exit 1
fi

# Open users' original directory
cd "$original_dir"
