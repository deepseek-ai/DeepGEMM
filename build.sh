# Change current directory into project root
original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

# Link CUTLASS includes
ln -sf $script_dir/third-party/cutlass/include/cutlass deep_gemm/include
ln -sf $script_dir/third-party/cutlass/include/cute deep_gemm/include

# Remove old dist file, build files, and install
rm -rf build dist
rm -rf *.egg-info
python setup.py bdist_wheel

# Open users' original directory
cd "$original_dir"
