# Change current directory into project root
original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

# Link includes for JIT (matching setup.py make_jit_include_symlinks)
# Ensure include dir is a real directory, not a symlink (prevent symlink traversal)
[ -L deep_gemm/include/actlize_v1.0.0 ] && rm -f deep_gemm/include/actlize_v1.0.0
mkdir -p deep_gemm/include/actlize_v1.0.0
ln -sfn $script_dir/third-party/actlize_v1.0.0/include/cute deep_gemm/include/actlize_v1.0.0/cute
ln -sfn $script_dir/third-party/actlize_v1.0.0/include/cutlass deep_gemm/include/actlize_v1.0.0/cutlass
ln -sfn $script_dir/third-party/actlize_v1.0.0/include/ppu_include.hpp deep_gemm/include/actlize_v1.0.0/ppu_include.hpp
ln -sfn $script_dir/third-party/actlize_v1.0.0/include/accutlass.hpp deep_gemm/include/actlize_v1.0.0/accutlass.hpp
ln -sfn $script_dir/third-party/actlize_v1.0.0/tools deep_gemm/include/actlize_v1.0.0/tools

# Remove old dist file, build files, and build
rm -rf build dist
rm -rf *.egg-info
python setup.py build

# Find the .so file in build directory and create symlink in current directory
so_file=$(find build -name "*.so" -type f | head -n 1)
if [ -n "$so_file" ]; then
    ln -sfn "../$so_file" deep_gemm/
else
    echo "Error: No SO file found in build directory" >&2
    exit 1
fi

# Open users' original directory
cd "$original_dir"
