import os
import shutil
import subprocess
from pathlib import Path


def test_include_tree_hash_follows_directory_symlinks(tmp_path: Path):
    root = Path(__file__).parents[1]
    stubs = tmp_path / "stubs"
    (stubs / "torch").mkdir(parents=True)
    (stubs / "torch" / "version.h").write_text(
        "#define TORCH_VERSION_MAJOR 2\n#define TORCH_VERSION_MINOR 1\n"
    )
    (stubs / "cuda.h").write_text("#define CUDA_VERSION 12010\n")
    (stubs / "cuda_runtime.h").write_text("#define CUDART_VERSION 12020\n")
    (stubs / "cublasLt.h").write_text("using cublasStatus_t = int;\n")

    compiler = os.environ.get("CXX") or shutil.which("c++")
    assert compiler is not None
    binary = tmp_path / "test_include_parser"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            f"-I{stubs}",
            f"-I{root / 'third-party' / 'fmt' / 'include'}",
            str(Path(__file__).with_suffix(".cpp")),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([binary], check=True)
