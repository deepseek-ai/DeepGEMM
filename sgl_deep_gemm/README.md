
## Introduction

sgl-deep-gemm is a pypi package built from SGLang's customized branch of DeepGemm. Comparing with origina DeepGemm, it supports the following features to better support SGLang:
1. ABI support: with the help of tvm-ffi wrappers, a single wheel can run on different python versions.
2. pypi support: easy installation with `pip install sgl-deep-gemm`. No need to manually search for wheel links.
3. Fast iteration: add custom kernels and bump versions at no time.

## Usage
To build it locally, run `bash build_sgl_deep_gemm.sh`, then pip install the wheel generated under `dist`.

To release a new set of wheels, please contact SGLang team and run the [release workflow](https://github.com/sgl-project/sglang/actions/workflows/release-whl-deepgemm.yml) under SGLang repo

For each major version release (0.X.Y -> 0.(X+1).0), a new branch should be created (release/v0.(X+1).0) for stability purpose.

For any incoming pull requests, it should be rebased upon `dev` branch.
