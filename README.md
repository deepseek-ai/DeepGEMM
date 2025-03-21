# DeepGEMM
This repository is based on the original DeepGEMM. In addition to that, we have added two key features:

* **Adaptation to Various Lengths of Group M**: We have enhanced the repository to support different lengths of group M, providing more flexibility for diverse use cases.
* **Support for Group K GEMM**: We have also added support for Group K GEMM operations, expanding the functionality of the original DeepGEMM. NOTE: k in each group should be padded to 128x.

## Quick start

### Requirements

- Hopper architecture GPUs, `sm_90a` must be supported
- Python 3.8 or above
- CUDA 12.3 or above
  - **But we highly recommend 12.8 or above for the best performance**
- PyTorch 2.1 or above
- CUTLASS 3.6 or above (could be cloned by Git submodule)

### Development

```bash
# Submodule must be cloned
git clone --recursive https://github.com/InternLM/AdaptiveGemm.git

# Make symbolic links for third-party (CUTLASS and CuTe) include directories
python setup.py develop

# Test JIT compilation
python tests/test_jit.py

# Test all GEMM implements (normal, contiguous-grouped and masked-grouped)
python tests/test_varlen_groupm.py
```

### Installation

```bash
python setup.py install
```

## Original deepgemm
To get detailed information of original deepgeem, please refer to https://github.com/deepseek-ai/DeepGEMM