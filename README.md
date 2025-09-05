# AdaptiveGEMM
<p align="center">
  <a href="https://arxiv.org/abs/2508.16584" target="_blank">
    <img src="https://img.shields.io/badge/Paper-ArXiv%3A2508.16584-blue?logo=arxiv" alt="Paper Link" />
  </a>
</p>
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
git clone --recursive git@github.com:InternLM/AdaptiveGemm.git

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

## Citation

If you find our work useful, please cite:

```bibtex
@misc{su2025tmaadaptivefp8groupedgemm,
  title={TMA-Adaptive FP8 Grouped GEMM: Eliminating Padding Requirements in Low-Precision Training and Inference on Hopper}, 
  author={Zhongling Su and Rong Fu and Weihan Cao and Jianfei Gao and Minxi Jin and Zhilin Pei and Hui Wang},
  year={2025},
  eprint={2508.16584},
  archivePrefix={arXiv},
  primaryClass={cs.AR},
  url={https://arxiv.org/abs/2508.16584}, 
}
```
## Original deepgemm
To get detailed information of original deepgeem, please refer to https://github.com/deepseek-ai/DeepGEMM