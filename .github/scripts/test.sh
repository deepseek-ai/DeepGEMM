#!/bin/bash

set -exou pipefail

pip install dist/*.whl
python -c "import deep_gemm; print(deep_gemm.__version__)"