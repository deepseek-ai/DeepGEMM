import pytest
import random
import torch

import deep_gemm
from deep_gemm.testing import get_arch_major, test_filter
from sm120_exercise import exercise_sm120_dense_fp8_fp4, sm120_dense_quantized


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('fmt', ((False, True), (True, False)))
@pytest.mark.parametrize('k', (128, 256, 384, 512, 640, 768, 896))
@pytest.mark.parametrize('grans,alpha,c_mode', (
    ((128, 128), None, 'none'),
    ((128, 128), 0.5, 'different'),
    ((32, 128), None, 'different'),
    ((128, 32), 0.5, 'none'),
))
def test_sm120_mixed_tail(fmt, k, grans, alpha, c_mode):
    _, _, ar = sm120_dense_quantized(32, k, fmt[0], grans[0], 0)
    _, _, br = sm120_dense_quantized(64, k, fmt[1], grans[1], 2)
    assert (ar @ br.T).abs().max() > 1
    exercise_sm120_dense_fp8_fp4(fmt, 'nt', (32, 64, k), torch.float32,
                                alpha, c_mode, 'float', grans)


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('fmt', ((False, False), (True, True)))
def test_sm120_truepath_control(fmt):
    exercise_sm120_dense_fp8_fp4(fmt, 'nt', (32, 64, 256), torch.float32,
                                None, 'none', 'float', (128, 128))


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    for fmt in ((False, True), (True, False)):
        for k in (128, 256, 384, 512, 640, 768, 896):
            for grans, alpha, c_mode in (((128, 128), None, 'none'), ((128, 128), 0.5, 'different'), ((32, 128), None, 'different'), ((128, 32), 0.5, 'none')):
                test_sm120_mixed_tail(fmt=fmt, k=k, grans=grans, alpha=alpha, c_mode=c_mode)
    for fmt in ((False, False), (True, True)):
        test_sm120_truepath_control(fmt=fmt)
