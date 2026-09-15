import pytest
import torch

from test_fp8_fp4 import exercise_sm120_dense_fp8_fp4, sm120_dense_quantized


pytestmark = pytest.mark.skipif(
    'not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12',
    reason='requires SM120',
)


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


@pytest.mark.parametrize('fmt', ((False, False), (True, True)))
def test_sm120_truepath_control(fmt):
    exercise_sm120_dense_fp8_fp4(fmt, 'nt', (32, 64, 256), torch.float32,
                                None, 'none', 'float', (128, 128))
