import pytest
import torch

import deep_gemm
from test_sm120_bf16_einsum import EXPRESSIONS, exercise_sm120_bf16_einsum


pytestmark = pytest.mark.skipif(
    'not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12',
    reason='requires SM120',
)


@pytest.mark.parametrize('expr', EXPRESSIONS)
@pytest.mark.parametrize('m', (1024, 769))
@pytest.mark.parametrize('d_layout', ('canonical', 'aligned'))
def test_native_tma_einsum(expr, m, d_layout):
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    old_pdl = deep_gemm.get_pdl()
    old_sms = deep_gemm.get_num_sms()
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(128)
        exercise_sm120_bf16_einsum(expr, (m, 2, 512, 128),
                                 ('canonical', 'canonical', d_layout))
        assert deep_gemm.get_num_sms() == old_sms
    finally:
        deep_gemm.set_pdl(old_pdl)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)
