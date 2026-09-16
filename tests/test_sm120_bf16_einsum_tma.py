import pytest
import random
import torch

import deep_gemm
from deep_gemm.testing import get_arch_major, test_filter
from test_sm120_bf16_einsum import EXPRESSIONS, exercise_sm120_bf16_einsum


@test_filter(lambda: get_arch_major() == 12)
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


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    for expr in EXPRESSIONS:
        for m in (1024, 769):
            for d_layout in ('canonical', 'aligned'):
                test_native_tma_einsum(expr=expr, m=m, d_layout=d_layout)
