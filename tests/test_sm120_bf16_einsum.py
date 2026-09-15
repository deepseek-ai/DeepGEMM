import pytest
import torch

import deep_gemm
from deep_gemm.testing import get_arch_major


pytestmark = pytest.mark.skipif(
    'not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12',
    reason='requires SM120',
)


EXPRESSIONS = ('bhr,hdr->bhd', 'bhd,hdr->bhr')


def einsum_storage(values, layout):
    x, y, z = values.shape
    if layout == 'canonical':
        row, batch, offset = z, y * z, 8
    elif layout == 'aligned':
        row = (z + 15) // 8 * 8
        batch, offset = y * row + 8, 8
    else:
        row, batch, offset = z + 3, y * (z + 3) + 5, 1
    size = offset + max(x, 1) * max(batch, 1) + max(row, 1) + 16
    storage = torch.full((size,), 19, dtype=torch.bfloat16, device=values.device)
    tensor = storage.as_strided(values.shape, (batch, row, 1), offset)
    tensor.copy_(values)
    guard = torch.zeros_like(storage, dtype=torch.bool)
    guard.as_strided(tensor.shape, tensor.stride(), tensor.storage_offset()).fill_(True)
    return tensor, storage, guard


def exercise_sm120_bf16_einsum(expr, shape, layouts, graph=False, pdl=False, deterministic=True):
    assert get_arch_major() == 12 and expr in EXPRESSIONS
    m, h, n, k = shape
    a_shape = (m, h, k)
    b_shape = (h, n, k) if expr == EXPRESSIONS[0] else (h, k, n)
    generator = torch.Generator().manual_seed(8191 + m + h + n + k)
    av = (torch.randn(a_shape, generator=generator) / 4).to(torch.bfloat16)
    bv = (torch.randn(b_shape, generator=generator) / 4).to(torch.bfloat16)
    a, astorage, aguard = einsum_storage(av.cuda(), layouts[0])
    b, bstorage, bguard = einsum_storage(bv.cuda(), layouts[1])
    d, dstorage, dguard = einsum_storage(torch.zeros((m, h, n), dtype=torch.bfloat16, device='cuda'), layouts[2])

    def run():
        deep_gemm.einsum(expr, a, b, d)

    def check():
        expected = torch.einsum(expr, av.float(), bv.float())
        actual = d.cpu().float()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.015)
        if k == 0:
            assert (actual == 0).all()
        for storage, guard in ((astorage, aguard), (bstorage, bguard), (dstorage, dguard)):
            assert (storage[~guard] == 19).all()
        torch.testing.assert_close(a.cpu(), av, rtol=0, atol=0)
        torch.testing.assert_close(b.cpu(), bv, rtol=0, atol=0)

    old_pdl = deep_gemm.get_pdl()
    try:
        deep_gemm.use_deterministic_algorithms(deterministic)
        deep_gemm.set_pdl(pdl)
        for _ in range(3):
            d.fill_(float('nan'))
            run()
            check()
        if graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=stream):
                run()
            for phase in range(3):
                if phase:
                    av = (-av.float() / 2).to(torch.bfloat16)
                    bv = (bv.float() + 0.125).to(torch.bfloat16)
                    a.copy_(av)
                    b.copy_(bv)
                d.fill_(float('nan'))
                captured.replay()
                torch.cuda.synchronize()
                check()
    finally:
        deep_gemm.set_pdl(old_pdl)
        deep_gemm.use_deterministic_algorithms(False)
    print(f' > Native SM120 BF16 einsum fixture: {expr=}, {shape=}, {layouts=}, {graph=}, {pdl=}, {deterministic=}')


@pytest.mark.parametrize('expr', EXPRESSIONS)
def test_sm120_bf16_einsum_native_layouts(expr):
    cases = ((1, 2, 1, 13), (2, 3, 7, 67), (65, 2, 17, 129),
             (129, 3, 33, 65), (16, 2, 8, 128), (64, 3, 32, 256))
    for shape in cases:
        for layouts in (('canonical',) * 3, ('aligned',) * 3, ('unaligned',) * 3,
                        ('aligned', 'unaligned', 'canonical'), ('unaligned', 'aligned', 'unaligned')):
            exercise_sm120_bf16_einsum(expr, shape, layouts)


@pytest.mark.parametrize('expr', EXPRESSIONS)
def test_sm120_bf16_einsum_wide_controls(expr):
    for shape in ((128, 2, 128, 128), (257, 4, 256, 256)):
        for d_layout in ('canonical', 'aligned'):
            exercise_sm120_bf16_einsum(expr, shape, ('canonical', 'canonical', d_layout))


@pytest.mark.parametrize('expr', EXPRESSIONS)
def test_sm120_bf16_einsum_graph(expr):
    for i, n in enumerate((1, 7, 17, 33, 8, 32)):
        for pdl in (False, True):
            exercise_sm120_bf16_einsum(expr, (65, 2, n, 67 if i % 2 else 128),
                                     ('aligned', 'unaligned', 'unaligned') if i % 2 else ('canonical',) * 3,
                                     graph=True, pdl=pdl)


@pytest.mark.parametrize('expr', EXPRESSIONS)
def test_sm120_bf16_einsum_empty(expr):
    for shape in ((0, 2, 7, 13), (2, 0, 7, 13), (2, 3, 0, 13), (2, 3, 7, 0)):
        for graph in (False, True):
            exercise_sm120_bf16_einsum(expr, shape, ('unaligned',) * 3, graph=graph)


@pytest.mark.parametrize('expr', EXPRESSIONS)
def test_sm120_bf16_einsum_c_rejection(expr):
    assert get_arch_major() == 12
    a = torch.ones((2, 2, 128), dtype=torch.bfloat16, device='cuda')
    b = torch.ones((2, 8, 128) if expr == EXPRESSIONS[0] else (2, 128, 8), dtype=torch.bfloat16, device='cuda')
    d = torch.full((2, 2, 8), 19, dtype=torch.bfloat16, device='cuda')
    for same in (False, True):
        c = d if same else torch.zeros_like(d)
        with pytest.raises(RuntimeError, match='not c.has_value'):
            deep_gemm.einsum(expr, a, b, d, c=c)
        assert (d == 19).all()


@pytest.mark.parametrize('expr', EXPRESSIONS)
def test_sm120_bf16_einsum_default_policy(expr):
    exercise_sm120_bf16_einsum(expr, (16, 2, 32, 128), ('canonical',) * 3, deterministic=False)
