"""SM120 masked BF16 projection + SwiGLU; compare with the original GEMM."""
import pytest
import torch
import torch.nn.functional as F

import deep_gemm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="requires SM120")


def make_case(m=129, i=136, d=136, expected_m=33):
    torch.manual_seed(123)
    e = 3
    a = torch.randn(e, m, d, device='cuda', dtype=torch.bfloat16) / 8
    b = torch.randn(e, 2 * i, d, device='cuda', dtype=torch.bfloat16) / 8
    # Reuse the official SM100 BF16 layout converter; Wdown is unchanged.
    b8, _ = deep_gemm.transform_weights_for_mega_moe(
        b, torch.empty(e, d, i, device='cuda', dtype=torch.bfloat16))
    h = torch.full((e, m, i), 19, device='cuda', dtype=torch.bfloat16)
    counts = torch.tensor([0, 1, m], device='cuda', dtype=torch.int32)
    return a, b, b8, h, counts, expected_m


def check_case(case):
    a, b, b8, h, counts, expected = case
    p = torch.empty((*a.shape[:2], b.shape[1]), device='cuda', dtype=torch.bfloat16)
    deep_gemm.m_grouped_bf16_gemm_nt_masked(a, b, p, counts, expected)
    g, u = p.chunk(2, dim=-1)
    ref = (F.silu(g.float()) * u.float()).to(torch.bfloat16)
    deep_gemm.m_grouped_bf16_gemm_swiglu_nt_masked(a, b8, h, counts, expected)
    valid = torch.arange(a.shape[1], device='cuda')[None, :] < counts[:, None]
    torch.testing.assert_close(h[valid], ref[valid], rtol=0.008, atol=2e-5)
    assert bool((h[~valid] == 19).all()), 'padding was modified'
    # Verify that interleaving alone retains original GEMM arithmetic.
    q = torch.empty_like(p)
    deep_gemm.m_grouped_bf16_gemm_nt_masked(a, b8, q, counts, expected)
    q = q.unflatten(-1, (-1, 2, 8))
    restored = torch.cat((q[..., 0, :].flatten(-2), q[..., 1, :].flatten(-2)), -1)
    torch.testing.assert_close(restored[valid], p[valid], rtol=0, atol=0)
    # Independent FP32 reference with the original BF16 projection boundary.
    for e, n in enumerate(counts.tolist()):
        if n:
            proj = (a[e, :n].float() @ b[e].float().T).to(torch.bfloat16)
            pg, pu = proj.chunk(2, -1)
            oracle = (F.silu(pg.float()) * pu.float()).to(torch.bfloat16)
            torch.testing.assert_close(h[e, :n], oracle, rtol=0.02, atol=0.002)
    torch.cuda.synchronize()


@pytest.mark.parametrize('m,i,d,expected', (
    (1, 32, 64, 1), (65, 136, 136, 1), (129, 136, 136, 33),
    (193, 256, 256, 65), (256, 768, 2048, 128),
    (257, 1536, 4096, 256), (128, 1408, 2048, 2048)))
def test_projection_and_tails(m, i, d, expected):
    check_case(make_case(m, i, d, expected))


@pytest.mark.parametrize('pdl', (False, True))
def test_graph_changed_counts_and_inputs(pdl):
    old = deep_gemm.get_pdl()
    try:
        deep_gemm.set_pdl(pdl)
        case = make_case()
        check_case(case)
        a, b, b8, h, counts, expected = case
        def run():
            deep_gemm.m_grouped_bf16_gemm_swiglu_nt_masked(a, b8, h, counts, expected)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
        for values in ([0, 0, 0], [129, 0, 7], [1, 129, 0]):
            counts.copy_(torch.tensor(values, device='cuda', dtype=torch.int32))
            a.neg_()
            h.fill_(19)
            graph.replay()
            saved = h.clone()
            check_case(case)
            torch.testing.assert_close(h, saved, rtol=0, atol=0)
    finally:
        deep_gemm.set_pdl(old)


def test_bf16_projection_rounding_boundary():
    a, b, b8, h, counts, expected = make_case(m=1, i=32, d=64, expected_m=1)
    # Exact products, non-BF16 sums: activation must consume rounded projections.
    a.zero_(); b.zero_()
    a[..., :2] = 1
    b[:, :32, 0] = 1; b[:, :32, 1] = 1 / 256
    b[:, 32:, 0] = 2; b[:, 32:, 1] = 1 / 128
    b8, _ = deep_gemm.transform_weights_for_mega_moe(b, torch.empty(3, 64, 32, device='cuda', dtype=torch.bfloat16))
    check_case((a, b, b8, h, counts, expected))
    unrounded = F.silu(torch.tensor(1 + 1 / 256, device='cuda')) * (2 + 1 / 128)
    rounded = F.silu(torch.tensor(1., device='cuda')) * 2
    assert unrounded.to(torch.bfloat16) != rounded.to(torch.bfloat16)
    torch.testing.assert_close(h[1:, 0], rounded.to(torch.bfloat16).expand(2, 32), rtol=0, atol=0)


@pytest.mark.parametrize('kind', ('dtype', 'cpu', 'shape', 'strided', 'alias', 'partial_alias'))
def test_reject_invalid_inputs(kind):
    a, _, b, h, counts, expected = make_case(m=128, i=128, d=128)
    if kind == 'dtype':
        h = h.float()
    elif kind == 'cpu':
        counts = counts.cpu()
    elif kind == 'shape':
        h = torch.empty(3, 128, 120, device='cuda', dtype=torch.bfloat16)
    elif kind == 'strided':
        a = a.transpose(1, 2)
    elif kind == 'alias':
        h = a
    elif kind == 'partial_alias':
        storage = torch.empty(a.numel() + 8, device='cuda', dtype=torch.bfloat16)
        a = storage[:a.numel()].view_as(a)
        h = storage[8:].view_as(h)
    with pytest.raises(RuntimeError):
        deep_gemm.m_grouped_bf16_gemm_swiglu_nt_masked(a, b, h, counts, expected)


@pytest.mark.parametrize('alignment', (32, 64, 128))
def test_projection_warp_geometry(alignment):
    old = deep_gemm.get_mk_alignment_for_contiguous_layout()
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        check_case(make_case(m=129, i=136, d=136, expected_m=33))
    finally:
        deep_gemm.set_mk_alignment_for_contiguous_layout(old)
