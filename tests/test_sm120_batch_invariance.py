"""SM120 batch invariance under `deep_gemm.use_deterministic_algorithms(True)`.

Each test computes the same input rows inside batches of different sizes and requires every row's
result to be bitwise identical to its result in the largest batch. The shapes and recipes follow how
DeepInfer calls these APIs (per-token FP8 activations with packed UE8M0 scales, `gran_k` 32).
"""
import contextlib

import pytest
import torch

import deep_gemm
from deep_gemm.testing import get_arch_major, test_filter
from deep_gemm.utils import align, per_token_cast_to_fp4, per_token_cast_to_fp8

GRAN_K = 32
BATCH_SIZES = (1, 2, 3, 4, 7, 8, 16, 31, 32, 64, 100, 128, 256, 512, 1000, 2048, 4096)


@contextlib.contextmanager
def deterministic(enabled=True):
    old = deep_gemm.get_deterministic_algorithms() if hasattr(deep_gemm, 'get_deterministic_algorithms') else False
    deep_gemm.use_deterministic_algorithms(enabled)
    try:
        yield
    finally:
        deep_gemm.use_deterministic_algorithms(old)


def packed_fp8(x):
    raw, sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=GRAN_K)
    return raw, deep_gemm.transform_sf_into_required_layout(sf, x.shape[0], x.shape[1], (1, GRAN_K))


def float_sf_fp8(x):
    """Per-token FP8 over the last dimension, with power-of-two FP32 scales; DeepGEMM packs them."""
    raw, sf = per_token_cast_to_fp8(x.reshape(-1, x.shape[-1]), use_ue8m0=True, gran_k=GRAN_K)
    return raw.view(x.shape), sf.view(*x.shape[:-1], -1)


def float_sf_fp4(x):
    raw, sf = per_token_cast_to_fp4(x.reshape(-1, x.shape[-1]), use_ue8m0=True, gran_k=GRAN_K)
    return raw.view(*x.shape[:-1], -1), sf.view(*x.shape[:-1], -1)


def mismatching_batch_sizes(compute, max_rows):
    """Returns the batch sizes whose rows differ from the same rows computed in a `max_rows` batch."""
    reference = compute(max_rows)
    return [m for m in BATCH_SIZES if m <= max_rows and not torch.equal(compute(m), reference[:m])]


DENSE_SHAPES = ((512, 7168), (1536, 4096), (4096, 1024), (7168, 2048), (64, 4096))


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('n,k', DENSE_SHAPES)
@pytest.mark.parametrize('accumulate', (False, True))
def test_sm120_fp8_gemm_batch_invariant(n, k, accumulate):
    torch.manual_seed(n + k)
    max_rows = BATCH_SIZES[-1]
    x = torch.randn(max_rows, k, device='cuda', dtype=torch.bfloat16)
    acc = torch.randn(max_rows, n, device='cuda', dtype=torch.bfloat16)
    weight = packed_fp8(torch.randn(n, k, device='cuda', dtype=torch.bfloat16))

    def compute(m):
        a = packed_fp8(x[:m])
        d = acc[:m].clone() if accumulate else torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
        deep_gemm.fp8_fp4_gemm_nt(a, weight, d, d if accumulate else None, recipe=(1, 1, GRAN_K))
        return d

    with deterministic():
        assert mismatching_batch_sizes(compute, max_rows) == []


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('n,k', DENSE_SHAPES)
def test_sm120_bf16_gemm_batch_invariant(n, k):
    torch.manual_seed(n * 3 + k)
    max_rows = BATCH_SIZES[-1]
    x = torch.randn(max_rows, k, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)

    def compute(m):
        d = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
        deep_gemm.bf16_gemm_nt(x[:m], weight, d, None)
        return d

    with deterministic():
        assert mismatching_batch_sizes(compute, max_rows) == []


def grouped_rows(counts, alignment):
    """Row offsets of each group in a psum-layout buffer, with every group start aligned."""
    starts, cursor = [], 0
    for count in counts:
        starts.append(cursor)
        cursor = align(cursor + count, alignment)
    return starts, max(cursor, alignment)


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('dtype', ('fp8_fp4', 'bf16'))
@pytest.mark.parametrize('n,k', ((4096, 2048), (2048, 2048)))
def test_sm120_m_grouped_contiguous_batch_invariant(dtype, n, k):
    # Every expert computes the same leading rows whatever the other experts receive
    torch.manual_seed(n + k + len(dtype))
    num_groups, max_per_group = 8, 512
    alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    tokens = torch.randn(num_groups, max_per_group, k, device='cuda', dtype=torch.bfloat16)
    weights = torch.randn(num_groups, n, k, device='cuda', dtype=torch.bfloat16)
    if dtype == 'fp8_fp4':
        b = float_sf_fp4(weights)
    else:
        b = weights

    def compute(counts):
        starts, total = grouped_rows(counts, alignment)
        a = torch.zeros(total, k, device='cuda', dtype=torch.bfloat16)
        for g, (start, count) in enumerate(zip(starts, counts)):
            a[start:start + count] = tokens[g, :count]
        psum = torch.tensor([start + count for start, count in zip(starts, counts)], device='cuda', dtype=torch.int32)
        d = torch.empty(total, n, device='cuda', dtype=torch.bfloat16)
        expected_m = max(1, sum(counts) // num_groups)
        if dtype == 'fp8_fp4':
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
                float_sf_fp8(a), b, d, psum, recipe_a=(1, GRAN_K), recipe_b=(1, GRAN_K),
                use_psum_layout=True, expected_m_for_psum_layout=expected_m, ensure_zero_padding=False)
        else:
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
                a, b, d, psum, use_psum_layout=True, expected_m_for_psum_layout=expected_m, ensure_zero_padding=False)
        return [d[start:start + count] for start, count in zip(starts, counts)]

    with deterministic():
        reference = compute([max_per_group] * num_groups)
        mismatches = []
        for scale in (1, 2, 5, 17, 64, 200, 511):
            counts = [(scale * (g + 1) * 7) % max_per_group + (g % 2) for g in range(num_groups)]
            for g, rows in enumerate(compute(counts)):
                if not torch.equal(rows, reference[g][:rows.shape[0]]):
                    mismatches.append((counts, g))
        assert mismatches == []


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('h,d,r', ((16, 512, 1024), (8, 1024, 512)))
def test_sm120_fp8_einsum_batch_invariant(h, d, r):
    # DeepInfer's attention output projection: 'bhr,hdr->bhd', batch over tokens
    torch.manual_seed(h + d + r)
    max_rows = BATCH_SIZES[-1]
    context = torch.randn(max_rows, h, r, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(h, d, r, device='cuda', dtype=torch.bfloat16)
    w = float_sf_fp8(weight)

    def compute(m):
        out = torch.empty(m, h, d, device='cuda', dtype=torch.bfloat16)
        deep_gemm.fp8_einsum('bhr,hdr->bhd', float_sf_fp8(context[:m]), w, out, recipe=(1, 1, GRAN_K))
        return out

    with deterministic():
        assert mismatching_batch_sizes(compute, max_rows) == []


@test_filter(lambda: get_arch_major() == 12)
@pytest.mark.parametrize('num_splits', (1, 16))
def test_sm120_hc_prenorm_batch_invariant(num_splits):
    torch.manual_seed(num_splits)
    max_rows, n, k = BATCH_SIZES[-1], 24, 4 * 4096
    x = torch.randn(max_rows, k, device='cuda', dtype=torch.bfloat16)
    fn = torch.randn(n, k, device='cuda', dtype=torch.float32)

    def compute(m):
        out = torch.empty(num_splits, m, n, device='cuda', dtype=torch.float32)
        sqr_sum = torch.empty(num_splits, m, device='cuda', dtype=torch.float32)
        if num_splits == 1:
            deep_gemm.tf32_hc_prenorm_gemm(x[:m], fn, out[0], sqr_sum[0])
        else:
            deep_gemm.tf32_hc_prenorm_gemm(x[:m], fn, out, sqr_sum, num_splits)
        return torch.cat([out.transpose(0, 1).reshape(m, -1), sqr_sum.transpose(0, 1)], dim=1)

    with deterministic():
        assert mismatching_batch_sizes(compute, max_rows) == []
