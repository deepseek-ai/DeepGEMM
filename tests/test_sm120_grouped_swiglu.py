"""Contract tests for the SM120 grouped BF16 SwiGLU entry point."""

import pytest
import torch

import deep_gemm


@pytest.fixture(autouse=True)
def sm120_device():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 GPU required")
    alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(128)
    try:
        yield
    finally:
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)


def materialized(a, b, counts, n, hint):
    groups, m, k = a.shape
    # The ordinary API's physical N must be a multiple of 8. The fused output
    # may have an odd logical width; pad only the reference's unused channels.
    physical_n = ((2 * n + 7) // 8) * 8
    weights = b
    if physical_n != 2 * n:
        weights = torch.zeros((groups, physical_n, k), dtype=b.dtype, device=b.device)
        weights[:, :2 * n].copy_(b)
    l1 = torch.empty((groups, m, physical_n), dtype=a.dtype, device=a.device)
    deep_gemm.m_grouped_bf16_gemm_nt_masked(a, weights, l1, counts, hint)
    return (torch.nn.functional.silu(l1[..., :n].float()) *
            l1[..., n:2 * n].float()).bfloat16()


def compare_valid(actual, reference, rows):
    for group, count in enumerate(rows):
        if count:
            assert torch.isfinite(actual[group, :count]).all()
            torch.testing.assert_close(actual[group, :count], reference[group, :count],
                                       rtol=1e-2, atol=2e-3)
        assert torch.isnan(actual[group, count:]).all()


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("rows,m,k,n", [
    ((16,) * 4, 128, 896, 4864),
    ((128,) * 4, 128, 896, 4864),
    ((16,) * 48, 128, 256, 128),
    ((0, 1, 31, 65), 128, 256, 65),
    ((0, 7, 129, 255), 256, 832, 128),
    ((1025,) * 4, 1152, 256, 128),
    ((0,) * 4, 128, 256, 128),
    ((65, 1, 0, 31), 65, 136, 17),
    ((32,) * 8, 128, 256, 32),
    ((16,) * 32, 128, 256, 128),
], ids=["e4_m16", "e4_m128", "e48_m16", "mixed_odd_n", "mixed_k_tail",
        "m1025", "all_empty", "capacity_tail", "e8_n32", "e32_n128"])
def test_materialized(rows, m, k, n, seed):
    torch.manual_seed(seed)
    groups = len(rows)
    a = torch.randn((groups, m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((groups, 2 * n, k), device="cuda", dtype=torch.bfloat16) / k**0.5
    counts = torch.tensor(rows, device="cuda", dtype=torch.int32)
    saved = (a.clone(), b.clone(), counts.clone())
    storage = torch.full((16 + groups * m * n + 16,), float("nan"),
                         device="cuda", dtype=torch.bfloat16)
    actual = storage[16:-16].view(groups, m, n)
    hint = max(1, max(rows))
    reference = materialized(a, b, counts, n, hint)
    deep_gemm.m_grouped_bf16_swiglu_nt_masked(a, b, actual, counts, hint)
    compare_valid(actual, reference, rows)
    assert torch.isnan(storage[:16]).all() and torch.isnan(storage[-16:]).all()
    for original, old in zip((a, b, counts), saved):
        torch.testing.assert_close(original, old, rtol=0, atol=0)


@pytest.mark.parametrize("n", [65, 128])
@pytest.mark.parametrize("offset", [16, 17])
def test_output_pair_alignment_and_tail(n, offset):
    # A contiguous BF16 output may start at a 2-byte offset. With odd N,
    # successive rows also alternate between 2-byte and 4-byte alignment.
    torch.manual_seed(73)
    groups, m, k = 4, 128, 256
    rows = (0, 1, 31, 65)
    a = torch.randn((groups, m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((groups, 2 * n, k), device="cuda", dtype=torch.bfloat16) / k**0.5
    counts = torch.tensor(rows, device="cuda", dtype=torch.int32)
    storage = torch.full((offset + groups * m * n + 16,), float("nan"),
                         device="cuda", dtype=torch.bfloat16)
    actual = storage[offset:-16].view(groups, m, n)
    reference = materialized(a, b, counts, n, 65)
    deep_gemm.m_grouped_bf16_swiglu_nt_masked(a, b, actual, counts, 65)
    compare_valid(actual, reference, rows)
    assert torch.isnan(storage[:offset]).all() and torch.isnan(storage[-16:]).all()


def test_exact_dot_product_oracle():
    groups, m, k, n = 4, 128, 256, 65
    # Bounded dyadic products/sums are exactly representable in FP32. This
    # independently checks projection rounding, gate/up order, and activation.
    a = (((torch.arange(groups * m * k, device="cuda").reshape(groups, m, k) % 17)
          - 8).float() / 16).bfloat16()
    b = (((torch.arange(groups * 2 * n * k, device="cuda").reshape(groups, 2 * n, k) % 13)
          - 6).float() / 64).bfloat16()
    rows = (0, 1, 31, 128)
    counts = torch.tensor(rows, device="cuda", dtype=torch.int32)
    actual = torch.full((groups, m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    projections = (a.double() @ b.double().transpose(-1, -2)).bfloat16().float()
    reference = (torch.nn.functional.silu(projections[..., :n]) * projections[..., n:]).bfloat16()
    deep_gemm.m_grouped_bf16_swiglu_nt_masked(a, b, actual, counts, 128)
    compare_valid(actual, reference, rows)


def test_graph_replay_updates_inputs_weights_and_counts():
    groups, m, k, n = 4, 128, 256, 128
    torch.manual_seed(42)
    a = torch.randn((groups, m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((groups, 2 * n, k), device="cuda", dtype=torch.bfloat16) / k**0.5
    counts = torch.full((groups,), 16, device="cuda", dtype=torch.int32)
    actual = torch.full((groups, m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    invoke = lambda: deep_gemm.m_grouped_bf16_swiglu_nt_masked(a, b, actual, counts, 128)
    # Compile before capture. Synchronize stream ownership before side-stream warmup.
    invoke()
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            invoke()
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    for rows in ((0, 1, 31, 128), (0, 0, 0, 0), (128, 16, 7, 0)):
        counts.copy_(torch.tensor(rows, device="cuda", dtype=torch.int32))
        a.mul_(0.5)
        b.neg_()
        actual.fill_(float("nan"))
        reference = materialized(a, b, counts, n, 128)
        graph.replay()
        compare_valid(actual, reference, rows)


@pytest.mark.parametrize("alignment", [32, 64, 128])
def test_runtime_alignment(alignment):
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    torch.manual_seed(37)
    rows = (0, 1, 31, 65)
    a = torch.randn((4, 128, 256), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((4, 256, 256), device="cuda", dtype=torch.bfloat16) / 16
    counts = torch.tensor(rows, device="cuda", dtype=torch.int32)
    actual = torch.full((4, 128, 128), float("nan"), device="cuda", dtype=torch.bfloat16)
    reference = materialized(a, b, counts, 128, 65)
    deep_gemm.m_grouped_bf16_swiglu_nt_masked(a, b, actual, counts, 65)
    compare_valid(actual, reference, rows)


def test_small_n_rejects_alignment32():
    deep_gemm.set_mk_alignment_for_contiguous_layout(32)
    a = torch.zeros((4, 128, 256), device="cuda", dtype=torch.bfloat16)
    b = torch.zeros((4, 34, 256), device="cuda", dtype=torch.bfloat16)
    actual = torch.full((4, 128, 17), 7, device="cuda", dtype=torch.bfloat16)
    counts = torch.tensor([0, 1, 31, 65], device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="requires alignment >= 64"):
        deep_gemm.m_grouped_bf16_swiglu_nt_masked(a, b, actual, counts, 65)
    assert (actual == 7).all()


@pytest.mark.parametrize("seed", [0, 1])
def test_random_selected_experts_fp64_oracle(seed):
    # Keep the full 48-expert launch. Independently evaluate three fixed experts;
    # do not substitute the ordinary GEMM result for the mathematical reference.
    torch.manual_seed(seed)
    groups, m, k, n = 48, 128, 896, 4864
    a = torch.randn((groups, m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((groups, 2 * n, k), device="cuda", dtype=torch.bfloat16) / k**0.5
    counts = torch.full((groups,), m, device="cuda", dtype=torch.int32)
    actual = torch.empty((groups, m, n), device="cuda", dtype=torch.bfloat16)
    reference = materialized(a, b, counts, n, m)
    deep_gemm.m_grouped_bf16_swiglu_nt_masked(a, b, actual, counts, m)
    failures = []
    for group in (0, 24, 47):
        projections = (a[group].double() @ b[group].double().T).bfloat16().float()
        oracle = (torch.nn.functional.silu(projections[:, :n]) * projections[:, n:]).bfloat16()
        for label, value in (("materialized", reference[group]), ("fused", actual[group])):
            try:
                torch.testing.assert_close(value, oracle, rtol=1e-2, atol=2e-3)
            except AssertionError as error:
                failures.append(f"seed={seed}, expert={group}, path={label}\n{error}")
    assert not failures, "\n\n".join(failures)


@pytest.mark.parametrize("invalid", [
    "weights", "counts_rank", "counts_dtype", "output", "hint",
    "hint_large", "a_dtype", "a_offset", "b_offset", "noncontiguous",
])
def test_reject_invalid_contract(invalid):
    a = torch.empty((2, 128, 256), device="cuda", dtype=torch.bfloat16)
    b = torch.empty((2, 256, 256), device="cuda", dtype=torch.bfloat16)
    d = torch.full((2, 128, 128), 7, device="cuda", dtype=torch.bfloat16)
    counts = torch.tensor([0, 7], device="cuda", dtype=torch.int32)
    hint = 7
    if invalid == "weights":
        b = b[:, :255].contiguous()
    elif invalid == "counts_rank":
        counts = counts.view(1, 2)
    elif invalid == "counts_dtype":
        counts = counts.long()
    elif invalid == "output":
        d = d.float()
    elif invalid == "hint":
        hint = 0
    elif invalid == "hint_large":
        hint = 129
    elif invalid == "a_dtype":
        a = a.float()
    elif invalid in ("a_offset", "b_offset"):
        tensor = a if invalid == "a_offset" else b
        misaligned = torch.empty(tensor.numel() + 1, device="cuda", dtype=tensor.dtype)[1:].view_as(tensor)
        if invalid == "a_offset":
            a = misaligned
        else:
            b = misaligned
    elif invalid == "noncontiguous":
        a = torch.empty((2, 256, 128), device="cuda", dtype=a.dtype).transpose(-1, -2)
    with pytest.raises(RuntimeError):
        deep_gemm.m_grouped_bf16_swiglu_nt_masked(a, b, d, counts, hint)
    assert (d == 7).all()
