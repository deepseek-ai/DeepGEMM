"""Per-call BM selection must not mutate process-wide alignment or Graph state."""

import pytest
import torch

import deep_gemm
from tests.test_sm120_grouped_swiglu import compare_valid, sm120_device


def inputs(n=65):
    e, m, k = 4, 65, 136
    # Dyadic products and sums are exact in FP32, including partial M/N/K tiles.
    a = (((torch.arange(e*m*k, device="cuda").reshape(e, m, k) % 17)-8).float()/16).bfloat16()
    b = (((torch.arange(e*2*n*k, device="cuda").reshape(e, 2*n, k) % 13)-6).float()/64).bfloat16()
    counts = torch.tensor([0, 1, 31, 65], device="cuda", dtype=torch.int32)
    return a, b, counts


def oracle(a, b):
    projections = (a.double() @ b.double().transpose(-1, -2)).bfloat16().float()
    gate, up = projections.chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate) * up).bfloat16()


@pytest.mark.parametrize("alignment", [32, 128])
@pytest.mark.parametrize("tile", [0, 32, 64, 128])
def test_per_call_tile_oracle_and_sentinels(tile, alignment):
    a, b, counts = inputs()
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    # Exercise an unaligned output base as well as odd row strides.
    storage = torch.full((17+4*65*65+16,), float("nan"), device="cuda", dtype=a.dtype)
    out = storage[17:-16].view(4, 65, 65)
    deep_gemm.m_grouped_bf16_swiglu_nt_masked(a, b, out, counts, 65, sm120_m_tile=tile)
    compare_valid(out, oracle(a, b), (0, 1, 31, 65))
    assert torch.isnan(storage[:17]).all() and torch.isnan(storage[-16:]).all()
    assert deep_gemm.get_mk_alignment_for_contiguous_layout() == alignment


@pytest.mark.parametrize("alignment", [32, 64, 128])
def test_omitted_tile_preserves_default(alignment):
    a, b, counts = inputs()
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    outputs = [torch.full((4, 65, 65), float("nan"), device="cuda", dtype=a.dtype)
               for _ in range(2)]
    deep_gemm.m_grouped_bf16_swiglu_nt_masked(a, b, outputs[0], counts, 65)
    deep_gemm.m_grouped_bf16_swiglu_nt_masked(a, b, outputs[1], counts, 65, sm120_m_tile=0)
    torch.testing.assert_close(*outputs, rtol=0, atol=0, equal_nan=True)
    assert deep_gemm.get_mk_alignment_for_contiguous_layout() == alignment


@pytest.mark.parametrize("tile", [-1, 1, 16, 48, 256])
def test_invalid_tile_and_positional_argument(tile):
    a, b, counts = inputs()
    out = torch.full((4, 65, 65), 7, device="cuda", dtype=a.dtype)
    with pytest.raises(RuntimeError):
        deep_gemm.m_grouped_bf16_swiglu_nt_masked(a, b, out, counts, 65, sm120_m_tile=tile)
    with pytest.raises(TypeError):
        deep_gemm.m_grouped_bf16_swiglu_nt_masked(a, b, out, counts, 65, 32)
    assert (out == 7).all()
    assert deep_gemm.get_mk_alignment_for_contiguous_layout() == 128


def test_small_n_explicit_tile_overrides_global_alignment():
    a, b, counts = inputs(n=17)
    out = torch.full((4, 65, 17), 7, device="cuda", dtype=a.dtype)
    with pytest.raises(RuntimeError):
        deep_gemm.m_grouped_bf16_swiglu_nt_masked(a, b, out, counts, 65, sm120_m_tile=32)
    assert (out == 7).all()
    deep_gemm.set_mk_alignment_for_contiguous_layout(32)
    for tile in (64, 128):
        out.fill_(float("nan"))
        deep_gemm.m_grouped_bf16_swiglu_nt_masked(a, b, out, counts, 65, sm120_m_tile=tile)
        compare_valid(out, oracle(a, b), (0, 1, 31, 65))
        assert deep_gemm.get_mk_alignment_for_contiguous_layout() == 32


def test_independent_stream_graphs_with_per_call_tiles():
    states = []
    for tile in (32, 64, 128):
        a, b, counts = inputs()
        out = torch.empty((4, 65, 65), device="cuda", dtype=a.dtype)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                deep_gemm.m_grouped_bf16_swiglu_nt_masked(
                    a, b, out, counts, 65, sm120_m_tile=tile)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                deep_gemm.m_grouped_bf16_swiglu_nt_masked(
                    a, b, out, counts, 65, sm120_m_tile=tile)
        torch.cuda.current_stream().wait_stream(stream)
        states.append((a, b, counts, out, stream, graph))

    # Captured tiles remain fixed when another caller changes the default.
    for alignment, rows in ((32, (65, 0, 7, 1)), (128, (0, 0, 0, 0)), (64, (0, 1, 31, 65))):
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        references = []
        for a, b, counts, out, stream, graph in states:
            a.neg_()
            b.mul_(0.5)
            counts.copy_(torch.tensor(rows, device="cuda", dtype=torch.int32))
            out.fill_(float("nan"))
            references.append(oracle(a, b))
        for _, _, _, _, stream, graph in states:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph.replay()
        for state in states:
            torch.cuda.current_stream().wait_stream(state[4])
        for state, reference in zip(states, references):
            compare_valid(state[3], reference, rows)
        assert deep_gemm.get_mk_alignment_for_contiguous_layout() == alignment
