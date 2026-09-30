"""Correctness and API edge cases for gathered SM90 M-grouped BF16 GEMM."""
import argparse

import torch


def check_rejected(fn):
    try:
        fn()
    except RuntimeError:
        return
    raise AssertionError('Expected a host-side argument error')


def check_case(deep_gemm, counts, hidden, features, source_padding=0, output_padding=0, source_offset=0):
    torch.manual_seed(42)
    source_rows = 1031
    x = torch.randn((source_rows, hidden + source_padding), device='cuda', dtype=torch.bfloat16)[:, source_offset:source_offset + hidden]
    weights = torch.randn((len(counts), features, hidden), device='cuda', dtype=torch.bfloat16)
    m = sum(counts)
    row_indices = torch.randint(source_rows, (m,), device='cuda', dtype=torch.int64)
    grouped_layout = torch.repeat_interleave(
        torch.arange(len(counts), device='cuda', dtype=torch.int32),
        torch.tensor(counts, device='cuda', dtype=torch.int64), output_size=m)
    reference = torch.empty((m, features), device='cuda', dtype=torch.bfloat16)
    output = torch.full((m, features + output_padding), float('nan'), device='cuda', dtype=torch.bfloat16)[:, :features]
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(x.index_select(0, row_indices), weights, reference, grouped_layout)
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous_gathered(x, weights, output, grouped_layout, row_indices)
    assert torch.isfinite(output).all()
    assert torch.equal(output, reference), f'Gather changed BF16 output: {counts=}, {hidden=}, {features=}'
    repeated = torch.empty_like(reference)
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous_gathered(x, weights, repeated, grouped_layout, row_indices)
    assert torch.equal(output, repeated)
    start = 0
    for expert, count in enumerate(counts):
        if count:
            dense = (x[row_indices[start:start + count]].float() @ weights[expert].float().T).to(torch.bfloat16)
            assert deep_gemm.testing.calc_diff(output[start:start + count], dense) < 1e-5
        start += count
    check_rejected(lambda: deep_gemm.m_grouped_bf16_gemm_nt_contiguous_gathered(
        x, weights, output, grouped_layout, row_indices.int()))
    check_rejected(lambda: deep_gemm.m_grouped_bf16_gemm_nt_contiguous_gathered(
        x, weights, output, grouped_layout, row_indices[:-1]))
    misaligned = torch.empty((source_rows, hidden + 8), device='cuda', dtype=torch.bfloat16)[:, 1:hidden + 1]
    assert misaligned.stride(0) % 8 == 0 and misaligned.data_ptr() % 16 == 2
    check_rejected(lambda: deep_gemm.m_grouped_bf16_gemm_nt_contiguous_gathered(
        misaligned, weights, output, grouped_layout, row_indices))
    print(f'BITWISE_PASS {counts=} {hidden=} {features=} {source_padding=} {output_padding=} {source_offset=}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    import deep_gemm

    deep_gemm.use_deterministic_algorithms(True)
    deep_gemm.set_mk_alignment_for_contiguous_layout(128)
    for counts in [[1], [17], [128, 0, 256, 128]]:
        for hidden, features in [(64, 64), (256, 144), (512, 256)]:
            check_case(deep_gemm, counts, hidden, features)
    check_case(deep_gemm, [128, 256], 256, 128, source_padding=8, output_padding=8)
    check_case(deep_gemm, [128, 256], 256, 128, source_padding=8, source_offset=8)
    x = torch.empty((0, 64), device='cuda', dtype=torch.bfloat16)
    w = torch.empty((1, 64, 64), device='cuda', dtype=torch.bfloat16)
    d = torch.empty((0, 64), device='cuda', dtype=torch.bfloat16)
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous_gathered(
        x, w, d, torch.empty(0, device='cuda', dtype=torch.int32),
        torch.empty(0, device='cuda', dtype=torch.int64))
    # Exercise the original non-grouped launcher after its internal ABI extension.
    for m, n, k in [(17, 64, 128), (256, 256, 256)]:
        a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
        b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
        output = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
        deep_gemm.bf16_gemm_nt(a, b, output)
        reference = (a.float() @ b.float().T).to(torch.bfloat16)
        assert deep_gemm.testing.calc_diff(output, reference) < 1e-5
    print('ALL_PASS', flush=True)


if __name__ == '__main__':
    main()
