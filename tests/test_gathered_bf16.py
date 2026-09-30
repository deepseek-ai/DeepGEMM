import torch

import deep_gemm


def check_case(counts, hidden, features, source_padding=0, output_padding=0, source_offset=0):
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
    print(f'PASS {counts=} {hidden=} {features=} {source_padding=} {output_padding=} {source_offset=}', flush=True)


def main():
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    deep_gemm.use_deterministic_algorithms(True)
    deep_gemm.set_mk_alignment_for_contiguous_layout(128)
    for counts, hidden, features in [([1], 64, 64), ([17], 256, 144), ([128, 0, 256, 128], 512, 256)]:
        check_case(counts, hidden, features)
    check_case([128, 256], 256, 128, source_padding=8, output_padding=8)
    check_case([128, 256], 256, 128, source_padding=8, source_offset=8)
    x = torch.empty((0, 64), device='cuda', dtype=torch.bfloat16)
    w = torch.empty((1, 64, 64), device='cuda', dtype=torch.bfloat16)
    d = torch.empty((0, 64), device='cuda', dtype=torch.bfloat16)
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous_gathered(
        x, w, d, torch.empty(0, device='cuda', dtype=torch.int32),
        torch.empty(0, device='cuda', dtype=torch.int64))
    print('ALL_PASS', flush=True)


if __name__ == '__main__':
    main()
