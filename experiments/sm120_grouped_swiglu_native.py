"""Measure single-launch SM120 grouped BF16 SwiGLU against grouped GEMM plus epilogue."""

import argparse
from statistics import median

import deep_gemm
import torch
import triton
import triton.testing
import triton.language as tl


@triton.jit
def _valid_row_swiglu(l1_ptr, output_ptr, counts_ptr,
                      max_m: tl.constexpr, n: tl.constexpr,
                      groups: tl.constexpr, block_m: tl.constexpr, block_n: tl.constexpr):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.program_id(1) * block_n + tl.arange(0, block_n)
    group = rows // max_m
    valid = tl.load(counts_ptr + group, group < groups, other=0)
    mask = (group[:, None] < groups) & ((rows % max_m)[:, None] < valid[:, None]) & (cols[None, :] < n)
    gate = tl.load(l1_ptr + rows[:, None] * (2 * n) + cols[None, :], mask, other=0).to(tl.float32)
    up = tl.load(l1_ptr + rows[:, None] * (2 * n) + n + cols[None, :], mask, other=0).to(tl.float32)
    hidden = (gate * tl.sigmoid(gate) * up).to(tl.bfloat16)
    tl.store(output_ptr + rows[:, None] * n + cols[None, :], hidden, mask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[16, 16, 16, 16])
    parser.add_argument("--max-m", type=int, default=128)
    parser.add_argument("--n", type=int, default=4864)
    parser.add_argument("--k", type=int, default=896)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        parser.error("SM120 GPU required")
    if not args.rows or min(args.rows) < 0 or max(args.rows) == 0 or max(args.rows) > args.max_m:
        parser.error("rows must be between zero and max-m")
    if not 0 < args.max_m or not 0 < args.n or not 0 < args.k or not 0 < args.repeats:
        parser.error("max-m, n, k, and repeats must be positive")

    groups = len(args.rows)
    torch.manual_seed(0)
    a = torch.randn((groups, args.max_m, args.k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((groups, 2 * args.n, args.k), device="cuda", dtype=torch.bfloat16) / args.k**0.5
    counts = torch.tensor(args.rows, device="cuda", dtype=torch.int32)
    l1 = torch.empty((groups, args.max_m, 2 * args.n), device="cuda", dtype=torch.bfloat16)
    baseline = torch.empty((groups, args.max_m, args.n), device="cuda", dtype=torch.bfloat16)
    fused = torch.empty_like(baseline)
    row_mask = torch.arange(args.max_m, device="cuda")[None, :] < counts[:, None]

    def grouped_plus_swiglu():
        deep_gemm.m_grouped_bf16_gemm_nt_masked(a, b, l1, counts, max(args.rows))
        _valid_row_swiglu[(triton.cdiv(groups * args.max_m, 16), triton.cdiv(args.n, 128))](
            l1, baseline, counts, args.max_m, args.n, groups, 16, 128, num_warps=4)

    def grouped_fused():
        deep_gemm.m_grouped_bf16_swiglu_nt_masked(a, b, fused, counts, max(args.rows))

    grouped_plus_swiglu()
    grouped_fused()
    torch.cuda.synchronize()
    delta = (baseline[row_mask].float() - fused[row_mask].float()).abs()
    bad = delta > 0.002 + 0.01 * baseline[row_mask].float().abs()
    print(f"valid_rows={sum(args.rows)}, max_m={args.max_m}, n={args.n}, k={args.k}")
    print(f"correctness: unequal={(delta != 0).sum().item()}, "
          f"outside_tolerance={bad.sum().item()}, max_abs={delta.max().item() if delta.numel() else 0}")
    if bad.any():
        raise AssertionError("Fused output differs from grouped BF16 baseline")

    samples = {"grouped_plus_swiglu": [], "grouped_fused": []}
    for repeat in range(args.repeats):
        order = ("grouped_plus_swiglu", "grouped_fused") if repeat % 2 == 0 else ("grouped_fused", "grouped_plus_swiglu")
        for name in order:
            fn = grouped_plus_swiglu if name == "grouped_plus_swiglu" else grouped_fused
            samples[name].append(triton.testing.do_bench_cudagraph(fn) * 1000)
    for name, values in samples.items():
        print(f"{name}_us: {[round(value, 2) for value in values]}, median={median(values):.2f}")
    print(f"speedup={median(samples['grouped_plus_swiglu']) / median(samples['grouped_fused']):.3f}x")


if __name__ == "__main__":
    main()
