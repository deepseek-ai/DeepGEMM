import argparse
import csv
import time
from pathlib import Path
from typing import Optional

import torch

import deep_gemm
from deep_gemm.testing import calc_diff, get_arch_major
from deep_gemm.utils import per_token_cast_to_fp8, per_token_cast_to_fp4


def load_tensor(path: Optional[str], shape: tuple, seed: int, name: str) -> torch.Tensor:
    if path is None:
        torch.manual_seed(seed)
        return torch.randn(shape, device='cuda', dtype=torch.bfloat16)

    tensor = torch.load(path, map_location='cuda')
    if isinstance(tensor, dict):
        for key in (name, 'tensor', 'data', 'a', 'b', 'c'):
            if key in tensor:
                tensor = tensor[key]
                break
    assert isinstance(tensor, torch.Tensor), f'{path} did not contain a tensor'
    assert tuple(tensor.shape) == shape, f'{path} shape {tuple(tensor.shape)} != expected {shape}'
    return tensor.to(device='cuda', dtype=torch.bfloat16).contiguous()


def read_cases(path: str) -> list:
    cases = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = [field.strip() for field in (reader.fieldnames or [])]
        required = {'testcase_id', 'm', 'k', 'n'}
        missing = required.difference(fieldnames)
        assert not missing, f'CSV missing columns: {sorted(missing)}'
        for row_idx, row in enumerate(reader):
            row = {key.strip(): value.strip() for key, value in row.items() if key is not None}
            cases.append({
                'testcase_id': row['testcase_id'],
                'm': int(row['m']),
                'n': int(row['n']),
                'k': int(row['k']),
                'row_idx': row_idx,
            })
    return cases


def run_case(case: dict, args, out_dtype: torch.dtype) -> dict:
    testcase_id, m, n, k = case['testcase_id'], case['m'], case['n'], case['k']
    result = {
        'testcase_id': testcase_id,
        'm': m,
        'k': k,
        'n': n,
        'passed': False,
        'diff': '',
        'max_abs': '',
        'elapsed_ms': '',
        'error': '',
    }

    try:
        if k % 2 != 0:
            raise ValueError('K must be even because FP4 packs two values per byte')

        case_seed = args.seed + case.get('row_idx', 0) * 1000
        a = load_tensor(args.a, (m, k), case_seed, 'a')
        b = load_tensor(args.b, (n, k), case_seed + 1, 'b')
        c = None
        if args.accumulate or args.c is not None:
            c = load_tensor(args.c, (m, n), case_seed + 2, 'c').to(out_dtype)

        ref_d = (a.float() @ b.float().t() + (c.float() if c is not None else 0)).to(out_dtype)
        d = torch.empty((m, n), device='cuda', dtype=out_dtype)

        # MXFP8/MXFP4: UE8M0 scale with gran_k=32 for both A and B.
        a_mxfp8 = per_token_cast_to_fp8(a, use_ue8m0=True, gran_k=32)
        b_mxfp4 = per_token_cast_to_fp4(b, use_ue8m0=True, gran_k=32)

        torch.cuda.synchronize()
        start = time.perf_counter()
        deep_gemm.fp8_fp4_gemm_nt(
            a_mxfp8,
            b_mxfp4,
            d,
            c=c,
            disable_ue8m0_cast=False,
            recipe_a=(1, 32),
            recipe_b=(1, 32),
        )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000

        diff = float(calc_diff(d, ref_d))
        max_abs = float((d.float() - ref_d.float()).abs().amax().item())
        passed = diff < args.max_diff
        result.update({
            'passed': passed,
            'diff': f'{diff:.8f}',
            'max_abs': f'{max_abs:.8f}',
            'elapsed_ms': f'{elapsed_ms:.3f}',
            'error': '' if passed else f'Diff too large: {diff:.8f} >= {args.max_diff}',
        })

        if args.save is not None and args.csv is None:
            save_path = Path(args.save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'d': d.cpu(), 'ref_d': ref_d.cpu(), 'diff': diff, 'max_abs': max_abs}, save_path)

    except Exception as exc:
        result['error'] = repr(exc)

    return result


def write_results(path: str, results: list) -> None:
    fieldnames = ['testcase_id', 'm', 'k', 'n', 'passed', 'diff', 'max_abs', 'elapsed_ms', 'error']
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run MXFP8 x MXFP4 GEMM cases.')
    parser.add_argument('--m', type=int, default=None)
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--csv', type=str, default=None, help='CSV with testcase_id,m,k,n columns.')
    parser.add_argument('--result-csv', type=str, default='result.csv')
    parser.add_argument('--a', type=str, default=None, help='Optional .pt tensor with shape [M, K].')
    parser.add_argument('--b', type=str, default=None, help='Optional .pt tensor with shape [N, K].')
    parser.add_argument('--c', type=str, default=None, help='Optional .pt tensor with shape [M, N] for accumulation.')
    parser.add_argument('--accumulate', action='store_true', help='Run D = A @ B.T + C.')
    parser.add_argument('--out-dtype', choices=('bf16', 'fp32'), default='bf16')
    parser.add_argument('--max-diff', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save', type=str, default=None, help='Optional output .pt path for D/ref.')
    args = parser.parse_args()

    assert torch.cuda.is_available(), 'CUDA is required'
    assert get_arch_major() == 10, 'MXFP8 x MXFP4 path requires SM100/B200 class GPU'

    out_dtype = torch.bfloat16 if args.out_dtype == 'bf16' else torch.float

    if args.csv is not None:
        assert args.a is None and args.b is None and args.c is None, 'CSV mode generates random A/B/C per shape; do not pass .pt tensors'
        cases = read_cases(args.csv)
        assert cases, f'No cases found in {args.csv}'
    else:
        assert args.m is not None and args.n is not None and args.k is not None, 'Provide --m --n --k, or provide --csv'
        cases = [{'testcase_id': 'single', 'm': args.m, 'n': args.n, 'k': args.k, 'row_idx': 0}]

    print('MXFP8 x MXFP4 custom GEMM:')
    print(' > config: recipe_a=(1, 32), recipe_b=(1, 32), disable_ue8m0_cast=False')
    print(' > A: FP8 e4m3 + UE8M0 scale, gran_k=32')
    print(' > B: FP4 e2m1 + UE8M0 scale, gran_k=32')

    results = []
    for idx, case in enumerate(cases, start=1):
        result = run_case(case, args, out_dtype)
        results.append(result)
        status = 'PASS' if result['passed'] else 'FAIL'
        print(f" > [{idx}/{len(cases)}] {case['testcase_id']}: "
              f"m={case['m']}, n={case['n']}, k={case['k']} -> {status}, "
              f"diff={result['diff']}, max_abs={result['max_abs']}, error={result['error']}")

    write_results(args.result_csv, results)
    num_passed = sum(int(r['passed']) for r in results)
    print(f' > result csv: {args.result_csv}')
    print(f' > summary: {num_passed}/{len(results)} passed')

    if args.csv is None and not results[0]['passed']:
        raise AssertionError(results[0]['error'])


if __name__ == '__main__':
    main()
