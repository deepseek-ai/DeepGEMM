import argparse
import csv
from pathlib import Path

import torch

import deep_gemm
from deep_gemm.testing import bench_kineto, calc_diff, count_bytes, get_arch_major
from generators import (
    KernelType, MajorTypeAB, QuantConfig,
    generate_normal, get_ue8m0_usage,
)


def read_cases(path: str) -> list:
    cases = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = [field.strip() for field in (reader.fieldnames or [])]
        required = {'testcase_id', 'm', 'k', 'n'}
        missing = required.difference(fieldnames)
        assert not missing, f'CSV missing columns: {sorted(missing)}'
        for row in reader:
            row = {key.strip(): value.strip() for key, value in row.items() if key is not None}
            cases.append({
                'testcase_id': row['testcase_id'],
                'm': int(row['m']),
                'k': int(row['k']),
                'n': int(row['n']),
            })
    return cases


def run_case(case: dict, out_dtype: torch.dtype, max_diff: float, accumulate: bool) -> dict:
    testcase_id, m, k, n = case['testcase_id'], case['m'], case['k'], case['n']
    result = {
        'testcase_id': testcase_id,
        'm': m,
        'k': k,
        'n': n,
        'passed': False,
        'diff': '',
        'max_abs': '',
        'elapsed_us': '',
        'tflops': '',
        'gbps': '',
        'error': '',
    }

    try:
        if k % 128 != 0:
            raise ValueError('K must be a multiple of 128 for K-major FP4 TMA unpacked shared memory')

        kernel_type = KernelType.Kernel1D1D
        quant_config = QuantConfig((32, 32, False, True))
        major_a = MajorTypeAB.KMajor
        major_b = MajorTypeAB.KMajor
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0
        recipe, recipe_a, recipe_b = quant_config.get_recipes(is_wgrad=(kernel_type.is_1d1d() and accumulate))

        a, b, c, d, ref_d = generate_normal(
            m, n, k,
            major_a, major_b,
            accumulate, out_dtype,
            kernel_type,
            use_ue8m0=use_ue8m0,
            quant_config=quant_config,
        )

        def test_func():
            deep_gemm.fp8_fp4_gemm_nt(
                a, b, d, c=c,
                disable_ue8m0_cast=disable_ue8m0_cast,
                recipe=recipe,
                recipe_a=recipe_a,
                recipe_b=recipe_b,
            )

        test_func()
        diff = float(calc_diff(d, ref_d))
        max_abs = float((d.float() - ref_d.float()).abs().amax().item())
        if diff >= max_diff:
            raise AssertionError(f'Diff too large: {diff:.8f} >= {max_diff}')

        t = bench_kineto(test_func, 'gemm_', suppress_kineto_output=True)
        bytes_count = count_bytes(a, b, d) + count_bytes(c) * int(accumulate)
        result.update({
            'passed': True,
            'diff': f'{diff:.8f}',
            'max_abs': f'{max_abs:.8f}',
            'elapsed_us': f'{t * 1e6:.3f}',
            'tflops': f'{2 * m * n * k / t / 1e12:.3f}' if t > 0 else '',
            'gbps': f'{bytes_count / 1e9 / t:.3f}' if t > 0 else '',
        })
    except Exception as exc:
        result['error'] = repr(exc)

    return result


def write_results(path: str, results: list) -> None:
    fieldnames = ['testcase_id', 'm', 'k', 'n', 'passed', 'diff', 'max_abs', 'elapsed_us', 'tflops', 'gbps', 'error']
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run MXFP8 x MXFP4 GEMM cases from m/k/n CSV.')
    parser.add_argument('--csv', type=str, required=True, help='CSV with testcase_id,m,k,n columns.')
    parser.add_argument('--result-csv', type=str, default='result.csv')
    parser.add_argument('--out-dtype', choices=('bf16', 'fp32'), default='bf16')
    parser.add_argument('--max-diff', type=float, default=0.01)
    parser.add_argument('--accumulate', action='store_true', help='Run D = A @ B.T + C.')
    args = parser.parse_args()

    assert torch.cuda.is_available(), 'CUDA is required'
    assert get_arch_major() == 10, 'MXFP8 x MXFP4 path requires SM100/B200 class GPU'

    out_dtype = torch.bfloat16 if args.out_dtype == 'bf16' else torch.float
    cases = read_cases(args.csv)
    assert cases, f'No cases found in {args.csv}'

    print('Testing custom MXFP8 x MXFP4 GEMM cases:')
    print(' > QuantConfig=(32, 32, False, True), recipe_a=(1, 32), recipe_b=(1, 32)')

    results = []
    for idx, case in enumerate(cases, start=1):
        result = run_case(case, out_dtype, args.max_diff, args.accumulate)
        results.append(result)
        status = 'PASS' if result['passed'] else 'FAIL'
        print(f" > [{idx}/{len(cases)}] id={case['testcase_id']}, "
              f"m={case['m']}, n={case['n']}, k={case['k']}: {status}, "
              f"{result['elapsed_us']} us, {result['tflops']} TFLOPS, "
              f"diff={result['diff']}, error={result['error']}")

    write_results(args.result_csv, results)
    num_passed = sum(int(r['passed']) for r in results)
    print(f' > result csv: {args.result_csv}')
    print(f' > summary: {num_passed}/{len(results)} passed')


if __name__ == '__main__':
    main()
