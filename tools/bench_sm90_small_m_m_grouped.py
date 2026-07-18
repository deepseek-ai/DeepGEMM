import argparse
import random
import sys
from pathlib import Path

import statistics
import torch


ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import deep_gemm  # noqa: E402
from deep_gemm.testing import bench_kineto, calc_diff  # noqa: E402
from generators import MajorTypeAB, QuantConfig, generate_m_grouped_contiguous  # noqa: E402


def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x]


def parse_bool_list(value: str) -> list[bool]:
    values = []
    for item in value.split(","):
        item = item.strip().lower()
        if item in ("1", "true", "t", "yes", "y"):
            values.append(True)
        elif item in ("0", "false", "f", "no", "n"):
            values.append(False)
        elif item:
            raise ValueError(f"Invalid boolean value: {item}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-groups", type=int, default=4)
    parser.add_argument("--num-groups-list", type=parse_int_list, default=None)
    parser.add_argument("--expected-m-per-group", type=int, default=128)
    parser.add_argument("--expected-m-list", type=parse_int_list, default=None)
    parser.add_argument("--n-list", type=parse_int_list, default=parse_int_list("3072,4096,6144,7168"))
    parser.add_argument("--k-list", type=parse_int_list, default=parse_int_list("128,256"))
    parser.add_argument("--psum-list", type=parse_bool_list, default=parse_bool_list("false,true"))
    parser.add_argument("--mk-alignment", type=int, default=128)
    parser.add_argument("--block-n-multiple", type=int, default=1)
    parser.add_argument("--print-configs", action="store_true")
    parser.add_argument("--compare-recommended", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("This benchmark targets SM90 GPUs")

    deep_gemm.set_block_size_multiple_of((1, args.block_n_multiple))
    if args.print_configs:
        import os
        os.environ["DG_PRINT_CONFIGS"] = "1"

    quant_config = QuantConfig()
    recipe, recipe_a, recipe_b = quant_config.get_recipes()
    num_groups_list = args.num_groups_list if args.num_groups_list is not None else [args.num_groups]
    expected_m_list = args.expected_m_list if args.expected_m_list is not None else [args.expected_m_per_group]

    def percentile(values: list[float], pct: float) -> float:
        assert values
        if len(values) == 1:
            return values[0]
        sorted_values = sorted(values)
        rank = (len(sorted_values) - 1) * pct
        lower = int(rank)
        upper = min(lower + 1, len(sorted_values) - 1)
        weight = rank - lower
        return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight

    def bench_case(num_groups: int, expected_m: int, psum: bool, n: int, k: int,
                   mk_alignment: int) -> tuple[int, float, float, float, float, float, float]:
        deep_gemm.set_mk_alignment_for_contiguous_layout(mk_alignment)
        times = []
        last_m = None
        last_diff = None
        for _ in range(args.repeat):
            torch.manual_seed(args.seed)
            random.seed(args.seed)
            m, a, b, grouped_layout, d, ref_d = generate_m_grouped_contiguous(
                num_groups,
                expected_m,
                n,
                k,
                MajorTypeAB.KMajor,
                MajorTypeAB.KMajor,
                use_ue8m0=False,
                use_psum_layout=psum,
                quant_config=quant_config,
            )

            def run():
                deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
                    a,
                    b,
                    d,
                    grouped_layout,
                    disable_ue8m0_cast=True,
                    use_psum_layout=psum,
                    recipe=recipe,
                    recipe_a=recipe_a,
                    recipe_b=recipe_b,
                )

            run()
            diff = calc_diff(d, ref_d)
            if diff >= quant_config.max_diff():
                raise AssertionError((psum, n, k, diff))

            t = bench_kineto(run, "gemm_", suppress_kineto_output=True)
            times.append(t * 1e6)
            last_m = m
            last_diff = diff
        return (
            last_m,
            min(times),
            statistics.median(times),
            sum(times) / len(times),
            percentile(times, 0.9),
            max(times),
            last_diff,
        )

    if args.compare_recommended:
        print(
            "num_groups,expected_m_per_group,psum,n,k,baseline_alignment,recommended_alignment,"
            "baseline_m,recommended_m,baseline_median_us,recommended_median_us,speedup,"
            "baseline_avg_us,recommended_avg_us,baseline_p90_us,recommended_p90_us,diff",
            flush=True,
        )
        for num_groups in num_groups_list:
            for expected_m in expected_m_list:
                for psum in args.psum_list:
                    for n in args.n_list:
                        for k in args.k_list:
                            baseline_alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(expected_m)
                            recommended_alignment = deep_gemm.get_recommended_mk_alignment_for_contiguous_layout(
                                psum, expected_m, k, num_groups
                            )
                            baseline = bench_case(num_groups, expected_m, psum, n, k, baseline_alignment)
                            recommended = bench_case(num_groups, expected_m, psum, n, k, recommended_alignment)
                            baseline_m, _, baseline_median, baseline_avg, baseline_p90, _, _ = baseline
                            recommended_m, _, recommended_median, recommended_avg, recommended_p90, _, diff = recommended
                            print(
                                f"{num_groups},{expected_m},{int(psum)},{n},{k},"
                                f"{baseline_alignment},{recommended_alignment},"
                                f"{baseline_m},{recommended_m},"
                                f"{baseline_median:.3f},{recommended_median:.3f},"
                                f"{baseline_median / recommended_median:.4f},"
                                f"{baseline_avg:.3f},{recommended_avg:.3f},"
                                f"{baseline_p90:.3f},{recommended_p90:.3f},{diff:.6f}",
                                flush=True,
                            )
        return

    print(
        "num_groups,expected_m_per_group,psum,n,k,m,mk_alignment,block_n_multiple,"
        "repeat,min_us,median_us,avg_us,p90_us,max_us,diff",
        flush=True,
    )
    for num_groups in num_groups_list:
        for expected_m in expected_m_list:
            for psum in args.psum_list:
                for n in args.n_list:
                    for k in args.k_list:
                        m, min_us, median_us, avg_us, p90_us, max_us, diff = bench_case(
                            num_groups, expected_m, psum, n, k, args.mk_alignment
                        )
                        print(
                            f"{num_groups},{expected_m},{int(psum)},{n},{k},{m},{args.mk_alignment},"
                            f"{args.block_n_multiple},{args.repeat},"
                            f"{min_us:.3f},{median_us:.3f},{avg_us:.3f},{p90_us:.3f},{max_us:.3f},{diff:.6f}",
                            flush=True,
                        )


if __name__ == "__main__":
    main()
