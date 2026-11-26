import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import deep_gemm
from deep_gemm.testing import calc_diff, get_arch_major

from generators import (  # noqa: E402
    MajorTypeAB,
    QuantConfig,
    align,
    generate_m_grouped_contiguous,
    get_mk_alignment_for_contiguous_layout,
)


def test_sm90_m_grouped_gemm_contiguous_small_m_recommended_alignment() -> None:
    if get_arch_major() != 9:
        return

    print("Testing SM90 m-grouped contiguous GEMM small-M recommended alignment:")
    quant_config = QuantConfig()
    recipe, recipe_a, recipe_b = quant_config.get_recipes()
    num_groups = 4
    expected_m_per_group = 128
    theoretical_alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(expected_m_per_group)
    assert deep_gemm.get_recommended_mk_alignment_for_contiguous_layout(False, 128, 128, 4) == 64
    assert deep_gemm.get_recommended_mk_alignment_for_contiguous_layout(False, 128, 256, 4) == 64
    assert deep_gemm.get_recommended_mk_alignment_for_contiguous_layout(True, 128, 128, 4) == theoretical_alignment
    assert deep_gemm.get_recommended_mk_alignment_for_contiguous_layout(False, 64, 128, 4) == theoretical_alignment
    assert deep_gemm.get_recommended_mk_alignment_for_contiguous_layout(False, 128, 512, 4) == theoretical_alignment
    assert deep_gemm.get_recommended_mk_alignment_for_contiguous_layout(False, 128, 128, 8) == theoretical_alignment

    for use_psum_layout, n in ((False, 4096), (True, 7168)):
        for k in (128, 256):
            deep_gemm.set_mk_alignment_for_contiguous_layout(
                deep_gemm.get_recommended_mk_alignment_for_contiguous_layout(
                    use_psum_layout, expected_m_per_group, k, num_groups
                )
            )
            m, a, b, grouped_layout, d, ref_d = generate_m_grouped_contiguous(
                num_groups=num_groups,
                expected_m_per_group=expected_m_per_group,
                n=n,
                k=k,
                major_a=MajorTypeAB.KMajor,
                major_b=MajorTypeAB.KMajor,
                use_ue8m0=False,
                use_psum_layout=use_psum_layout,
                quant_config=quant_config,
            )

            deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
                a,
                b,
                d,
                grouped_layout,
                disable_ue8m0_cast=True,
                use_psum_layout=use_psum_layout,
                recipe=recipe,
                recipe_a=recipe_a,
                recipe_b=recipe_b,
            )

            if use_psum_layout:
                for j in range(num_groups):
                    start = 0 if j == 0 else align(grouped_layout[j - 1], get_mk_alignment_for_contiguous_layout())
                    end = grouped_layout[j]
                    diff = calc_diff(d[start:end], ref_d[start:end])
                    assert diff < quant_config.max_diff(), (
                        f"{m=}, {n=}, {k=}, psum={use_psum_layout}, {j=}, {diff:.5f}"
                    )
            else:
                diff = calc_diff(d, ref_d)
                assert diff < quant_config.max_diff(), f"{m=}, {n=}, {k=}, psum={use_psum_layout}, {diff:.5f}"


if __name__ == "__main__":
    test_sm90_m_grouped_gemm_contiguous_small_m_recommended_alignment()
