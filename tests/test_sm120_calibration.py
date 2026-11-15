import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calibrate_sm120_heuristic as calibration


@pytest.mark.parametrize("cache_policy", ("steady", "cold"))
def test_timing_excludes_host_submission_delays(cache_policy):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")

    a = torch.ones((64, 64), device="cuda")
    b = torch.ones_like(a)
    output = torch.empty_like(a)

    def run():
        time.sleep(0.02)
        torch.mm(a, b, out=output)

    samples = calibration.measure_samples(
        run, cache_policy=cache_policy, warmups=0, repeats=3, iterations=4)

    assert len(samples) == 3
    assert all(0 < sample < 10_000 for sample in samples), samples
    assert torch.equal(output, torch.full_like(output, 64))


def test_fp8_small_m_calibration_uses_effective_descriptor():
    shape = (16, 2048, 2048)
    assert calibration.effective_shape("fp8", shape) == (2048, 16, 2048)

    layouts = {
        (block_m, block_n, block_k)
        for block_m, block_k, block_n in calibration.candidate_layouts("fp8", shape)
    }
    assert (64, 16, 64) in layouts
    assert (128, 16, 64) in layouts

    # The API swaps only FP8 inputs with M <= 16.
    assert calibration.effective_shape("fp8", (17, 2048, 2048)) == (17, 2048, 2048)
    assert calibration.effective_shape("bf16", (16, 2048, 2048)) == (16, 2048, 2048)


def test_small_m_fp8_split_k_and_cycle_reconstruction():
    record = {
        "dtype": "fp8",
        "shape": (16, 1024, 2048),
        "layout": (64, 16, 128),
        "sm_count": 70,
        "num_waves": 1,
        "num_stages": 9,
    }
    record["split_k"] = calibration.split_k_factor(record)
    assert record["split_k"] == 2
    assert int(calibration.model_features(record) @ calibration.CURRENT_FEATURE_COEFFICIENTS) == 13433


def test_worker_json_payload_is_independent_of_native_stdout():
    output = (
        "SM120GemmConfig(block_m=64, block_n=32, block_k=64, num_stages=8, "
        "LayoutInfo(num_waves=1, last_wave_util=2, num_cycles=16441))\n"
        "shared memDG_CALIB_JSON:{native output split this line\n"
    )
    payload = json.dumps({"status": "ok", "shape": [64, 64, 1024]})

    record, config = calibration.parse_worker_output(output, payload)

    assert record["status"] == "ok"
    assert config == (64, 32, 64, 8, 1, 2, 16441)


def test_run_candidate_rejects_a_different_selected_layout(tmp_path, monkeypatch):
    shape = (64, 64, 1024)
    args = argparse.Namespace(
        dtype="bf16", cache_policy="steady", warmups=0, repeats=3,
        iterations=1, seed=0, cache_dir=tmp_path,
    )

    def fake_worker(command, **_kwargs):
        result_path = Path(command[command.index("--result-file") + 1])
        result_path.write_text(json.dumps({"status": "ok"}))
        stdout = (
            "block_m=64, block_n=32, block_k=64, num_stages=8, "
            "LayoutInfo(num_waves=1, last_wave_util=2, num_cycles=16441)\n"
        )
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(calibration.subprocess, "run", fake_worker)
    with pytest.raises(RuntimeError) as exc_info:
        calibration.run_candidate(args, shape, (128, 64, 128))
    assert str(exc_info.value) == "forced layout mismatch: requested (128, 64, 128), got (64, 32, 64)"


def test_run_candidate_includes_worker_record_on_failure(tmp_path, monkeypatch):
    args = argparse.Namespace(
        dtype="fp8", cache_policy="steady", warmups=0, repeats=3,
        iterations=1, seed=0, cache_dir=tmp_path,
    )
    record = {"status": "incorrect", "diff": 0.2, "tolerance": 0.001}

    def failed_worker(command, **_kwargs):
        result_path = Path(command[command.index("--result-file") + 1])
        result_path.write_text(json.dumps(record))
        return SimpleNamespace(stdout="worker diagnostics", stderr="", returncode=1)

    monkeypatch.setattr(calibration.subprocess, "run", failed_worker)
    with pytest.raises(RuntimeError) as exc_info:
        calibration.run_candidate(args, (16, 1024, 2048), (64, 16, 128))
    assert '"diff": 0.2' in str(exc_info.value)
    assert '"tolerance": 0.001' in str(exc_info.value)


@pytest.mark.parametrize("cache_policy", ("steady", "cold"))
@pytest.mark.parametrize(
    ("dtype", "shape", "layout", "expected_status"),
    (
        ("bf16", (64, 64, 1024), (64, 64, 64), "ok"),
        ("bf16", (64, 64, 1024), (1, 1, 1), "invalid"),
        ("fp8", (16, 1024, 2048), None, "ok"),
        ("fp8", (16, 1024, 2048), (128, 16, 128), "ok"),
        ("fp8", (16, 1024, 2048), (1, 1, 1), "invalid"),
    ),
)
def test_sm120_worker_candidates(dtype, shape, layout, expected_status, cache_policy, tmp_path):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires an SM120 GPU")

    args = argparse.Namespace(
        dtype=dtype, cache_policy=cache_policy, warmups=0, repeats=3,
        iterations=1, seed=0, cache_dir=tmp_path,
    )
    record = calibration.run_candidate(args, shape, layout)

    assert record["status"] == expected_status
    if expected_status == "ok":
        assert calibration.is_acceptable_diff(
            record["replay_diff"], 1e-5 if dtype == "bf16" else 0.001)
    if expected_status == "ok" and layout is not None:
        assert record["layout"] == layout
    if dtype == "fp8" and shape[0] <= 16 and expected_status == "ok":
        assert record["effective_shape"] == (shape[1], shape[0], shape[2])


def test_sm120_worker_suppresses_diagnostics_during_timing(tmp_path, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires an SM120 GPU")

    monkeypatch.setenv("DG_JIT_DEBUG", "1")
    monkeypatch.setenv("DJ_JIT_DEBUG", "1")
    original_run = calibration.subprocess.run
    outputs = []

    def capture_worker_output(command, **kwargs):
        completed = original_run(command, **kwargs)
        outputs.append(completed.stdout + completed.stderr)
        return completed

    monkeypatch.setattr(calibration.subprocess, "run", capture_worker_output)
    for warmups, iterations in ((0, 1), (2, 4)):
        args = argparse.Namespace(
            dtype="bf16", cache_policy="steady", warmups=warmups, repeats=3,
            iterations=iterations, seed=0, cache_dir=tmp_path,
        )
        record = calibration.run_candidate(args, (64, 64, 1024), (64, 64, 64))
        assert record["layout"] == (64, 64, 64)

    counts = [output.count("Making TMA desc:") for output in outputs]
    assert counts[0] > 0
    assert counts[1] == counts[0]


def test_fit_and_evaluate_preserve_shape_order():
    shapes = ((256, 1024, 8192), (64, 128, 2048), (2048, 1024, 2048))
    records = []
    for shape_index, shape in enumerate(shapes):
        for candidate_index, (layout, stages, waves, median_us) in enumerate((
            ((128, 64, 128), 3, 2 + shape_index, 12.0 + shape_index),
            ((128, 128, 128), 2, 1 + shape_index, 10.0 + shape_index),
        )):
            records.append({
                "dtype": "fp8",
                "shape": list(shape),
                "layout": layout,
                "sm_count": 70,
                "num_stages": stages,
                "num_waves": waves,
                "split_k": 1,
                "mode": "forced",
                "candidate_index": candidate_index,
                "median_us": median_us,
            })

    fitted_forward, _ = calibration.fit_coefficients(records, shapes[:2])
    fitted_reversed, _ = calibration.fit_coefficients(list(reversed(records)), shapes[:2])
    assert np.allclose(fitted_forward, fitted_reversed)

    evaluation = calibration.evaluate(list(reversed(records)), shapes, calibration.CURRENT_FEATURE_COEFFICIENTS)
    assert [entry["shape"] for entry in evaluation["selections"]] == list(shapes)


def test_sm120_rejects_nonpositive_forced_layout(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires an SM120 GPU")

    import deep_gemm

    monkeypatch.setenv("DG_JIT_FORCE_LAYOUT", "0x64x64")
    a = torch.ones((64, 64), device="cuda", dtype=torch.bfloat16)
    b = torch.ones((64, 64), device="cuda", dtype=torch.bfloat16)
    d = torch.empty((64, 64), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError) as exc_info:
        deep_gemm.bf16_gemm_nt(a, b, d)
    assert "DG_JIT_FORCE_LAYOUT must use BMxBNxBK format" in str(exc_info.value)


@pytest.mark.parametrize("shape", ((64, 64, 1024), (16, 1024, 2048)))
def test_sm120_fp8_worker_inputs_exclude_scale_packing_kernels(shape):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires an SM120 GPU")

    from deep_gemm.testing import calc_diff

    torch.manual_seed(0)
    run, d, reference, tolerance = calibration.prepare_worker_inputs("fp8", *shape)
    run()
    torch.cuda.synchronize()
    assert calibration.is_acceptable_diff(float(calc_diff(d, reference)), tolerance)

    if int(os.environ.get("DG_USE_NVIDIA_TOOLS", "0")):
        pytest.skip("profiler disabled under DG_USE_NVIDIA_TOOLS")

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        run()
        torch.cuda.synchronize()

    kernel_names = [event.key for event in prof.key_averages()]
    assert any("sm120_fp8_fp4_gemm_1d1d" in name for name in kernel_names), kernel_names
    assert not any(
        "pack_fp32_into_ue8m0" in name or "index_select" in name
        for name in kernel_names
    ), kernel_names
