#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import tempfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_SHAPES = (
    (64, 64, 1024),
    (128, 256, 2048),
    (512, 512, 2048),
    (384, 1024, 4096),
    (2048, 512, 4096),
    (4096, 1024, 2048),
)
VALIDATION_SHAPES = (
    (64, 128, 2048),
    (256, 1024, 8192),
    (2048, 1024, 2048),
)
CURRENT_FEATURE_COEFFICIENTS = np.array((0.07, 120.0, 2000.0, 1.0))
FEATURE_NAMES = ("tma_bytes", "sync_work", "waves", "split_k_cycles")
CONFIG_RE = re.compile(
    r"block_m=(\d+), block_n=(\d+), block_k=(\d+).*?"
    r"num_stages=(\d+).*?LayoutInfo\(num_waves=(\d+), "
    r"last_wave_util=(\d+), num_cycles=(\d+)"
)


def parse_shape(value: str) -> tuple[int, int, int]:
    try:
        shape = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must use MxNxK format") from exc
    if len(shape) != 3 or min(shape) <= 0:
        raise argparse.ArgumentTypeError("shape must contain three positive dimensions")
    return shape


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep SM120 GEMM layouts and fit the heuristic cost model.")
    parser.add_argument("--dtype", choices=("fp8", "bf16"), default="fp8")
    parser.add_argument("--shape", action="append", type=parse_shape,
                        help="custom MxNxK shape; repeat for multiple shapes")
    parser.add_argument("--cache-policy", choices=("steady", "cold"), default="steady")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--iterations", type=int, default=50,
                        help="launches per steady-state timing sample")
    parser.add_argument("--seed", type=int, default=0,
                        help="tensor and candidate-order seed")
    parser.add_argument("--output", type=Path,
                        help="optional JSONL output path")
    parser.add_argument("--cache-dir", type=Path,
                        help="JIT cache directory; defaults to a new /tmp directory")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--layout", type=parse_shape, help=argparse.SUPPRESS)
    return parser


def emit(record: dict) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def run_worker(args: argparse.Namespace) -> int:
    os.environ["DG_PRINT_CONFIGS"] = "1"
    if args.layout is None:
        os.environ.pop("DG_JIT_FORCE_LAYOUT", None)
    else:
        os.environ["DG_JIT_FORCE_LAYOUT"] = "x".join(map(str, args.layout))

    sys.path.insert(0, str(REPO_ROOT))

    import torch

    import deep_gemm
    from deep_gemm.testing import calc_diff

    if torch.cuda.get_device_capability()[0] != 12:
        emit({"status": "error", "error": "SM120 GPU required"})
        return 1

    torch.manual_seed(args.seed)
    m, n, k = args.shape[0]
    if args.dtype == "bf16":
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        d = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        reference = (a.float() @ b.float().T).to(torch.bfloat16)

        def run() -> None:
            deep_gemm.bf16_gemm_nt(a, b, d)

        tolerance = 1e-5
    else:
        sys.path.insert(0, str(REPO_ROOT / "tests"))
        from generators import KernelType, MajorTypeAB, generate_normal

        a, b, _, d, reference = generate_normal(
            m, n, k,
            MajorTypeAB.KMajor, MajorTypeAB.KMajor,
            False, torch.bfloat16, KernelType.Kernel1D1D,
            use_ue8m0=True,
        )

        def run() -> None:
            deep_gemm.fp8_fp4_gemm_nt(a, b, d)

        tolerance = 0.001

    try:
        run()
        torch.cuda.synchronize()
    except RuntimeError as exc:
        if "DG_JIT_FORCE_LAYOUT is not a valid candidate" in str(exc):
            emit({"status": "invalid", "shape": (m, n, k), "layout": args.layout})
            return 0
        raise

    diff = float(calc_diff(d, reference))
    if diff >= tolerance:
        emit({
            "status": "incorrect",
            "shape": (m, n, k),
            "layout": args.layout,
            "diff": diff,
            "tolerance": tolerance,
        })
        return 1

    for _ in range(args.warmups):
        run()
    torch.cuda.synchronize()

    cache = None
    if args.cache_policy == "cold":
        cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")

    samples_us = []
    iterations = args.iterations if args.cache_policy == "steady" else 1
    for _ in range(args.repeats):
        if cache is not None:
            cache.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            run()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000 / iterations)

    emit({
        "status": "ok",
        "shape": (m, n, k),
        "layout": args.layout,
        "dtype": args.dtype,
        "cache_policy": args.cache_policy,
        "diff": diff,
        "samples_us": samples_us,
        "median_us": statistics.median(samples_us),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
    })
    return 0


def candidate_layouts(dtype: str, shape: tuple[int, int, int]):
    m, n, _ = shape
    block_ms = (128, 64)
    block_ns = (16, 32) if n <= 32 else ((32, 64, 96, 128) if dtype == "bf16" else (64, 128))
    elem_size = 2 if dtype == "bf16" else 1
    block_ks = (64 // elem_size, 128 // elem_size) if m >= 2048 else (128 // elem_size,)
    return itertools.product(block_ms, block_ks, block_ns)


def parse_worker_output(output: str) -> tuple[dict, tuple[int, ...] | None]:
    record = None
    for line in reversed(output.splitlines()):
        if line.startswith("{"):
            record = json.loads(line)
            break
    if record is None:
        raise RuntimeError(f"worker produced no JSON record:\n{output}")

    match = CONFIG_RE.search(output.replace("\n", " "))
    config = tuple(map(int, match.groups())) if match else None
    return record, config


def run_candidate(args: argparse.Namespace, shape: tuple[int, int, int],
                  layout: tuple[int, int, int] | None) -> dict:
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker", "--dtype", args.dtype,
        "--shape", "x".join(map(str, shape)),
        "--cache-policy", args.cache_policy,
        "--warmups", str(args.warmups),
        "--repeats", str(args.repeats),
        "--iterations", str(args.iterations),
        "--seed", str(args.seed),
    ]
    if layout is not None:
        command.extend(("--layout", "x".join(map(str, layout))))

    env = os.environ.copy()
    env["DG_PRINT_CONFIGS"] = "1"
    env["DG_JIT_CACHE_DIR"] = str(args.cache_dir)
    env.pop("DG_JIT_FORCE_LAYOUT", None)
    completed = subprocess.run(command, env=env, text=True, capture_output=True)
    output = completed.stdout + completed.stderr
    record, config = parse_worker_output(output)
    if completed.returncode != 0 or record["status"] not in ("ok", "invalid"):
        raise RuntimeError(output)
    if record["status"] == "invalid":
        return record
    if config is None:
        raise RuntimeError(f"worker produced no config:\n{output}")

    block_m, block_n, block_k, stages, waves, last_wave_util, predicted_cycles = config
    record.update({
        "layout": (block_m, block_n, block_k),
        "num_stages": stages,
        "num_waves": waves,
        "last_wave_util": last_wave_util,
        "predicted_cycles": predicted_cycles,
        "mode": "auto" if layout is None else "forced",
    })
    record["split_k"] = split_k_factor(record)
    reconstructed_cycles = int(model_features(record) @ CURRENT_FEATURE_COEFFICIENTS)
    if reconstructed_cycles != predicted_cycles:
        raise RuntimeError(
            f"model reconstruction mismatch: {reconstructed_cycles} != {predicted_cycles}")
    return record


def split_k_factor(record: dict) -> int:
    if record["dtype"] == "bf16":
        return 1

    m, n, k = record["shape"]
    block_m, block_n, block_k = record["layout"]
    num_mn_blocks = math.ceil(m / block_m) * math.ceil(n / block_n)
    if num_mn_blocks >= record["sm_count"] // 2:
        return 1

    num_k_blocks = math.ceil(k / block_k)
    split_k = math.ceil((record["sm_count"] * 3 // 4) / num_mn_blocks)
    sf_tile_kblocks = 4 * 128 // block_k
    if sf_tile_kblocks == 0:
        return 1
    while split_k > 1 and (
        num_k_blocks % split_k != 0
        or (num_k_blocks // split_k) % sf_tile_kblocks != 0
    ):
        split_k -= 1

    split_k = min(split_k, num_k_blocks // (2 * sf_tile_kblocks))
    max_workspace_bytes = 32 * 1024 * 1024
    split_k = min(split_k, max(max_workspace_bytes // (m * n * 4), 1))
    return max(split_k, 1)


def model_features(record: dict) -> np.ndarray:
    m, n, k = record["shape"]
    block_m, block_n, block_k = record["layout"]
    elem_size = 2 if record["dtype"] == "bf16" else 1
    sf_bytes = 0
    if record["dtype"] == "fp8":
        sf_bytes = math.ceil(block_m * 4 / 128) * 128 + math.ceil(block_n * 4 / 128) * 128
    tma_bytes_per_kblock = (block_m + block_n) * block_k * elem_size + sf_bytes
    kblocks = math.ceil(k / block_k) // record["split_k"]
    waves = record["num_waves"]
    split_overhead = 0.0
    if record["split_k"] > 1:
        split_overhead = 5000.0 + 0.01 * m * n
    return np.array((
        waves * kblocks * tma_bytes_per_kblock,
        waves * kblocks / math.sqrt(record["num_stages"]),
        waves,
        split_overhead,
    ), dtype=float)


def nnls(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    num_columns = x.shape[1]
    best_coefficients = np.zeros(num_columns)
    best_error = float(y @ y)
    for mask in range(1, 1 << num_columns):
        columns = [index for index in range(num_columns) if mask & (1 << index)]
        coefficients, _, _, _ = np.linalg.lstsq(x[:, columns], y, rcond=None)
        if np.any(coefficients < 0):
            continue
        residual = y - x[:, columns] @ coefficients
        error = float(residual @ residual)
        if error < best_error:
            best_error = error
            best_coefficients = np.zeros(num_columns)
            best_coefficients[columns] = coefficients
    return best_coefficients


def select_layout(records: list[dict], coefficients: np.ndarray) -> dict:
    best = records[0]
    best_score = float(model_features(best) @ coefficients)
    for candidate in records[1:]:
        candidate_score = float(model_features(candidate) @ coefficients)
        ratio = candidate_score / best_score if best_score > 0 else 1.0
        candidate_layout = candidate["layout"]
        best_layout = best["layout"]
        choose = ratio < 0.95
        if 0.95 <= ratio <= 1.05:
            if candidate_layout[1] != best_layout[1]:
                choose = candidate_layout[1] > best_layout[1]
            elif candidate_layout[2] != best_layout[2]:
                choose = candidate_layout[2] < best_layout[2]
            elif candidate_layout[0] != best_layout[0]:
                choose = candidate_layout[0] > best_layout[0]
            else:
                choose = candidate_score < best_score
        if choose:
            best = candidate
            best_score = candidate_score
    return best


def fit_coefficients(records: list[dict], fit_shapes: set[tuple[int, int, int]]) -> tuple[np.ndarray, dict]:
    x_rows = []
    y_rows = []
    for shape in fit_shapes:
        candidates = [record for record in records if tuple(record["shape"]) == shape and record["mode"] == "forced"]
        features = np.stack([model_features(record) for record in candidates])
        times = np.array([record["median_us"] for record in candidates])
        x_rows.append(features - features.mean(axis=0))
        y_rows.append(times - times.mean())
    x = np.concatenate(x_rows)
    y = np.concatenate(y_rows)
    scale = np.linalg.norm(x, axis=0)
    active = scale > 0
    if not np.any(active):
        raise RuntimeError("candidate matrix does not vary any model features")
    normalized_x = x[:, active] / scale[active]
    rank = int(np.linalg.matrix_rank(normalized_x))
    active_coefficients = nnls(normalized_x, y) / scale[active]
    coefficients = np.zeros(x.shape[1])
    coefficients[active] = active_coefficients
    condition_number = float(np.linalg.cond(normalized_x))
    return coefficients, {
        "feature_rank": rank,
        "active_features": [name for name, enabled in zip(FEATURE_NAMES, active) if enabled],
        "full_model_rank": rank == len(FEATURE_NAMES),
        "condition_number": condition_number if math.isfinite(condition_number) else None,
    }


def evaluate(records: list[dict], shapes: set[tuple[int, int, int]],
             coefficients: np.ndarray) -> dict:
    regrets = []
    selections = []
    for shape in shapes:
        candidates = sorted(
            (record for record in records
             if tuple(record["shape"]) == shape and record["mode"] == "forced"),
            key=lambda record: record["candidate_index"],
        )
        selected = select_layout(candidates, coefficients)
        fastest = min(candidates, key=lambda record: record["median_us"])
        regret = selected["median_us"] / fastest["median_us"] - 1
        regrets.append(regret)
        selections.append({
            "shape": shape,
            "selected": selected["layout"],
            "fastest": fastest["layout"],
            "regret": regret,
        })
    return {
        "mean_regret": statistics.mean(regrets),
        "max_regret": max(regrets),
        "selections": selections,
    }


def hardware_manifest() -> dict:
    query = (
        "name,pci.device_id,compute_cap,memory.total,driver_version,"
        "clocks.current.sm,clocks.max.sm,power.limit,temperature.gpu"
    )
    completed = subprocess.run(
        ("nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"),
        text=True, capture_output=True, check=True)
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": completed.stdout.strip(),
    }


def write_record(handle, record: dict) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    if handle is not None:
        handle.write(line + "\n")
        handle.flush()


def run_sweep(args: argparse.Namespace) -> int:
    if args.warmups < 0 or args.repeats < 3 or args.iterations < 1:
        raise SystemExit("warmups must be nonnegative; repeats >= 3; iterations >= 1")
    if args.cache_dir is None:
        args.cache_dir = Path(tempfile.mkdtemp(prefix="deepgemm-sm120-calibration-"))
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if args.shape:
        if len(args.shape) < 3 or len(set(args.shape)) != len(args.shape):
            raise SystemExit("custom calibration requires at least three distinct shapes")
        shapes = tuple(args.shape)
        split = max(1, math.ceil(len(shapes) * 2 / 3))
        calibration_shapes = set(shapes[:split])
        validation_shapes = set(shapes[split:])
    else:
        shapes = CALIBRATION_SHAPES + VALIDATION_SHAPES
        calibration_shapes = set(CALIBRATION_SHAPES)
        validation_shapes = set(VALIDATION_SHAPES)

    output_handle = args.output.open("w") if args.output is not None else None
    try:
        manifest = {
            "type": "manifest",
            "git_sha": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=REPO_ROOT, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(
                ("git", "status", "--porcelain"), cwd=REPO_ROOT, text=True).strip()),
            "dtype": args.dtype,
            "cache_policy": args.cache_policy,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "iterations": args.iterations,
            "seed": args.seed,
            "jit_cache_dir": str(args.cache_dir),
            **hardware_manifest(),
        }
        write_record(output_handle, manifest)

        records = []
        for shape in shapes:
            auto = run_candidate(args, shape, None)
            records.append(auto)
            write_record(output_handle, auto)
            candidates = [
                (block_m, block_n, block_k)
                for block_m, block_k, block_n in candidate_layouts(args.dtype, shape)
            ]
            candidate_indices = {layout: index for index, layout in enumerate(candidates)}
            random.Random(
                args.seed + shape[0] * 1_000_003 + shape[1] * 1_009 + shape[2]
            ).shuffle(candidates)
            for layout in candidates:
                record = run_candidate(args, shape, layout)
                if record["status"] == "invalid":
                    continue
                record["candidate_index"] = candidate_indices[layout]
                records.append(record)
                write_record(output_handle, record)

        fitted, fit_diagnostics = fit_coefficients(records, calibration_shapes)
        relative_mads = []
        for record in records:
            if record["mode"] != "forced":
                continue
            median = record["median_us"]
            relative_mads.append(
                statistics.median(abs(sample - median) for sample in record["samples_us"]) / median)
        fit_diagnostics["median_relative_mad"] = statistics.median(relative_mads)
        fit_diagnostics["max_relative_mad"] = max(relative_mads)
        normalized = fitted / fitted[0] if fitted[0] > 0 else (None, None, None)
        fitted_constants = None
        if fit_diagnostics["full_model_rank"] and fitted[3] > 0:
            fitted_constants = {
                "kCyPerTmaByte": fitted[0] / fitted[3],
                "kSyncBaseCy": fitted[1] / fitted[3],
                "kBlockOverheadCy": fitted[2] / fitted[3],
            }
        current_validation = evaluate(records, validation_shapes, CURRENT_FEATURE_COEFFICIENTS)
        fitted_validation = evaluate(records, validation_shapes, fitted)
        summary = {
            "type": "summary",
            "fitted_feature_coefficients_us": fitted.tolist(),
            "fitted_constants_cycles": fitted_constants,
            "fit_diagnostics": fit_diagnostics,
            "fitted_ratios": {
                "tma": 1.0 if fitted[0] > 0 else None,
                "sync_over_tma": normalized[1],
                "block_over_tma": normalized[2],
            },
            "current_ratios": {
                "tma": 1.0,
                "sync_over_tma": CURRENT_FEATURE_COEFFICIENTS[1] / CURRENT_FEATURE_COEFFICIENTS[0],
                "block_over_tma": CURRENT_FEATURE_COEFFICIENTS[2] / CURRENT_FEATURE_COEFFICIENTS[0],
            },
            "current_validation": current_validation,
            "fitted_validation": fitted_validation,
            "scope": "SKU-local measurement; not a global default recommendation",
        }
        write_record(output_handle, summary)
    finally:
        if output_handle is not None:
            output_handle.close()
    return 0


def main() -> int:
    args = make_parser().parse_args()
    if args.worker:
        if len(args.shape or ()) != 1:
            raise SystemExit("worker requires exactly one --shape")
        return run_worker(args)
    return run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
