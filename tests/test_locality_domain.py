import os
import shutil
import subprocess
import sys
import tempfile
import time
import torch
from pathlib import Path
from typing import Callable

import deep_gemm
from deep_gemm import locality_domain
from deep_gemm.testing import assert_bitwise_equal
from deep_gemm.utils import mlopart

NUM_SMS_PER_TPC = 2


def assert_raises(fn: Callable[[], None]) -> None:
    try:
        fn()
    except RuntimeError:
        return
    raise AssertionError('Expected a `RuntimeError`')


def random_bytes(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    num_bytes = shape[-1] * torch.tensor([], dtype=dtype).element_size()
    return torch.randint(0, 256, (*shape[:-1], num_bytes), dtype=torch.uint8, device='cuda').view(dtype)


def test_sm_locality_domains() -> None:
    print('Testing SM locality domains:')
    sm_domain = locality_domain.get_sm_locality_domains()
    balanced = locality_domain.get_balanced_sm_locality_domains()
    num_sms = sm_domain.numel()

    # Both domains are present, and the SMs of a TPC share theirs
    assert sm_domain.dtype == balanced.dtype == torch.uint8 and 0 < sm_domain.sum().item() < num_sms
    for t in (sm_domain, balanced):
        tpcs = t.view(-1, NUM_SMS_PER_TPC)
        assert torch.equal(tpcs[:, 0], tpcs[:, 1])

    # Balancing moves the fewest TPCs from the larger domain
    assert balanced.sum().item() * 2 == num_sms
    num_moved_tpcs = (balanced != sm_domain).view(-1, NUM_SMS_PER_TPC).all(1).sum().item()
    assert num_moved_tpcs == abs(sm_domain.sum().item() - num_sms // 2) // NUM_SMS_PER_TPC
    print(f' > {torch.bincount(sm_domain.long(), minlength=2).tolist()} SMs per domain, balanced {torch.bincount(balanced.long(), minlength=2).tolist()}')
    print()


def test_empty() -> None:
    print('Testing tensors in one locality domain:')
    num_domains = deep_gemm.get_num_locality_domains()
    for domain_idx in range(num_domains):
        shape, dtype = (3, 4608, 5120), torch.float8_e4m3fn
        plain = random_bytes(shape, dtype)
        t = locality_domain.empty(shape, dtype, domain_idx)
        assert t.shape == shape and t.dtype == dtype and t.device == plain.device and t.is_contiguous()
        t.copy_(plain)
        assert_bitwise_equal(t, plain)
        print(f' > {dtype} {shape} in domain {domain_idx}')
    assert_raises(lambda: locality_domain.empty((1, ), torch.uint8, num_domains))
    print()


def as_plain(localized: torch.Tensor, dim: int = -2) -> torch.Tensor:
    # `(num_domains, ..., D / num_domains, ...)` back to the plain tensor sliced along `dim`
    dim = dim % (localized.dim() - 1)
    return localized.movedim(0, dim).flatten(dim, dim + 1)


def test_localize() -> None:
    print('Testing `localize`:')
    granularity, num_domains = locality_domain.get_granularity(), deep_gemm.get_num_locality_domains()
    # The query starts the MLOPart processes
    assert deep_gemm.is_localization_available()
    for dtype, shape, dim in ((torch.float8_e4m3fn, (7, 4608, 5120), -2), (torch.int8, (4, 6144, 3584), -2), (torch.bfloat16, (3, 5120, 2304), -2),
                              (torch.float8_e4m3fn, (4608, 5120), -2), (torch.float8_e4m3fn, (384, 4608, 5120), -2), (torch.int8, (4, 6144, 3584), 0)):
        plain, dim = random_bytes(shape, dtype), dim % len(shape)
        start = time.perf_counter()
        localized = deep_gemm.localize(plain, dim=dim)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        assert localized.shape == (num_domains, *shape[:dim], shape[dim] // num_domains, *shape[dim + 1:]) and localized.dtype == dtype
        assert all(localized[d].is_contiguous() for d in range(num_domains))
        num_slice_bytes = localized.stride(0) * localized.element_size()
        assert num_slice_bytes % granularity == 0 and num_slice_bytes - granularity < localized[0].nbytes <= num_slice_bytes
        assert deep_gemm.is_localized(localized) and deep_gemm.is_localized(torch.nn.Parameter(localized, requires_grad=False))
        assert deep_gemm.is_localized(localized[:, :1]) and not deep_gemm.is_localized(torch.empty(localized.shape, dtype=dtype, device='cuda'))
        assert not deep_gemm.is_localized(plain) and not deep_gemm.is_localized(localized[0])
        for bad in (plain.mT, localized[..., :1]):
            assert_raises(lambda: deep_gemm.is_localized(bad))

        # Round trip, and an expert moved with a plain copy
        assert_bitwise_equal(as_plain(localized, dim), plain)
        if len(shape) == 3 and dim == 1:
            src_idx, dst_idx = shape[0] - 1, shape[0] // 2
            localized[:, dst_idx].copy_(localized[:, src_idx])
            plain[dst_idx] = plain[src_idx]
            assert_bitwise_equal(as_plain(localized, dim), plain)
        print(f' > {dtype} {shape} along dim {dim}: {num_domains} x {num_slice_bytes >> 20} MB localized in {elapsed * 1e3:.0f} ms')
    deep_gemm.destroy_localizer()
    print()


def test_m_grouped_gemm_contiguous_localized() -> None:
    print('Testing m-grouped contiguous GEMM with localized weights:')
    from generators import (MajorTypeAB, QuantConfig, generate_k_grouped_contiguous,
                            generate_m_grouped_contiguous, generate_m_grouped_masked, reset_seed)

    num_sms = deep_gemm.get_num_sms()

    reset_seed()
    for quant_config in QuantConfig.get_list_from_dtype(torch.float8_e4m3fn):
        recipe, recipe_a, recipe_b = quant_config.get_recipes()
        for num_groups, expected_m_per_group, n, k in ((4, 8192, 4096, 7168), (8, 4096, 7168, 2048),
                                                       (32, 512, 3072, 4096), (3, 64, 1024, 512), (2, 64, 512, 512)):
            m, a, b, grouped_layout, d, ref_d, valid_mask = generate_m_grouped_contiguous(
                num_groups, expected_m_per_group, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor, use_ue8m0=True, quant_config=quant_config)
            # Leave a group empty, so that the domain boundary is crossed with fewer M blocks
            grouped_layout[grouped_layout == 0] = -1
            valid_mask &= grouped_layout >= 0

            def launch(weights: torch.Tensor, out: torch.Tensor) -> None:
                deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(a, (weights, b[1]), out, grouped_layout,
                                                               recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)

            def run(weights: torch.Tensor) -> torch.Tensor:
                out = torch.full_like(d, float('nan'))
                launch(weights, out)
                return out

            localized = deep_gemm.localize(b[0])
            plain_d, localized_d = run(b[0]), run(localized)
            assert not localized_d[valid_mask].isnan().any()
            assert_bitwise_equal(localized_d[valid_mask], plain_d[valid_mask])
            print(f' > {quant_config.gran_k_a=}, {quant_config.gran_k_b=}, fp4=({quant_config.is_fp4_a}, {quant_config.is_fp4_b}), '
                  f'{num_groups=}, {m=}, {n=}, {k=}: bitwise equal')

    # Unsupported: invalid N split, psum layout, and other grouped GEMM APIs
    m, a, b, grouped_layout, d, _, _ = generate_m_grouped_contiguous(
        2, 256, 768, 512, MajorTypeAB.KMajor, MajorTypeAB.KMajor, use_ue8m0=True)
    assert_raises(lambda: deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
        a, (deep_gemm.localize(b[0]), b[1]), d, grouped_layout))

    _, a, b, grouped_layout, d, _, _ = generate_m_grouped_contiguous(
        2, 256, 1024, 512, MajorTypeAB.KMajor, MajorTypeAB.KMajor, use_ue8m0=True,
        use_psum_layout=True)
    assert_raises(lambda: deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
        a, (deep_gemm.localize(b[0]), b[1]), d, grouped_layout, use_psum_layout=True))

    _, a, b, grouped_layout, d, _, _ = generate_m_grouped_contiguous(
        2, 256, 1024, 512, MajorTypeAB.KMajor, MajorTypeAB.KMajor, use_ue8m0=True)
    nn_b = (deep_gemm.localize(b[0].transpose(1, 2).contiguous(), dim=-1),
            b[1].transpose(1, 2).contiguous())
    assert_raises(lambda: deep_gemm.m_grouped_fp8_fp4_gemm_nn_contiguous(a, nn_b, d, grouped_layout))

    masked_a, masked_b, masked_m, masked_d, _, _ = generate_m_grouped_masked(
        2, 256, 128, 1024, 512, use_ue8m0=True)
    assert_raises(lambda: deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(
        masked_a, (deep_gemm.localize(masked_b[0]), masked_b[1]), masked_d, masked_m, 128))

    _, k_a, k_b, k_c, k_d, _, k_layout, host_ks = generate_k_grouped_contiguous(
        2, 128, 512, MajorTypeAB.MNMajor, MajorTypeAB.MNMajor, [256, 256], use_ue8m0=True)
    assert_raises(lambda: deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
        k_a, (deep_gemm.localize(k_b[0]), k_b[1]), k_d, host_ks, k_layout, k_c))

    # CUDA Graph capture after the kernel and its locality schedule have been initialized
    _, a, b, grouped_layout, d, _, valid_mask = generate_m_grouped_contiguous(
        2, 256, 1024, 512, MajorTypeAB.KMajor, MajorTypeAB.KMajor, use_ue8m0=True)
    localized = deep_gemm.localize(b[0])

    def launch_graph(out: torch.Tensor) -> None:
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(a, (localized, b[1]), out, grouped_layout)

    expected = torch.full_like(d, float('nan'))
    launch_graph(expected)
    capture_stream = torch.cuda.Stream()
    graph_out = torch.full_like(d, float('nan'))
    with torch.cuda.stream(capture_stream):
        launch_graph(graph_out)
    capture_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        launch_graph(graph_out)
    graph.replay()
    capture_stream.synchronize()
    assert_bitwise_equal(graph_out[valid_mask], expected[valid_mask])
    print(' > CUDA Graph: bitwise equal')

    # Fewer SMs, including the minimum even split and unequal cluster counts across domains
    _, a, b, grouped_layout, d, _, valid_mask = generate_m_grouped_contiguous(
        4, 1024, 2048, 1024, MajorTypeAB.KMajor, MajorTypeAB.KMajor, use_ue8m0=True)
    plain_d = d.clone()
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, plain_d, grouped_layout)
    localized = deep_gemm.localize(b[0])
    try:
        for test_num_sms in (num_sms - 20, 4, 6):
            deep_gemm.set_num_sms(test_num_sms)
            out = torch.full_like(d, float('nan'))
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(a, (localized, b[1]), out, grouped_layout)
            assert not out[valid_mask].isnan().any()
            assert_bitwise_equal(out[valid_mask], plain_d[valid_mask])
            print(f' > {test_num_sms} SMs: bitwise equal')

        deep_gemm.set_num_sms(2)
        assert_raises(lambda: deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            a, (localized, b[1]), d, grouped_layout))
        print(' > 2 SMs: rejected')
    finally:
        deep_gemm.set_num_sms(num_sms)
    deep_gemm.destroy_localizer()
    print()


def test_mlopart() -> None:
    print('Testing the MPS MLOPart processes:')
    # On their own: start, serve a descriptor per domain and quit
    device_uuid = bytes(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid.bytes)
    start = time.perf_counter()
    processes = mlopart.MLOPart(device_uuid)
    for domain_idx in range(deep_gemm.get_num_locality_domains()):
        os.close(processes.create_memory(locality_domain.get_granularity(), domain_idx))
    served = time.perf_counter()
    processes.close()
    assert processes.client.returncode == 0
    print(f' > started and served in {(served - start) * 1e3:.0f} ms, quit in {(time.perf_counter() - served) * 1e3:.0f} ms')

    # Behind `localize`: the memory outlives `destroy_localizer`, and the next `localize` restarts them
    plain = random_bytes((4, 4608, 5120), torch.float8_e4m3fn)
    localized = deep_gemm.localize(plain)
    deep_gemm.destroy_localizer()
    assert_bitwise_equal(as_plain(localized), plain)
    start = time.perf_counter()
    assert_bitwise_equal(as_plain(deep_gemm.localize(plain)), plain)
    deep_gemm.destroy_localizer()
    print(f' > restarted for a new tensor in {(time.perf_counter() - start) * 1e3:.0f} ms')
    print()


def test_without_mlopart() -> None:
    print('Testing without MLOPart:')
    # MPS predating MLOPart ignores `-mlopart`: emulate it by stripping the flag before the real control CLI
    real = shutil.which('nvidia-cuda-mps-control')
    script = ('import warnings, torch, deep_gemm\n'
              'with warnings.catch_warnings(record=True) as caught:\n'
              '    warnings.simplefilter("always")\n'
              '    assert not deep_gemm.is_localization_available()\n'
              'assert len(caught) == 1 and "MLOPart is unavailable" in str(caught[0].message)\n'
              'assert deep_gemm.locality_domain.get_sm_locality_domains().numel() == torch.cuda.get_device_properties(0).multi_processor_count\n'
              'try:\n'
              '    deep_gemm.localize(torch.zeros((2, 16), dtype=torch.uint8, device="cuda"))\n'
              'except AssertionError as e:\n'
              '    assert "MLOPart is unavailable" in str(e)\n'
              'else:\n'
              '    raise AssertionError("`localize` must fail")\n')
    with tempfile.TemporaryDirectory() as bin_dir:
        fake = Path(bin_dir, 'nvidia-cuda-mps-control')
        fake.write_text(f'#!/bin/sh\n[ "$1" = -f ] && exec {real} "$@"\nsed "s/ -mlopart//" | exec {real} "$@"\n')
        fake.chmod(0o755)
        result = subprocess.run([sys.executable, '-c', script], env={**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}'}, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0, result.stderr
    assert '1 MLOPart devices' in result.stderr, result.stderr
    print(' > detected, `localize` refused, SM table available')
    print()


if __name__ == '__main__':
    torch.manual_seed(0)

    test_sm_locality_domains()
    test_empty()
    test_localize()
    test_m_grouped_gemm_contiguous_localized()
    test_mlopart()
    test_without_mlopart()
