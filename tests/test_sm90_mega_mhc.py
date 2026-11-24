"""SM90 Mega mHC regressions: CUDA_VISIBLE_DEVICES=0 python tests/test_sm90_mega_mhc.py."""

import unittest
import torch

import deep_gemm

from test_mega_mhc import make_inputs, make_outputs, logical_outputs, run_mega_mhc, mhc_reference
from deep_gemm.utils import per_token_cast_to_fp8

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
    raise unittest.SkipTest("requires SM90")


def check(inputs, outputs, layout, quant_input_bf16=None):
    shared_block = outputs.get("shared_sf_block_m", 0)
    expected = logical_outputs(mhc_reference(**inputs, sf_layout=layout,
                                             shared_sf_block_m=shared_block), shared_block)
    actual = logical_outputs(outputs)
    rounded = outputs.get("y_bf16") if quant_input_bf16 is None else quant_input_bf16
    quant_sf = None
    if "y_fp8" in actual:
        # TF32 and reduction order can move a BF16 norm result across a scale
        # exponent boundary. First check norm accuracy independently below;
        # then require exact quantization and layout for the actual rounded data.
        assert rounded is not None
        quantized, quant_sf = per_token_cast_to_fp8(rounded, True, 32, True)
        assert torch.equal(actual["y_fp8"].view(torch.uint8), quantized.view(torch.uint8))
    if "y_fp8" in actual and outputs.get("y_bf16") is None:
        actual.pop("y_bf16", None)
    for name, tensor in actual.items():
        ref = expected[name]
        if tensor.dtype == torch.int32:
            assert torch.equal(tensor, quant_sf), name
            continue
        a, b = tensor.float(), ref.float()
        diff = (a - b).square().sum() / (a.square().sum() + b.square().sum()).clamp_min(1e-30)
        assert torch.isfinite(a).all(), name
        assert diff < (2e-4 if name == "y_fp8" else 5e-5), (name, diff.item())


def check_modes_and_tails(shifted, tokens, hidden, layout):
    inputs, _ = make_inputs(tokens, hidden, 20261118, shifted)
    outputs = make_outputs(inputs, layout)
    run_mega_mhc(inputs, outputs)
    torch.cuda.synchronize()
    check(inputs, outputs, layout)


def test_modes_and_tails():
    for shifted in (False, True):
        for tokens, hidden, layout in ((1, 4096, "bf16"), (32, 5120, "bf16"),
                                       (65, 5120, "col"), (129, 7168, "extra")):
            print("CASE", shifted, tokens, hidden, layout, flush=True)
            check_modes_and_tails(shifted, tokens, hidden, layout)


def check_fp8_only(shifted):
    inputs, _ = make_inputs(65, 5120, 123, shifted)
    paired = make_outputs(inputs, "col")
    run_mega_mhc(inputs, paired)
    torch.cuda.synchronize()
    check(inputs, paired, "col")
    outputs = make_outputs(inputs, "col")
    outputs["y_bf16"] = None
    run_mega_mhc(inputs, outputs)
    torch.cuda.synchronize()
    check(inputs, outputs, "col", quant_input_bf16=paired["y_bf16"])


def test_fp8_only():
    for shifted in (False, True):
        check_fp8_only(shifted)


def test_tf32_shared_a_swizzle():
    # Each projection selects one exactly representable input. This catches
    # shared A row/column swizzle mistakes independently of random tolerance.
    inputs, _ = make_inputs(65, 5120, 199, True)
    inputs["fn"].zero_()
    inputs["mix_scales"].fill_(1.)
    inputs["mix_bases"].zero_()
    for n in range(24):
        inputs["fn"][n, n * 257] = 1.0
    outputs = make_outputs(inputs, "bf16")
    run_mega_mhc(inputs, outputs)
    torch.cuda.synchronize()
    check(inputs, outputs, "bf16")


def check_same_capture_stream_concurrent_replay(token_counts, shifted=True):
    # Upstream #503: graph nodes captured on the same stream must own different
    # split state when replayed concurrently on different streams.
    fixtures = []
    for seed, tokens in zip((31, 47), token_counts):
        inputs, _ = make_inputs(tokens, 5120, seed, shifted)
        outputs = make_outputs(inputs, "bf16")
        run_mega_mhc(inputs, outputs)
        fixtures.append((inputs, outputs))
    torch.cuda.synchronize()
    graphs = []
    for inputs, outputs in fixtures:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_mega_mhc(inputs, outputs)
        graphs.append(graph)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for _ in range(10):
        for inputs, _outputs in fixtures:
            inputs["x"].normal_()
        for graph, stream in zip(graphs, streams):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph.replay()
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        for inputs, outputs in fixtures:
            check(inputs, outputs, "bf16")


def test_same_capture_stream_concurrent_replay():
    check_same_capture_stream_concurrent_replay((65, 65))


def test_normal_concurrent_replay():
    check_same_capture_stream_concurrent_replay((65, 65), shifted=False)


def test_prefill_concurrent_replay():
    # Exercise different partial-workspace sizes in the SM90 prefill heuristic.
    check_same_capture_stream_concurrent_replay((1024, 4096))


def test_decode_concurrent_replay():
    check_same_capture_stream_concurrent_replay((32, 32))


def test_decode_fp8_and_cooperative_reuse():
    old_sms = deep_gemm.get_num_sms()
    try:
        # 27 uneven splits and two tokens per Mix WG exercise scratch reuse.
        deep_gemm.set_num_sms(32)
        check_modes_and_tails(True, 64, 5120, "bf16")
        deep_gemm.set_num_sms(old_sms)
        for layout in ("col", "extra"):
            check_modes_and_tails(True, 32, 5120, layout)
        check_modes_and_tails(True, 32, 8192, "bf16")
    finally:
        deep_gemm.set_num_sms(old_sms)


def test_deterministic_repeated_launch():
    inputs, _ = make_inputs(65, 5120, 11, True)
    a, b = make_outputs(inputs, "bf16"), make_outputs(inputs, "bf16")
    try:
        deep_gemm.use_deterministic_algorithms(True)
        run_mega_mhc(inputs, a)
        run_mega_mhc(inputs, b)
        torch.cuda.synchronize()
        for name in logical_outputs(a):
            assert torch.equal(a[name], b[name]), name
    finally:
        deep_gemm.use_deterministic_algorithms(False)


def test_deterministic_across_batch_and_grid():
    inputs, _ = make_inputs(65, 5120, 73, True)
    small = dict(inputs)
    for name in ("x", "residual", "post_mix", "comb_res_mix", "shifted_prev_mix"):
        small[name] = inputs[name][:32]
    expected = make_outputs(inputs, "bf16")
    old_sms = deep_gemm.get_num_sms()
    try:
        deep_gemm.use_deterministic_algorithms(True)
        run_mega_mhc(inputs, expected)
        for sms in (old_sms, 32):
            deep_gemm.set_num_sms(sms)
            actual = make_outputs(small, "bf16")
            run_mega_mhc(small, actual)
            torch.cuda.synchronize()
            for name, tensor in logical_outputs(actual).items():
                assert torch.equal(tensor, expected[name][:32]), (sms, name)
    finally:
        deep_gemm.set_num_sms(old_sms)
        deep_gemm.use_deterministic_algorithms(False)


if __name__ == "__main__":
    suite = unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in globals().copy().items()
                               if name.startswith("test_") and callable(fn))
    result = unittest.TextTestRunner(verbosity=2, failfast=True).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
