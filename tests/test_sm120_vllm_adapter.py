"""Exercise adapter dispatch with metadata doubles, without importing CUDA/vLLM.

These tests cover host fallback decisions, not GPU arithmetic or framework ABI.
"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class TensorMetadata:
    def __init__(self, shape, dtype, *, device='cuda:0', address=4096, contiguous=True):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.ndim, self.is_cuda = len(shape), device.startswith('cuda:')
        self.address, self.contiguous = address, contiguous

    def is_contiguous(self):
        return self.contiguous

    def data_ptr(self):
        return self.address


def module(name, **attributes):
    result = types.ModuleType(name)
    result.__dict__.update(attributes)
    return result


class AdapterDispatchTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        events = self.events

        class Native:
            def __init__(self, moe):
                self.moe = moe
                self.unquantized_backend = 'triton'

            @classmethod
            def register_oot(cls, implementation):
                cls.implementation = implementation
                return implementation

            def forward_native(self, *args):
                events.append(('native', args))
                return 'native-result'

        class Slots:
            def __getitem__(self, grid):
                return lambda *args, **kwargs: events.append(('slots', args))

        def launch(*args, **kwargs):
            # Match the private C++ API's metadata preconditions.
            if any(t.data_ptr() % 16 for t in args[:5]):
                raise RuntimeError('Private API requires 16-byte alignment')
            if any(t.device != args[0].device for t in args[:5]):
                raise RuntimeError('Private API requires one device')
            events.append(('fused', args, kwargs))

        configuration = types.SimpleNamespace(
            scheduler_config=types.SimpleNamespace(async_scheduling=False),
            parallel_config=types.SimpleNamespace(enable_dbo=False))
        cuda = types.SimpleNamespace(is_current_stream_capturing=lambda: False,
            get_device_properties=lambda device: types.SimpleNamespace(
                major=12, minor=0, name='NVIDIA GeForce RTX 5090', multi_processor_count=170))
        fake_torch = module('torch', bfloat16='bf16', float32='fp32', int32='int32', int64='int64', cuda=cuda)
        logger = types.SimpleNamespace(info=lambda *a: None, info_once=lambda *a: None)
        self.modules = {
            'torch': fake_torch,
            'triton': module('triton', next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
            'deep_gemm': module('deep_gemm', _C=types.SimpleNamespace(_sm120_moe_small=launch)),
            'vllm.config': module('vllm.config', get_current_vllm_config=lambda: configuration),
            'vllm.logger': module('vllm.logger', init_logger=lambda name: logger),
            'vllm.model_executor.layers.fused_moe.activation': module('activation',
                MoEActivation=types.SimpleNamespace(SILU='silu')),
            'vllm.model_executor.layers.fused_moe.oracle.unquantized': module('unquantized',
                UnquantizedMoeBackend=types.SimpleNamespace(TRITON='triton', FLASHINFER_CUTLASS='cutlass')),
            'vllm.model_executor.layers.fused_moe.runner.shared_experts': module('shared_experts',
                SharedExpertsOrder=types.SimpleNamespace(MK_INTERNAL_OVERLAPPED='internal')),
            'vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method': module('method',
                UnquantizedFusedMoEMethod=Native),
        }
        self.start_patch(patch.dict(sys.modules, self.modules))
        self.start_patch(patch.dict('os.environ', {'SM120_VLLM_ENABLE': '1', 'SM120_VLLM_TP_ENABLE': '0'}))
        self.start_patch(patch('importlib.metadata.version', return_value='0.32.0'))
        self.load('deep_gemm.small_moe_policy', ROOT / 'deep_gemm/small_moe_policy.py')
        self.load('sm120_tp_policy', ROOT / 'examples/vllm_sm120/sm120_tp_policy.py')
        adapter = self.load('adapter_dispatch_under_test', ROOT / 'examples/vllm_sm120/sm120_vllm.py')
        adapter.slot_kernel = lambda: Slots()
        adapter.register()
        pc = types.SimpleNamespace(dp_size=1, ep_size=1, pcp_size=1, sp_size=1, tp_size=1, enable_eplb=False)
        moe = types.SimpleNamespace(moe_parallel_config=pc, has_bias=False, is_lora_enabled=False,
            has_hash_routing=False, _defer_moe_finalize=False, activation='silu', swiglu_limit=None,
            swiglu_alpha=None, swiglu_beta=None, num_experts=128, experts_per_token=8)
        self.method = Native.implementation(moe)
        self.layer = types.SimpleNamespace(
            w13_weight=TensorMetadata((128, 1536, 2048), 'bf16'),
            w2_weight=TensorMetadata((128, 2048, 768), 'bf16'),
            global_num_experts=128, apply_router_weight_on_input=False, expert_map=None)
        from deep_gemm.small_moe_policy import select_single_token_config
        self.method.sm120_workspace = {
            'route_shape': (1, 8), 'direct_ids': True, 'swapped': False,
            'config': select_single_token_config(1, 128, 8, 2048, 768),
            'h': TensorMetadata((128, 128, 768), 'bf16'), 'y': TensorMetadata((1, 2048), 'bf16'),
            'slots': TensorMetadata((1, 8), 'int64'), 'route_weights': TensorMetadata((1, 8), 'fp32')}
        self.x = TensorMetadata((1, 2048), 'bf16')
        self.weights = TensorMetadata((1, 8), 'fp32')

    def load(self, name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        instance = importlib.util.module_from_spec(spec)
        self.start_patch(patch.dict(sys.modules, {name: instance}))
        spec.loader.exec_module(instance)
        return instance

    def start_patch(self, patcher):
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_offset_expert_ids_retain_native_without_partial_fused_execution(self):
        for dtype, offset in (('int32', 4), ('int64', 8)):
            with self.subTest(dtype=dtype):
                self.events.clear()
                ids = TensorMetadata((1, 8), dtype, address=4096 + offset)
                result = self.method.forward_native(self.layer, self.x, self.weights, ids, None, None)
                self.assertEqual(result, 'native-result')
                self.assertEqual([event[0] for event in self.events], ['native'])

    def test_valid_slot_and_direct_dispatch_preserve_order_and_plans(self):
        for direct in (False, True):
            for dtype in ('int32', 'int64'):
                with self.subTest(direct=direct, dtype=dtype):
                    self.events.clear()
                    self.method.sm120_workspace['direct_ids'] = direct
                    # The Triton slot preparation can read offset IDs; only
                    # direct IDs are passed through the stricter private API.
                    offset = 4 if dtype == 'int32' else 8
                    ids = TensorMetadata((1, 8), dtype, address=4096 if direct else 4096 + offset)
                    shared = lambda *args: self.events.append(('shared', args))
                    result = self.method.forward_native(self.layer, self.x, self.weights, ids, shared, self.x)
                    expected = ([] if direct else ['slots']) + ['fused', 'shared', 'fused']
                    self.assertEqual([event[0] for event in self.events], expected)
                    calls = [event for event in self.events if event[0] == 'fused']
                    self.assertEqual(calls[0][1][5:], (True, 32, 3, 2))
                    self.assertEqual(calls[1][1][5:], (False, 16, 4, 8))
                    self.assertIs(result, self.method.sm120_workspace['y'])
                    self.assertEqual(calls[1][2], {'expert_ids': True} if direct else {})

    def test_weight_placement_mismatch_is_rejected_before_workspace_creation(self):
        self.assertIsNone(self.method._support_reason(self.layer))
        for device in ('cpu', 'cuda:1'):
            with self.subTest(device=device):
                self.layer.w2_weight = TensorMetadata((128, 2048, 768), 'bf16', device=device)
                self.assertIsNotNone(self.method._support_reason(self.layer))

    def test_registration_remains_disabled_by_default(self):
        with patch.dict('os.environ', {'SM120_VLLM_ENABLE': '0'}):
            adapter = self.load('adapter_disabled_under_test', ROOT / 'examples/vllm_sm120/sm120_vllm.py')
            with patch('importlib.metadata.version', side_effect=AssertionError('Unexpected vLLM import')):
                adapter.register()
            self.assertFalse(adapter._registered)


if __name__ == '__main__':
    unittest.main()
