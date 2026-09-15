import importlib.util
import itertools
import os
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def mega(monkeypatch):
    package = types.ModuleType('_mega_protocol_test')
    package.__path__ = []
    package._C = types.SimpleNamespace(
        get_num_sms=Mock(return_value=120),
        get_token_alignment_for_mega_moe=lambda mma='fp8xfp4': 384 if mma == 'fp4xfp4' else 1920,
        get_symm_buffer_size_for_mega_moe=Mock(return_value=(64, lambda raw: (raw,) * 12)),
    )
    for name in ('fp4_fp4_mega_moe', 'fp8_fp4_mega_moe', 'bf16_mega_moe', 'fp8_mega_moe'):
        setattr(package._C, name, Mock(side_effect=AssertionError('unexpected C launch')))
    utils = types.ModuleType('_mega_protocol_test.utils')
    math = types.ModuleType('_mega_protocol_test.utils.math')
    math.align = lambda n, alignment: (n + alignment - 1) // alignment * alignment
    for name, module in ((package.__name__, package), (utils.__name__, utils), (math.__name__, math)):
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(importlib.util.find_spec('deep_gemm').origin).parent / 'mega/__init__.py'
    spec = importlib.util.spec_from_file_location('_mega_protocol_test.mega', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    for name in ('fp4_fp4_mega_moe', 'fp8_fp4_mega_moe', 'bf16_mega_moe', 'fp8_mega_moe'):
        mock = getattr(package._C, name)
        if mock.side_effect is not None:
            mock.assert_not_called()


def buffer(mega, mma='fp4xfp4', shared=0, cls=None):
    obj = (cls or mega.SymmBuffer).__new__(cls or mega.SymmBuffer)
    obj.group = types.SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    obj.num_experts = 2
    obj.num_max_tokens_per_rank = 768
    obj.num_topk = 1
    obj.hidden = obj.intermediate_hidden = 512
    obj.num_shared_experts = shared
    obj.mma_type = mma
    obj.activation = 'swiglu'
    obj.buffer = torch.empty(64, dtype=torch.int8)
    obj.handle = types.SimpleNamespace(buffer_ptrs=[obj.buffer.data_ptr()])
    if mma == 'fp4xfp4':
        obj._nvfp4_layout_key = mega._nvfp4_layout_key(obj)
    return obj


def weights(e=2, h=512, ih=512, device='cpu'):
    def pair(n, k):
        return (torch.zeros((e, n, k // 2), dtype=torch.int8, device=device),
                torch.empty_strided((e, n, k // 64), (n * (k // 64), 1, n),
                                    dtype=torch.int32, device=device))
    return pair(2 * ih, h), pair(h, ih)


def nv_args(mega, shared=0):
    obj = buffer(mega, shared=shared)
    l1, l2 = weights()
    args = dict(y=torch.empty((2, 512), dtype=torch.bfloat16),
                l1_weights=l1, l2_weights=l2, sym_buffer=obj)
    if shared:
        args.update(shared_l1_weights=torch.empty((1024 * shared, 512), dtype=torch.bfloat16),
                    shared_l2_weights=torch.empty((512, 512 * shared), dtype=torch.bfloat16),
                    x_bf16=torch.empty((7, 512), dtype=torch.bfloat16))
    return args


@pytest.mark.parametrize('name', ['fp8_fp4_mega_moe', 'bf16_mega_moe', 'fp4_fp4_mega_moe'])
def test_main_rejects_sm90(mega, name):
    foreign = buffer(mega, cls=mega.SM90SymmBuffer)
    with pytest.raises(ValueError):
        getattr(mega, name)(None, None, None, foreign)


@pytest.mark.parametrize('mma', ['bf16xbf16', 'fp8xfp4', 'fp4xfp4'])
def test_sm90_rejects_main(mega, mma):
    with pytest.raises(ValueError, match='SM90SymmBuffer'):
        mega.fp8_mega_moe(None, None, None, buffer(mega, mma))


@pytest.mark.parametrize('name', ['fp8_fp4_mega_moe', 'bf16_mega_moe'])
def test_main_rejects_nv(mega, name):
    with pytest.raises(ValueError, match='NVFP4'):
        getattr(mega, name)(None, None, None, buffer(mega))


def test_original_nv_error_precedes_weights(mega):
    with pytest.raises(ValueError, match='requires an fp4xfp4 symmetric buffer'):
        mega.fp4_fp4_mega_moe(None, None, None, buffer(mega, 'fp8xfp4'))


def reuse(mega, base, **changes):
    args = dict(group=base.group, num_experts=base.num_experts,
                num_max_tokens_per_rank=base.num_max_tokens_per_rank, num_topk=base.num_topk,
                hidden=base.hidden, intermediate_hidden=base.intermediate_hidden,
                num_shared_experts=base.num_shared_experts, mma_type=base.mma_type,
                activation=base.activation, base=base)
    args.update(changes)
    return mega.SymmBuffer(**args)


@pytest.mark.parametrize('change', [dict(num_experts=4), dict(num_max_tokens_per_rank=384),
                                  dict(num_topk=2), dict(hidden=1024),
                                  dict(intermediate_hidden=1024), dict(num_shared_experts=1)])
def test_nv_base_rejects_changed_layout_even_with_enough_bytes(mega, change):
    base = buffer(mega)
    with pytest.raises(ValueError, match='identical buffer configuration'):
        reuse(mega, base, **change)
    mega._C.get_symm_buffer_size_for_mega_moe.assert_not_called()


def test_nv_base_exact_and_aligned_equivalent(mega):
    base = buffer(mega)
    for capacity in (768, 767):
        view = reuse(mega, base, num_max_tokens_per_rank=capacity)
        assert view.buffer is base.buffer and view.handle is base.handle
        assert view._nvfp4_layout_key == base._nvfp4_layout_key
        assert isinstance(view._nvfp4_layout_key, tuple)


def test_nv_constructor_records_configured_sms(mega, monkeypatch):
    empty = torch.empty
    monkeypatch.setattr(torch, 'empty', lambda *args, **kwargs: empty(
        *args, **{**kwargs, 'device': 'cpu'}))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    group = types.SimpleNamespace(size=lambda: 1, rank=lambda: 0, barrier=lambda: None)
    obj = mega.SymmBuffer(group, 2, 385, 1, 512, 512, mma_type='fp4xfp4')
    assert obj.num_max_tokens_per_rank == 768
    assert obj._nvfp4_layout_key == (1, 2, 768, 1, 512, 512, 0, 'fp4xfp4', 'swiglu', 120)
    view = reuse(mega, obj)
    assert view.buffer is obj.buffer
    view.destroy()
    obj.destroy()


def test_nv_sms_change_rejected_on_reuse_and_launch(mega):
    args = nv_args(mega)
    mega._C.get_num_sms.return_value = 118
    with pytest.raises(ValueError, match='identical buffer configuration'):
        reuse(mega, args['sym_buffer'])
    with pytest.raises(ValueError, match='SM count'):
        mega.fp4_fp4_mega_moe(**args)


def test_nv_mutated_metadata_rejected(mega):
    args = nv_args(mega)
    args['sym_buffer'].hidden = 1024
    with pytest.raises(ValueError, match='configuration'):
        mega.fp4_fp4_mega_moe(**args)


def test_main_base_shrink_and_situ_reuse(mega):
    base = buffer(mega, 'fp8xfp4')
    base.num_max_tokens_per_rank = 3840
    base.activation = 'situ'
    view = reuse(mega, base, num_max_tokens_per_rank=1920, activation='swiglu')
    assert view.buffer is base.buffer and view.handle is base.handle
    mega._C.get_num_sms.assert_not_called()


@pytest.mark.parametrize('mma', ['fp8xfp4', 'fp4xfp4'])
def test_cross_protocol_base_rejected(mega, mma):
    base = buffer(mega, mma)
    with pytest.raises(ValueError, match='protocols'):
        reuse(mega, base, mma_type='fp4xfp4' if mma == 'fp8xfp4' else 'fp8xfp4')


def test_sm90_base_rejected(mega):
    with pytest.raises(ValueError, match='SymmBuffer'):
        reuse(mega, buffer(mega, cls=mega.SM90SymmBuffer))


@pytest.mark.parametrize('bad', ['none', 'short', 'long', 'not_tensor', 'uint8', 'sf_float',
                                'rank', 'sf_shape', 'sf_stride', 'weight_stride', 'device'])
@pytest.mark.parametrize('operand', ['l1_weights', 'l2_weights'])
def test_nv_bad_pair(mega, bad, operand):
    args = nv_args(mega)
    w, sf = args[operand]
    replacement = {
        'none': None, 'short': (w,), 'long': (w, sf, sf), 'not_tensor': (w, None),
        'uint8': (w.to(torch.uint8), sf), 'sf_float': (w, sf.float()),
        'rank': (w[0], sf), 'sf_shape': (w, sf[:, :-1]),
        'sf_stride': (w, sf.contiguous()), 'weight_stride': (w.transpose(1, 2), sf),
        'device': (w.to('meta'), sf),
    }[bad]
    args[operand] = replacement
    with pytest.raises((TypeError, ValueError)):
        mega.fp4_fp4_mega_moe(**args)


@pytest.mark.parametrize('dims', [(1, 512, 512), (2, 1024, 512), (2, 512, 1024)])
def test_nv_consistent_weights_wrong_buffer_shape(mega, dims):
    args = nv_args(mega)
    args['l1_weights'], args['l2_weights'] = weights(*dims)
    with pytest.raises(ValueError, match='shapes'):
        mega.fp4_fp4_mega_moe(**args)


@pytest.mark.parametrize('present', [p for p in itertools.product((False, True), repeat=3)
                                     if len(set(p)) != 1])
def test_nv_shared_all_or_none(mega, present):
    args = nv_args(mega, shared=1)
    for name, keep in zip(('shared_l1_weights', 'shared_l2_weights', 'x_bf16'), present):
        if not keep:
            args[name] = None
    with pytest.raises(ValueError, match='provided together'):
        mega.fp4_fp4_mega_moe(**args)


@pytest.mark.parametrize('bad', ['tuple', 'rank', 'zero', 'nondivisible', 'shape', 'dtype',
                                'input_type', 'input_rows', 'input_hidden', 'count'])
def test_nv_shared_validation(mega, bad):
    args = nv_args(mega, shared=1)
    if bad == 'tuple':
        args['shared_l1_weights'] = (args['shared_l1_weights'], None)
    elif bad == 'rank':
        args['shared_l2_weights'] = args['shared_l2_weights'][0]
    elif bad in ('zero', 'nondivisible'):
        width = 0 if bad == 'zero' else 513
        args['shared_l2_weights'] = torch.empty((512, width), dtype=torch.bfloat16)
    elif bad == 'shape':
        args['shared_l1_weights'] = args['shared_l1_weights'][:-1]
    elif bad == 'dtype':
        args['shared_l1_weights'] = args['shared_l1_weights'].float()
    elif bad == 'input_type':
        args['x_bf16'] = ()
    elif bad == 'input_rows':
        args['x_bf16'] = args['x_bf16'][:1]
    elif bad == 'input_hidden':
        args['x_bf16'] = torch.empty((7, 1024), dtype=torch.bfloat16)
    else:
        args['shared_l1_weights'] = torch.empty((2048, 512), dtype=torch.bfloat16)
        args['shared_l2_weights'] = torch.empty((512, 1024), dtype=torch.bfloat16)
    with pytest.raises((TypeError, ValueError)):
        mega.fp4_fp4_mega_moe(**args)


@pytest.mark.parametrize('name', ['bf16_mega_moe', 'fp8_fp4_mega_moe'])
@pytest.mark.parametrize('bad', ['missing', 'tuple', 'rank', 'shape'])
def test_main_shared_validation(mega, name, bad):
    obj = buffer(mega, 'fp8xfp4', shared=1)
    l1 = torch.empty((1024, 512), dtype=torch.bfloat16)
    l2 = torch.empty((512, 512), dtype=torch.bfloat16)
    if bad == 'rank':
        l1 = l1[0]
    elif bad == 'shape':
        l1 = l1[:-1]
    if name == 'fp8_fp4_mega_moe':
        l1, l2 = (l1, torch.empty(1)), (l2, torch.empty(1))
    if bad == 'missing':
        l2 = None
    elif bad == 'tuple':
        l1 = (None,)
    with pytest.raises((TypeError, ValueError)):
        getattr(mega, name)(None, None, None, obj, shared_l1_weights=l1, shared_l2_weights=l2)


@pytest.mark.parametrize('name', ['bf16_mega_moe', 'fp8_fp4_mega_moe'])
@pytest.mark.parametrize('allocated,called', [(1, 1), (2, 0), (2, 1), (0, 1)])
def test_main_shared_forwarding_identity(mega, name, allocated, called):
    obj = buffer(mega, 'fp8xfp4', shared=allocated)
    l1 = l2 = None
    if called:
        l1 = torch.empty((1024 * called, 512), dtype=torch.bfloat16)
        l2 = torch.empty((512, 512 * called), dtype=torch.bfloat16)
        if name == 'fp8_fp4_mega_moe':
            l1, l2 = [l1, torch.empty(1)], (l2, torch.empty(1))
    mock = getattr(mega._C, name)
    mock.side_effect = None
    getattr(mega, name)(None, None, None, obj, shared_l1_weights=l1, shared_l2_weights=l2)
    assert mock.call_args.args[3] is l1 and mock.call_args.args[4] is l2
    assert mock.call_args.args[6] is obj.buffer


@pytest.mark.parametrize('shared', [0, 1])
def test_nv_forwarding_identity(mega, shared):
    args = nv_args(mega, shared)
    mock = mega._C.fp4_fp4_mega_moe
    mock.side_effect = None
    mega.fp4_fp4_mega_moe(**args)
    actual = mock.call_args.args
    assert len(actual) == 21
    for index, name in ((0, 'y'), (1, 'l1_weights'), (2, 'l2_weights'),
                        (3, 'shared_l1_weights'), (4, 'shared_l2_weights'), (5, 'x_bf16')):
        assert actual[index] is args.get(name)
    assert actual[10] == 768 and actual[13] == (1, 1, 16)
    assert actual[-4:] == (None, None, None, 1.0)


@pytest.mark.parametrize('name,arity', [('fp8_fp4_mega_moe', 18), ('bf16_mega_moe', 15),
                                      ('fp8_mega_moe', 14)])
def test_other_forwarding_and_subclasses(mega, name, arity):
    parent = mega.SM90SymmBuffer if name == 'fp8_mega_moe' else mega.SymmBuffer
    child = type('BufferSubclass', (parent,), {})
    obj = buffer(mega, 'fp8xfp4', cls=child)
    obj.activation = 'situ'
    mock = getattr(mega._C, name)
    mock.side_effect = None
    y, l1, l2 = object(), object(), object()
    getattr(mega, name)(y, l1, l2, obj, activation='swiglu')
    assert len(mock.call_args.args) == arity
    assert all(a is b for a, b in zip(mock.call_args.args[:3], (y, l1, l2)))


@pytest.mark.parametrize('scheduler', ['mega_moe', 'nvfp4_mega_moe'])
def test_task_info_release_orders_metadata_reads(scheduler):
    include = Path(__file__).resolve().parents[1] / 'deep_gemm/include/deep_gemm'
    source = (include / f'scheduler/{scheduler}.cuh').read_text()
    release = source.split('void release_task_info() const {', 1)[1].split('}', 1)[0]
    assert release.index('ptx::fence_acq_rel_cta();') < release.index(
        'task_info_empty_barriers[sched_stage_idx ^ 1].arrive(0u);')
    helper = (include / 'ptx/ld_st.cuh').read_text()
    assert 'asm volatile("fence.acq_rel.cta;" ::: "memory");' in helper


@pytest.mark.parametrize('kernel', ['sm90_fp8_mega_moe', 'sm100_fp4_fp4_mega_moe'])
def test_nv_moe_launch_preserves_stream_dependency(kernel):
    source = (Path(__file__).resolve().parents[1] /
              f'csrc/jit_kernels/impls/{kernel}.hpp').read_text()
    assert '.enable_pdl = false,' in source


@pytest.mark.parametrize('kernel', ['sm90_fp8_mega_moe', 'sm100_fp4_fp4_mega_moe',
                                  'sm100_bf16_mega_moe', 'sm100_fp8_fp4_mega_moe'])
def test_moe_combine_waits_for_readers_before_refill(kernel):
    source = (Path(__file__).resolve().parents[1] /
              f'deep_gemm/include/deep_gemm/impls/{kernel}.cuh').read_text()
    refill = source.split('const auto move_mask_and_load =', 1)[1].split('return true;', 1)[0]
    assert refill.index('__syncwarp();') < refill.index('cute::elect_one_sync()')
    assert refill.index('cute::elect_one_sync()') < refill.index('ptx::tma_load_1d(')


def test_nvfp4_amax_preserves_nv_dev_functor():
    source = (Path(__file__).resolve().parents[1] /
              'deep_gemm/include/deep_gemm/impls/sm100_fp4_fp4_mega_moe.cuh').read_text()
    assert 'CUTLASS_DEVICE T operator()(T a, T b) const { return a > b ? a : b; }' in source
    assert 'math::ReduceMax<float>()' not in source
    for component in ('x', 'y'):
        assert f'math::warp_reduce<4, true>(thread_local_amax.{component}, ReduceMax<float>())' in source


@pytest.mark.skipif(os.getenv('DEEPGEMM_TEST_NVFP4_REUSE') != '1',
                    reason='opt-in SM100 GPU protocol lifecycle test')
def test_nvfp4_exact_base_reuse_gpu(tmp_path):
    import deep_gemm
    import torch.distributed as dist

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip('requires an SM100-family GPU')
    if dist.is_initialized():
        pytest.skip('requires an isolated process group')
    torch.cuda.set_device(0)
    dist.init_process_group('gloo', init_method=f'file://{tmp_path / "rendezvous"}', rank=0, world_size=1)
    base = view = None
    try:
        group = dist.group.WORLD
        base = deep_gemm.SymmBuffer(group, 2, 384, 1, 512, 512, mma_type='fp4xfp4')
        l1, l2 = weights(device='cuda')
        for _, sf in (l1, l2):
            sf.fill_(0x38383838)
        l1, l2 = deep_gemm.transform_weights_for_mega_moe(l1, l2)

        def launch(obj):
            obj.x[:2].zero_()
            obj.x_sf[:2].fill_(0x38383838)
            obj.topk_idx[:2].zero_()
            obj.topk_weights[:2].fill_(1)
            y = torch.full((2, 512), float('nan'), dtype=torch.bfloat16, device='cuda')
            deep_gemm.fp4_fp4_mega_moe(y, l1, l2, obj)
            torch.cuda.synchronize()
            assert torch.equal(y, torch.zeros_like(y))

        launch(base)
        view = deep_gemm.SymmBuffer(group, 2, 384, 1, 512, 512, mma_type='fp4xfp4', base=base)
        assert view.buffer is base.buffer and view.handle is base.handle
        launch(view)
        launch(base)
    finally:
        if view is not None:
            view.destroy()
        if base is not None:
            base.destroy()
        dist.destroy_process_group()
