import torch

import deep_gemm
from deep_gemm.testing import (
    assert_bitwise_equal,
    calc_diff,
    get_arch_major,
)
from deep_gemm.utils import align


def sm120_bf16_matrix(values, padded=False, offset=0, extra=5, tma=True):
    from sm120_test_storage import native_matrix
    return native_matrix(values, padded, offset, extra, tma)


def exercise_sm120_bf16_native(layout, shape, out_dtype, alpha, c_mode,
                                padded=False, offset=0, graph=False, pdl=False):
    assert get_arch_major() == 12
    assert layout in ('nt', 'nn', 'tn', 'tt') and c_mode in ('none', 'same', 'different')
    m, n, k = shape
    generator = torch.Generator(device='cpu').manual_seed(1907 + m + n + k)
    a_values = (torch.randn((m, k), generator=generator) * 0.25).to(torch.bfloat16)
    b_values = (torch.randn((k, n), generator=generator) * 0.25).to(torch.bfloat16)
    c_values = torch.randn((m, n), generator=generator).to(out_dtype)
    a_physical = a_values if layout[0] == 'n' else a_values.T.contiguous()
    b_physical = b_values if layout[1] == 'n' else b_values.T.contiguous()
    a, a_storage = sm120_bf16_matrix(a_physical, padded, offset)
    b, b_storage = sm120_bf16_matrix(b_physical, padded, offset)
    d, d_storage = sm120_bf16_matrix(torch.zeros_like(c_values), padded, offset, extra=7)
    if c_mode == 'different':
        c, c_storage = sm120_bf16_matrix(c_values, padded, offset, extra=11, tma=False)
        assert c.data_ptr() != d.data_ptr()
        if padded:
            assert c.stride(0) != d.stride(0)
    else:
        c, c_storage = (d, d_storage) if c_mode == 'same' else (None, None)
    initial_c = c_values.cuda()
    d_valid = torch.zeros_like(d_storage, dtype=torch.bool)
    d_valid.as_strided(d.shape, d.stride(), d.storage_offset()).fill_(True)
    if padded:
        assert a.stride(0) > a.size(1) and b.stride(0) > b.size(1)
        assert d.stride(0) > n
    if offset:
        assert all(t.storage_offset() * t.element_size() == offset * 16 for t in (a, b, d))
    function = getattr(deep_gemm, f'bf16_gemm_{layout}')

    def run():
        if c_mode == 'same':
            d.copy_(initial_c)
        function(a, b, d, c=c, alpha=alpha)

    def reference():
        result = a_values.float() @ b_values.float()
        result *= 1.0 if alpha is None else alpha
        if c_mode != 'none':
            result += c_values.float()
        return result

    def check():
        actual = d.cpu().float()
        expected = reference()
        assert torch.isfinite(actual).all()
        if alpha == 0:
            exact = torch.zeros_like(c_values) if c_mode == 'none' else c_values
            torch.testing.assert_close(d.cpu(), exact, rtol=0, atol=0)
        else:
            rtol, atol = (0.008, 0.015) if out_dtype == torch.bfloat16 else (2e-5, 2e-5)
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            diff = calc_diff(actual, expected)
            assert diff < (1e-5 if out_dtype == torch.bfloat16 else 1e-10), f'{diff=}'
        assert (d_storage[~d_valid] == 19).all(), 'Output store touched padding or offset guards'
        torch.testing.assert_close(a.cpu(), a_values if layout[0] == 'n' else a_values.T, rtol=0, atol=0)
        torch.testing.assert_close(b.cpu(), b_values if layout[1] == 'n' else b_values.T, rtol=0, atol=0)
        if c_mode == 'different':
            torch.testing.assert_close(c.cpu(), c_values, rtol=0, atol=0)

    original_pdl = deep_gemm.get_pdl()
    deep_gemm.use_deterministic_algorithms(True)
    try:
        deep_gemm.set_pdl(pdl)
        for _ in range(3):
            d.fill_(float('nan'))
            run()
            check()
        if graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            check()
            torch.cuda.synchronize()
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=stream):
                run()
            for phase in range(3):
                if phase:
                    a_values = (-0.5 * a_values.float()).to(torch.bfloat16)
                    b_values = (b_values.float() + 0.25).to(torch.bfloat16)
                    c_values = (-c_values.float()).to(out_dtype)
                    a.copy_(a_values if layout[0] == 'n' else a_values.T)
                    b.copy_(b_values if layout[1] == 'n' else b_values.T)
                    initial_c.copy_(c_values)
                    if c_mode == 'different':
                        c.copy_(c_values)
                d.fill_(float('nan'))
                captured.replay()
                torch.cuda.synchronize()
                check()
    finally:
        deep_gemm.set_pdl(original_pdl)
        # This fixture starts and ends in the known nondeterministic default state.
        deep_gemm.use_deterministic_algorithms(False)
    print(f' > Native SM120 BF16: {layout=}, {shape=}, {out_dtype=}, {alpha=}, '
          f'{c_mode=}, {padded=}, {offset=}, {graph=}, {pdl=}')


def sm120_grouped_intervals(lengths, alignment):
    intervals, end = [], 0
    for length in lengths:
        start = (end + alignment - 1) // alignment * alignment
        end = start + length
        intervals.append((start, end))
    return intervals


def exercise_sm120_grouped_bf16(mode, alignment, groups, n=64, k=72,
                                 zero_padding=False, all_empty=False, graph=False, pdl=False, nn=False):
    assert get_arch_major() == 12 and mode in ('labels', 'psum', 'masked')
    assert alignment in (32, 64, 128) and not (mode == 'masked' and (nn or zero_padding))
    expected_m = {32: 32, 64: 33, 128: 65}[alignment]
    assert deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(expected_m) == alignment
    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    old_pdl = deep_gemm.get_pdl()
    max_m = alignment + 3
    capacity = groups * 2 * alignment + alignment
    m = max_m if mode == 'masked' else capacity
    generator = torch.Generator().manual_seed(731 + groups + alignment)
    a_shape = (groups, m, k) if mode == 'masked' else (m, k)
    original_a = (torch.randn(a_shape, generator=generator) * 0.25).to(torch.bfloat16)
    original_b = (torch.randn((groups, n, k), generator=generator) * 0.25).to(torch.bfloat16)
    a = original_a.cuda()
    b = (original_b.transpose(1, 2).contiguous() if nn else original_b).cuda()
    if mode == 'masked':
        storage = torch.full((8 + groups * m * n + 16,), 19, dtype=torch.bfloat16, device='cuda')
        d = storage.as_strided((groups, m, n), (m * n, n, 1), 8)
    else:
        d, storage = sm120_bf16_matrix(torch.zeros((m, n), dtype=torch.bfloat16), True, 1, extra=7)
    guard = torch.zeros_like(storage, dtype=torch.bool)
    guard.as_strided(d.shape, d.stride(), d.storage_offset()).fill_(True)
    metadata = torch.empty((groups if mode != 'labels' else m,), dtype=torch.int32, device='cuda')

    def inputs(phase):
        lengths = [0 if all_empty or phase == 1 or g % 3 == 0 else
                   (alignment + 1 if g % 3 == 1 else alignment - 3) for g in range(groups)]
        if phase == 2 and not all_empty:
            lengths = [alignment - 1 if g % 2 == 0 else 0 for g in range(groups)]
        valid = torch.zeros(d.shape[:-1], dtype=torch.bool)
        av = original_a.clone() * (-1 if phase == 2 else 1)
        bv = original_b.clone() * (0.5 if phase == 2 else 1)
        expected = torch.full(d.shape, 0.0 if zero_padding else 7.0, dtype=torch.float32)
        if mode == 'masked':
            meta = lengths
            for g, length in enumerate(lengths):
                valid[g, :length] = True
                expected[g, :length] = av[g, :length].float() @ bv[g].float().T
        else:
            intervals = sm120_grouped_intervals(lengths, alignment)
            labels = torch.full((m,), -1, dtype=torch.int32)
            for g, (start, end) in enumerate(intervals):
                assert start % alignment == 0 and end <= m
                labels[start:end] = g
                valid[start:end] = True
                expected[start:end] = av[start:end].float() @ bv[g].float().T
            meta = labels if mode == 'labels' else [end for _, end in intervals]
        av[~valid] = float('nan')
        a.copy_(av)
        b.copy_(bv.transpose(1, 2) if nn else bv)
        metadata.copy_(torch.as_tensor(meta, dtype=torch.int32))
        return expected, valid

    def run():
        if mode == 'masked':
            deep_gemm.m_grouped_bf16_gemm_nt_masked(a, b, d, metadata, expected_m)
        else:
            kwargs = dict(use_psum_layout=mode == 'psum', ensure_zero_padding=zero_padding)
            if mode == 'psum' and not nn:
                kwargs['expected_m_for_psum_layout'] = 1
            getattr(deep_gemm, f'm_grouped_bf16_gemm_{"nn" if nn else "nt"}_contiguous')(
                a, b, d, metadata, **kwargs)

    def check(expected, valid):
        actual = d.cpu().float()
        assert torch.isfinite(actual[valid]).all()
        torch.testing.assert_close(actual[valid], expected[valid], rtol=0.008, atol=0.015)
        if zero_padding:
            zero_rows = ~valid
            if mode == 'psum':
                zero_rows = torch.zeros_like(valid)
                for end in metadata.cpu().tolist():
                    zero_rows[end:(end + alignment - 1) // alignment * alignment] = True
            torch.testing.assert_close(actual[zero_rows], torch.zeros_like(actual[zero_rows]), rtol=0, atol=0)
        assert (storage[~guard] == 19).all(), 'Grouped output modified storage outside D'

    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        deep_gemm.set_pdl(pdl)
        expected, valid = inputs(0)
        for _ in range(3):
            d.fill_(7)
            run()
            check(expected, valid)
        if graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=stream):
                run()
            for phase in (0, 1, 2):
                expected, valid = inputs(phase)
                d.fill_(7)
                captured.replay()
                torch.cuda.synchronize()
                check(expected, valid)
    finally:
        deep_gemm.set_pdl(old_pdl)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)
    print(f' > Native SM120 grouped BF16: {mode=}, {alignment=}, {groups=}, {n=}, {k=}, '
          f'{zero_padding=}, {all_empty=}, {graph=}, {pdl=}, {nn=}')


def sm120_k_grouped_lengths(groups, alignment, psum, phase=0, empty=False):
    lengths = [((1, 127, 129, 0)[g % 4] if psum else (alignment, 0, 2 * alignment, 0)[g % 4])
               for g in range(groups)]
    if empty or (psum and phase == 1):
        return [0] * groups
    if psum and phase == 2:
        return list(reversed(lengths))
    return lengths


def exercise_sm120_k_grouped(api, groups, alignment, psum, out_dtype, c_mode,
                             shape=(48, 80), gran=128, packed=False, graph=False,
                             pdl=False, empty=False, ks_mode='list', default_recipe=False):
    assert get_arch_major() == 12 and api in ('bf16_tn', 'fp8_tn', 'fp8_nt')
    assert groups in (1, 3, 8) and alignment in (128, 256)
    assert not (api == 'fp8_nt' and (psum or alignment != 128))
    assert not packed or (api != 'bf16_tn' and gran == 32)
    assert c_mode in ('none', 'same', 'different') and ks_mode in ('list', 'none', 'empty')
    assert psum or ks_mode == 'list'
    assert not default_recipe or gran == 128
    m, n = shape
    quant = api != 'bf16_tn'
    assert not quant or (m % 4 == 0 and n % 4 == 0)
    lengths0 = sm120_k_grouped_lengths(groups, alignment, psum, empty=empty)
    padded0 = [(length + alignment - 1) // alignment * alignment for length in lengths0]
    total = sum(padded0)
    ks_cpu = padded0 if not psum or ks_mode == 'list' else (None if ks_mode == 'none' else [])
    assert not (graph and psum and ks_mode == 'list')
    dtype = torch.float8_e4m3fn if quant else torch.bfloat16
    a_shape, b_shape = ((m, total), (n, total)) if api == 'fp8_nt' else ((total, m), (total, n))
    a, b = torch.empty(a_shape, device='cuda', dtype=dtype), torch.empty(b_shape, device='cuda', dtype=dtype)
    sa = torch.empty((total // gran, m), device='cuda') if quant else None
    sb = torch.empty((total // gran, n), device='cuda') if quant else None
    metadata = torch.empty(groups, dtype=torch.int32, device='cuda')
    storage = torch.full((groups * m * n + 32,), 19, dtype=out_dtype, device='cuda')
    d = storage[16:-16].view(groups, m, n)
    c = torch.empty_like(d) if c_mode == 'different' else (d if c_mode == 'same' else None)
    initial_c = torch.empty_like(d)

    def inputs(phase):
        lengths = sm120_k_grouped_lengths(groups, alignment, psum, phase, empty)
        padded = [(length + alignment - 1) // alignment * alignment for length in lengths]
        assert sum(padded) <= total
        av, bv = torch.zeros((total, m), dtype=dtype), torch.zeros((total, n), dtype=dtype)
        sa_cpu, sb_cpu = torch.ones((total // gran, m)), torch.ones((total // gran, n))
        cv = (((torch.arange(groups * m * n).view(groups, m, n) + phase * 3) % 17 - 8).float() / 16).to(out_dtype)
        expected = torch.zeros((groups, m, n)) if c_mode == 'none' else cv.float().clone()
        meta, cursor, sf_cursor, word_prefix = [], 0, 0, [0]
        nt_a, nt_b = [], []
        for g, (length, capacity) in enumerate(zip(lengths, padded)):
            dim = torch.arange(capacity)[:, None]
            ag = (((dim * 3 + torch.arange(m)[None, :] * 5 + g * 7 + phase) % 15 - 7).float() / 4).to(dtype)
            bg = (((dim * 5 + torch.arange(n)[None, :] * 3 + g * 2 + phase * 2) % 13 - 6).float() / 4).to(dtype)
            ag[length:], bg[length:] = 0, 0
            ar, br = ag.float(), bg.float()
            if quant:
                rows = capacity // gran
                sga = torch.pow(2.0, ((torch.arange(rows)[:, None] + torch.arange(m)[None, :] + g + phase) % 3 - 3).float())
                sgb = torch.pow(2.0, ((torch.arange(rows)[:, None] * 2 + torch.arange(n)[None, :] + g * 2 + phase) % 3 - 3).float())
                sa_cpu[sf_cursor:sf_cursor + rows] = sga
                sb_cpu[sf_cursor:sf_cursor + rows] = sgb
                ar *= sga[torch.arange(capacity) // gran]
                br *= sgb[torch.arange(capacity) // gran]
                sf_cursor += rows
                word_prefix.append(word_prefix[-1] + (rows + 3) // 4)
            expected[g] += ar[:length].T @ br[:length]
            av[cursor:cursor + capacity], bv[cursor:cursor + capacity] = ag, bg
            nt_a.append(ag.T.contiguous().flatten())
            nt_b.append(bg.T.contiguous().flatten())
            meta.append(cursor + length if psum else capacity)
            cursor += capacity
        if api == 'fp8_nt':
            av, bv = torch.cat(nt_a).view(m, total), torch.cat(nt_b).view(n, total)
        a.copy_(av)
        b.copy_(bv)
        metadata.copy_(torch.tensor(meta, dtype=torch.int32))
        initial_c.copy_(cv)
        if c_mode == 'different':
            c.copy_(cv)
        if quant:
            sa.copy_(sa_cpu)
            sb.copy_(sb_cpu)
        return expected, cv, lengths, word_prefix

    function = getattr(deep_gemm, f'k_grouped_{"bf16" if not quant else "fp8"}_gemm_{"nt" if api == "fp8_nt" else "tn"}_contiguous')

    def run():
        aa, bb = a, b
        if quant:
            sf_a, sf_b = sa, sb
            if packed and total:
                sf_a = deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
                    sa, metadata, ks_cpu, gran, alignment, psum)
                sf_b = deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
                    sb, metadata, ks_cpu, gran, alignment, psum)
                for sf, mn in ((sf_a, m), (sf_b, n)):
                    assert sf.dtype == torch.int32 and sf.is_contiguous()
                    assert sf.stride() == (mn, 1) and sf.shape[1] == mn
            aa, bb = (a, sf_a), (b, sf_b)
        if c_mode == 'same':
            d.copy_(initial_c)
        kwargs = {} if not quant or default_recipe else dict(recipe=(1, 1, gran))
        function(aa, bb, d, ks_cpu, metadata, c=c, use_psum_layout=psum, **kwargs)

    def check():
        actual = d.cpu().float()
        tolerance = (0.008, 0.02) if out_dtype == torch.bfloat16 else (2e-4, 1e-4)
        torch.testing.assert_close(actual, expected, rtol=tolerance[0], atol=tolerance[1])
        for g, length in enumerate(lengths):
            if length == 0:
                torch.testing.assert_close(actual[g], expected[g], rtol=0, atol=0)
        assert (storage[:16] == 19).all() and (storage[-16:] == 19).all()
        if c_mode == 'different':
            torch.testing.assert_close(c.cpu(), cv, rtol=0, atol=0)
        if quant and total:
            for sf, mn in ((sa, m), (sb, n)):
                packed_sf = deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
                    sf, metadata, ks_cpu, gran, alignment, psum)[:word_prefix[-1]].cpu()
                exponent = ((sf.cpu().view(torch.int32).to(torch.int64) >> 23) & 255)
                cursor, expected_words = 0, []
                for length in lengths:
                    rows = ((length + alignment - 1) // alignment * alignment) // gran
                    for row in range(0, rows, 4):
                        word = torch.zeros(mn, dtype=torch.int64)
                        for byte in range(min(4, rows - row)):
                            word |= exponent[cursor + row + byte] << (8 * byte)
                        expected_words.append(word.to(torch.int32))
                    cursor += rows
                expected_packed = torch.stack(expected_words) if expected_words else torch.empty((0, mn), dtype=torch.int32)
                assert len(expected_packed) == word_prefix[-1]
                torch.testing.assert_close(packed_sf[:len(expected_packed)], expected_packed, rtol=0, atol=0)

    old_alignment, old_pdl = deep_gemm.get_mk_alignment_for_contiguous_layout(), deep_gemm.get_pdl()
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        deep_gemm.set_pdl(pdl)
        expected, cv, lengths, word_prefix = inputs(0)
        for _ in range(3):
            d.fill_(float('nan'))
            run()
            check()
        if graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=stream):
                run()
            for phase in (0, 1, 2):
                expected, cv, lengths, word_prefix = inputs(phase)
                d.fill_(float('nan'))
                captured.replay()
                torch.cuda.synchronize()
                check()
    finally:
        deep_gemm.set_pdl(old_pdl)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)
    print(f' > Native SM120 K-grouped: {api=}, {groups=}, {alignment=}, {psum=}, {out_dtype=}, '
          f'{c_mode=}, {shape=}, {gran=}, {packed=}, {graph=}, {empty=}, {ks_mode=}, {word_prefix=}')


def sm120_dense_quantized(rows, k, fp4, gran, phase, mn_major=False):
    row = torch.arange(rows)[:, None]
    dim = torch.arange(k)[None, :]
    indices = (row * 5 + dim * 3 + dim // 7 + phase).remainder(15)
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6])
    if fp4:
        codes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15], dtype=torch.uint8)[indices]
        if mn_major:
            assert rows % 2 == 0
            raw = (codes[0::2] | (codes[1::2] << 4)).T.contiguous().T.view(torch.int8)
            unpacked = torch.empty((rows, k), dtype=torch.uint8)
            unpacked[0::2] = raw.view(torch.uint8) & 15
            unpacked[1::2] = raw.view(torch.uint8) >> 4
        else:
            assert k % 2 == 0
            raw = (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(torch.int8)
            unpacked = torch.empty((rows, k), dtype=torch.uint8)
            unpacked[:, 0::2] = raw.view(torch.uint8) & 15
            unpacked[:, 1::2] = raw.view(torch.uint8) >> 4
        magnitude = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])[unpacked.long() & 7]
        decoded = torch.where((unpacked & 8) != 0, -magnitude, magnitude)
    else:
        raw = values[indices].to(torch.float8_e4m3fn)
        raw = raw.T.contiguous().T if mn_major else raw
        decoded = raw.float()
    sf = torch.pow(2.0, ((row + torch.arange((k + gran - 1) // gran)[None, :] + phase) % 3 - 4).float())
    decoded *= sf[:, torch.arange(k) // gran]
    return raw, sf, decoded


def sm120_dense_device_matrix(values, padded=False, offset=0, extra=5, tma=True):
    from sm120_test_storage import native_matrix
    return native_matrix(values, padded, offset, extra, tma)


def exercise_sm120_dense_fp8_fp4(fmt, layout, shape, out_dtype, alpha, c_mode, sf_kind,
                                 grans=(32, 128), padded=False, graph=False, pdl=False, common_recipe=False,
                                 mn_grans=(1, 1), compare_default=False, legacy_alias=False):
    assert get_arch_major() == 12
    assert fmt in ((False, False), (False, True), (True, False), (True, True))
    assert layout in ('nt', 'nn', 'tn', 'tt') and c_mode in ('none', 'same', 'different')
    assert sf_kind in ('float', 'packed')
    m, n, k = shape
    assert fmt[0] == fmt[1] or k % 128 == 0
    mn_a = layout[0] == 't' and fmt != (True, True)
    mn_b = layout[1] == 'n' and fmt != (True, True)

    assert sf_kind != 'packed' or mn_grans == (1, 1)
    if compare_default:
        assert grans == (128, 128)
        assert mn_grans == ((1, 128) if sf_kind == 'float' else (1, 1))
    assert not legacy_alias or (fmt == (False, False) and layout == 'nt')

    def operand(rows, fp4, gran, mn_gran, phase, mn_major):
        raw, old_sf, decoded = sm120_dense_quantized(rows, k, fp4, gran, phase, mn_major)
        if mn_gran == 128:
            decoded /= old_sf[:, torch.arange(k) // gran]
            sf = torch.pow(2.0, ((torch.arange((rows + 127) // 128)[:, None]
                                 + torch.arange((k + gran - 1) // gran)[None, :] + phase) % 3 - 4).float())
            decoded *= sf[torch.arange(rows) // 128][:, torch.arange(k) // gran]
            return raw, sf, decoded
        assert mn_gran == 1
        return raw, old_sf, decoded

    def make_inputs(phase):
        av, sa, ar = operand(m, fmt[0], grans[0], mn_grans[0], phase, mn_a)
        bv, sb, br = operand(n, fmt[1], grans[1], mn_grans[1], phase + 2, mn_b)
        cv = (((torch.arange(m)[:, None] * 3 + torch.arange(n)[None, :] + phase) % 17 - 8).float() / 16).to(out_dtype)
        return av, bv, sa, sb, ar, br, cv

    av, bv, sa_cpu, sb_cpu, ar, br, cv = make_inputs(0)
    a = av.cuda() if mn_a else sm120_dense_device_matrix(av)[0]
    b = bv.cuda() if mn_b else sm120_dense_device_matrix(bv)[0]
    assert (a.stride(0) == 1) if mn_a else (a.stride(1) == 1)
    assert (b.stride(0) == 1) if mn_b else (b.stride(1) == 1)
    sa, sb = sa_cpu.cuda(), sb_cpu.cuda()
    d, storage = sm120_dense_device_matrix(torch.zeros_like(cv), padded, int(padded), extra=7)
    c = sm120_dense_device_matrix(cv, padded, int(padded), extra=11, tma=False)[0] if c_mode == 'different' else (d if c_mode == 'same' else None)
    initial_c = cv.cuda()
    valid = torch.zeros_like(storage, dtype=torch.bool)
    valid.as_strided(d.shape, d.stride(), d.storage_offset()).fill_(True)
    if padded and c_mode == 'different':
        assert c.stride(0) != d.stride(0)
    kwargs = (dict(recipe=(*mn_grans, grans[0])) if common_recipe or compare_default else
              dict(recipe_a=(mn_grans[0], grans[0]), recipe_b=(mn_grans[1], grans[1])))
    assert not common_recipe or grans[0] == grans[1]
    function = getattr(deep_gemm, 'fp8_gemm_nt' if legacy_alias else f'fp8_fp4_gemm_{layout}')
    explicit_d = sm120_dense_device_matrix(torch.zeros_like(cv), padded, int(padded), extra=7)[0] if compare_default else None

    def run():
        sf_a, sf_b = sa, sb
        if sf_kind == 'packed':
            sf_a = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sa)
            sf_b = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sb)
            assert sf_a.dtype == sf_b.dtype == torch.int32
            assert sf_a.stride(0) == sf_b.stride(0) == 1
            assert sf_a.stride(1) == align(m, 4) and sf_b.stride(1) == align(n, 4)
        aa = (a, sf_a) if layout[0] == 'n' else (a.T, sf_a.T)
        bb = (b, sf_b) if layout[1] == 't' else (b.T, sf_b.T)
        if compare_default:
            explicit_c = c
            if c_mode == 'same':
                explicit_d.copy_(initial_c)
                explicit_c = explicit_d
            function(aa, bb, explicit_d, c=explicit_c, alpha=alpha,
                     disable_ue8m0_cast=sf_kind == 'packed' and alpha is not None, **kwargs)
        if c_mode == 'same':
            d.copy_(initial_c)
        function(aa, bb, d, c=c, alpha=alpha,
                 disable_ue8m0_cast=sf_kind == 'packed' and alpha is not None,
                 **({} if compare_default else kwargs))

    def check():
        if compare_default:
            assert_bitwise_equal(d, explicit_d, 'default versus explicit dense recipe')
        expected = (ar @ br.T) * (1.0 if alpha is None else alpha)
        if c_mode != 'none':
            expected += cv.float()
        actual = d.cpu().float()
        assert torch.isfinite(actual).all()
        if alpha == 0:
            torch.testing.assert_close(actual, torch.zeros_like(expected) if c_mode == 'none' else cv.float(), rtol=0, atol=0)
        else:
            rtol, atol = (0.008, 0.02) if out_dtype == torch.bfloat16 else (2e-4, 1e-4)
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            assert calc_diff(actual, expected) < (1e-5 if out_dtype == torch.bfloat16 else 1e-8)
        assert (storage[~valid] == 19).all(), 'Output store touched guard cells'
        if c_mode == 'different':
            torch.testing.assert_close(c.cpu(), cv, rtol=0, atol=0)

    original_pdl = deep_gemm.get_pdl()
    try:
        deep_gemm.set_pdl(pdl)
        for _ in range(3):
            d.fill_(float('nan'))
            run()
            check()
        if graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            check()
            torch.cuda.synchronize()
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=stream):
                run()
            for phase in range(3):
                if phase:
                    av, bv, sa_cpu, sb_cpu, ar, br, cv = make_inputs(phase)
                    a.copy_(av)
                    b.copy_(bv)
                    sa.copy_(sa_cpu)
                    sb.copy_(sb_cpu)
                    initial_c.copy_(cv)
                    if c_mode == 'different':
                        c.copy_(cv)
                d.fill_(float('nan'))
                captured.replay()
                torch.cuda.synchronize()
                check()
    finally:
        deep_gemm.set_pdl(original_pdl)
    print(f' > Native SM120 dense quantized: {fmt=}, {layout=}, {shape=}, {out_dtype=}, '
          f'{alpha=}, {c_mode=}, {sf_kind=}, {grans=}, {padded=}, {graph=}, {pdl=}')


def sm120_quant_grouped_intervals(lengths, alignment):
    intervals, end = [], 0
    for length in lengths:
        start = (end + alignment - 1) // alignment * alignment
        end = start + length
        intervals.append((start, end))
    return intervals


def exercise_sm120_quant_grouped(mode, fmt, alignment, sf_kind, grans=(32, 128),
                                  mn_grans=(1, 1), nn=False, zero_padding=False,
                                  all_empty=False, graph=False, pdl=False, defaults=False, k=256,
                                  n=None, padded_a=False):
    assert get_arch_major() == 12 and mode in ('labels', 'psum', 'masked')
    assert alignment in (32, 64, 128) and sf_kind in ('float', 'packed')
    assert not (mode == 'masked' and (nn or zero_padding))
    assert sf_kind != 'packed' or mn_grans == (1, 1)
    assert fmt[0] == fmt[1] or k % 128 == 0
    assert not defaults or (grans == (128, 128) and
                            mn_grans == ((1, 128) if sf_kind == 'float' else (1, 1)))
    groups = 5
    m = alignment + 3 if mode == 'masked' else groups * 2 * alignment + alignment
    n = (80 if nn else 72) if n is None else n
    assert not padded_a or mode != 'masked'
    mn_b = nn and fmt != (True, True)
    expected_m = {32: 32, 64: 33, 128: 65}[alignment]
    assert deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(expected_m) == alignment

    def operand(rows, fp4, gran, mn_gran, phase, mn_major=False):
        raw, old_sf, decoded = sm120_dense_quantized(rows, k, fp4, gran, phase, mn_major)
        decoded /= old_sf[:, torch.arange(k) // gran]
        sf = torch.pow(2.0, ((torch.arange((rows + mn_gran - 1) // mn_gran)[:, None]
                             + torch.arange((k + gran - 1) // gran)[None, :] + phase) % 3 - 4).float())
        decoded *= sf[torch.arange(rows) // mn_gran][:, torch.arange(k) // gran]
        return raw, sf, decoded

    def cpu_inputs(phase):
        aa = [operand(m, fmt[0], grans[0], mn_grans[0], phase + g * 2)
              for g in range(groups if mode == 'masked' else 1)]
        bb = [operand(n, fmt[1], grans[1], mn_grans[1], phase + g * 2 + 1, mn_b)
              for g in range(groups)]
        av, sa, ar = [torch.stack([item[i] for item in aa]) for i in range(3)]
        if mode != 'masked':
            av, sa, ar = av[0], sa[0], ar[0]
        bv, sb, br = [torch.stack([item[i] for item in bb]) for i in range(3)]
        lengths = [0 if all_empty or phase == 1 or g % 3 == 0 or g == groups - 1 else
                   (alignment + 1 if g % 3 == 1 else alignment - 3) for g in range(groups)]
        if phase == 2 and not all_empty:
            lengths = [alignment - 1 if g % 2 == 0 else 0 for g in range(groups)]
        shape = (groups, m, n) if mode == 'masked' else (m, n)
        valid = torch.zeros(shape[:-1], dtype=torch.bool)
        expected = torch.full(shape, 0.0 if zero_padding else 7.0)
        if mode == 'masked':
            meta = torch.tensor(lengths, dtype=torch.int32)
            for g, length in enumerate(lengths):
                valid[g, :length] = True
                expected[g, :length] = ar[g, :length] @ br[g].T
        else:
            intervals = sm120_quant_grouped_intervals(lengths, alignment)
            labels = torch.full((m,), -1, dtype=torch.int32)
            for g, (start, end) in enumerate(intervals):
                assert start % alignment == 0 and end <= m
                labels[start:end] = g
                valid[start:end] = True
                expected[start:end] = ar[start:end] @ br[g].T
            meta = labels if mode == 'labels' else torch.tensor([end for _, end in intervals], dtype=torch.int32)
        if mode != 'labels':
            for block in range(sa.shape[-2]):
                active = valid[..., block * mn_grans[0]:(block + 1) * mn_grans[0]].any(dim=-1)
                sa[..., block, :] = torch.where(active[..., None], sa[..., block, :],
                                                   float('nan') if mode == 'psum' else 1.0)
        return av, bv, sa, sb, meta, expected, valid

    av, bv, sa_cpu, sb_cpu, meta, expected, valid = cpu_inputs(0)
    if padded_a:
        a, storage_a = sm120_dense_device_matrix(av, True, 1, 16)
        guard_a = torch.zeros_like(storage_a, dtype=torch.bool)
        guard_a.as_strided(a.shape, a.stride(), a.storage_offset()).fill_(True)
    else:
        a = av.cuda()
    b = (bv.transpose(1, 2).contiguous().transpose(1, 2) if mn_b else bv).cuda()
    assert a.stride(-1) == 1 and (b.stride(-2) == 1 if mn_b else b.stride(-1) == 1)
    sa, sb, metadata = sa_cpu.cuda(), sb_cpu.cuda(), meta.cuda()
    if mode == 'masked':
        storage = torch.full((8 + groups * m * n + 16,), 19, dtype=torch.bfloat16, device='cuda')
        d = storage.as_strided((groups, m, n), (m * n, n, 1), 8)
    else:
        d, storage = sm120_dense_device_matrix(torch.zeros((m, n), dtype=torch.bfloat16), True, 1, 7)
    guard = torch.zeros_like(storage, dtype=torch.bool)
    guard.as_strided(d.shape, d.stride(), d.storage_offset()).fill_(True)
    explicit_d = torch.empty_like(d) if defaults else None
    recipes = dict(recipe_a=(mn_grans[0], grans[0]), recipe_b=(mn_grans[1], grans[1]))

    def packed(sf, rows, gran, psum=None):
        result = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf, psum)
        words = (k + 4 * gran - 1) // (4 * gran)
        assert result.dtype == torch.int32 and result.shape[-2:] == (rows, words)
        assert result.stride(-2) == 1 and result.stride(-1) == align(rows, 4)
        assert result.data_ptr() % 16 == 0
        if result.ndim == 3:
            assert result.stride(0) == words * align(rows, 4)
        return result

    def run():
        sf_a, sf_b = sa, sb
        if sf_kind == 'packed':
            if mode == 'masked':
                active = torch.arange(m, device='cuda')[None, :] < metadata[:, None]
                sf_a = sa.masked_fill(~active[..., None], 1.0)
            sf_a = packed(sf_a, m, grans[0], metadata if mode == 'psum' else None)
            sf_b = packed(sb, n, grans[1])
        aa = a, sf_a
        bb = (b.transpose(1, 2), sf_b.transpose(1, 2)) if nn else (b, sf_b)
        if mode == 'masked':
            function = deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked
            positional, kwargs = (metadata, expected_m), {}
        else:
            function = getattr(deep_gemm, f'm_grouped_fp8_fp4_gemm_{"nn" if nn else "nt"}_contiguous')
            positional = (metadata,)
            kwargs = dict(use_psum_layout=mode == 'psum', ensure_zero_padding=zero_padding)
            if mode == 'psum' and not nn:
                kwargs['expected_m_for_psum_layout'] = 1
        if defaults:
            explicit_d.fill_(7)
            function(aa, bb, explicit_d, *positional, **kwargs, **recipes)
        function(aa, bb, d, *positional, **kwargs, **({} if defaults else recipes))

    def check():
        actual = d.cpu().float()
        assert torch.isfinite(actual[valid]).all()
        torch.testing.assert_close(actual[valid], expected[valid], rtol=0.008, atol=0.02)
        comparison_rows = valid.clone()
        if zero_padding:
            zero_rows = ~valid
            if mode == 'psum':
                zero_rows = torch.zeros_like(valid)
                ends = metadata.cpu().tolist()
                for end in ends:
                    zero_rows[end:align(end, alignment)] = True
                assert (actual[align(ends[-1], alignment):] == 7).all(), 'PSUM cleared unused capacity'
            torch.testing.assert_close(actual[zero_rows], torch.zeros_like(actual[zero_rows]), rtol=0, atol=0)
            comparison_rows |= zero_rows
        if defaults:
            assert_bitwise_equal(d[comparison_rows], explicit_d[comparison_rows],
                                 'grouped default versus explicit scaling')
        assert (storage[~guard] == 19).all(), 'Grouped quantized output modified guard cells'
        if padded_a:
            sentinel = torch.tensor(19, dtype=a.dtype).float().item()
            assert (storage_a.float()[~guard_a] == sentinel).all(), 'Grouped quantized input modified guard cells'

    old_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    old_pdl = deep_gemm.get_pdl()
    try:
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
        deep_gemm.set_pdl(pdl)
        for _ in range(3):
            d.fill_(7)
            run()
            check()
        if graph:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=stream):
                run()
            for phase in (0, 1, 2):
                av, bv, sa_cpu, sb_cpu, meta, expected, valid = cpu_inputs(phase)
                a.copy_(av)
                b.copy_(bv)
                sa.copy_(sa_cpu)
                sb.copy_(sb_cpu)
                metadata.copy_(meta)
                d.fill_(7)
                captured.replay()
                torch.cuda.synchronize()
                check()
    finally:
        deep_gemm.set_pdl(old_pdl)
        deep_gemm.set_mk_alignment_for_contiguous_layout(old_alignment)
    print(f' > Native SM120 grouped quantized: {mode=}, {fmt=}, {alignment=}, {sf_kind=}, '
          f'{grans=}, {mn_grans=}, {nn=}, {zero_padding=}, {all_empty=}, {graph=}, {pdl=}, '
          f'{defaults=}, {k=}, {n=}, {padded_a=}')
