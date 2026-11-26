import tilelang
import tilelang.language as T
import torch

import deep_gemm
from deep_gemm.testing import (
    assert_check,
    bench_kineto,
    calc_diff,
    count_bytes,
    get_arch_major,
    test_filter,
)
from deep_gemm.utils import align, ceil_div, per_token_cast_to_fp8

HIDDEN, Q_RANK, KV_DIM, NUM_HEADS, ROPE_DIM = 5120, 1280, 512, 64, 64
INDEX_NUM_HEADS, INDEX_HEAD_DIM = 32, 128
CACHE_BLOCK_SIZE, SF_GRAN = 136, 32
NUM_GROUPS, INPUT_RANK, OUTPUT_RANK = 8, 4096, 1024


@T.macro
def _quant_exponent(scale: T.float32) -> T.int32:
    bits = T.reinterpret('uint32', scale)
    return -T.reinterpret('int32', ((bits - 1) >> 23) + 1 - 127)


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True})
def _q_rmsnorm(eps: float):
    m = T.dynamic('m')

    @T.prim_func
    def _norm(
        x: T.Tensor((m, 1792), T.bfloat16),
        w: T.Tensor((1280,), T.bfloat16),
        y: T.Tensor((m, 1280), T.float8_e4m3fn),
        sf: T.StridedTensor((m, 10, 4), (4, T.ceildiv(m, 4) * 16, 1), T.uint8),
    ):
        with T.Kernel(m, threads=32) as row:
            values = T.alloc_fragment((1, 1280), T.float32)
            squares = T.alloc_fragment((1, 1280), T.float32)
            weights = T.alloc_fragment((1280,), T.bfloat16)
            rstd = T.alloc_fragment((1,), T.float32)
            rounded = T.alloc_fragment((1, 1280), T.float32)
            quantized = T.alloc_fragment((1, 1280), T.float8_e4m3fn)
            amax = T.alloc_fragment((1, 40), T.float32)
            multiplier = T.alloc_fragment((1, 40), T.float32)
            exponent = T.alloc_fragment((1, 40), T.uint8)
            layout = T.Fragment((1, 1280), forward_fn=lambda i, j: (j // 8 % 32, j // 256 * 8 + j % 8))
            T.annotate_layout({values: layout, rounded: layout, quantized: layout})
            T.copy(w, weights)
            T.pdl_sync()
            for i, j in T.Parallel(1, 1280):
                values[i, j] = x[row, j]
                squares[i, j] = values[i, j] * values[i, j]
            T.reduce_sum(squares, rstd, dim=1)
            rstd[0] = T.rsqrt(rstd[0] / 1280 + eps)
            for i, j in T.Parallel(1, 1280):
                rounded[i, j] = T.cast(T.cast(values[i, j] * rstd[i] * (weights[j] * 2.0), T.bfloat16), T.float32)
            T.reduce_absmax(T.reshape(rounded, (1, 40, 32)), amax, dim=2)
            for i, j in T.Parallel(1, 40):
                e = _quant_exponent(T.max(amax[i, j], 1e-4) / 448.0)
                multiplier[i, j] = T.reinterpret('float32', (e + 127) << 23)
                exponent[i, j] = T.cast(127 - e, T.uint8)
            for i, j in T.Parallel(1, 1280):
                quantized[i, j] = T.cast(rounded[i, j] * multiplier[i, j // 32], T.float8_e4m3fn)
                y[row, j] = quantized[i, j]
            T.pdl_trigger()
            for i, j in T.Parallel(1, 40):
                sf[row, j // 4, j % 4] = exponent[i, j]

    return _norm


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True})
def _kv_norm_rope_store(block_size: int, eps: float):
    m, length, rope_stride, blocks, cache_stride = T.dynamic('m length rope_stride blocks cache_stride')

    @T.prim_func
    def _fused_norm_rope_hadamard_cast_store_kvcache(
        x: T.Tensor((m, 1792), T.bfloat16),
        w: T.Tensor((512,), T.bfloat16),
        positions: T.Tensor((m,), T.int32),
        cos: T.StridedTensor((length, 32), (rope_stride, 1), T.float32),
        sin: T.StridedTensor((length, 32), (rope_stride, 1), T.float32),
        slots: T.Tensor((m,), T.int32),
        cache: T.StridedTensor((blocks, block_size * 528), (cache_stride, 1), T.uint8),
    ):
        with T.Kernel(T.ceildiv(m, 4), threads=128) as bid:
            tx = T.get_thread_binding()
            row, col = bid * 4 + tx // 32, tx % 32 * 16
            values = T.alloc_local((16,), T.bfloat16)
            weights = T.alloc_local((16,), T.bfloat16)
            fp32 = T.alloc_local((16,), T.float32)
            squares = T.alloc_local((16,), T.float32)
            partial = T.alloc_local((2,), T.float32)
            pair_max = T.alloc_local((2,), T.bfloat16)
            quantized = T.alloc_local((16,), T.float8_e4m3fn)
            rope = T.alloc_shared((4, 64), T.bfloat16)
            T.pdl_sync()
            for k in T.vectorized(16):
                values[k] = T.if_then_else(row < m, x[row, 1280 + col + k], 0)
                weights[k] = w[col + k]
                fp32[k] = T.cast(values[k], T.float32)
            for k in T.unroll(16):
                squares[k] = fp32[k] * fp32[k]
            T.clear(partial)
            for k in T.unroll(8):
                partial[0] += squares[2 * k]
                partial[1] += squares[2 * k + 1]
            total = T.alloc_var(T.float32, partial[0] + partial[1])
            for r in T.unroll(5):
                total = total + T.shfl_xor(total, 1 << (4 - r))
            rstd = T.rsqrt(total / 512 + eps)
            for k in T.vectorized(16):
                values[k] = T.cast(fp32[k] * rstd * T.cast(weights[k], T.float32), T.bfloat16)
            if col >= 448:
                for part in T.unroll(2):
                    for k in T.vectorized(8):
                        rope[tx // 32, col - 448 + part * 8 + k] = values[part * 8 + k]
            T.sync_threads()
            for i, j in T.Parallel(4, 32):
                if bid * 4 + i < m:
                    position = positions[bid * 4 + i]
                    x0, x1 = T.cast(rope[i, 2 * j], T.float32), T.cast(rope[i, 2 * j + 1], T.float32)
                    c, s = cos[position, j], sin[position, j]
                    rope[i, 2 * j] = T.cast(T.ieee_fmaf(x0, c, -(x1 * s)), T.bfloat16)
                    rope[i, 2 * j + 1] = T.cast(T.ieee_fmaf(x0, s, x1 * c), T.bfloat16)
            T.sync_threads()
            if col >= 448:
                for part in T.unroll(2):
                    for k in T.vectorized(8):
                        values[part * 8 + k] = rope[tx // 32, col - 448 + part * 8 + k]
            T.clear(pair_max)
            for k in T.vectorized(16):
                pair_max[k % 2] = T.max(pair_max[k % 2], T.abs(values[k]))
            amax = T.cast(T.max(pair_max[0], pair_max[1]), T.float32)
            group_amax = T.max(amax, T.shfl_xor(amax, 1))
            e = _quant_exponent(T.max(group_amax, 1e-4) * (1.0 / 448.0))
            multiplier = T.reinterpret(T.cast((e + 127) << 7, T.uint16), T.bfloat16)
            for k in T.vectorized(16):
                quantized[k] = T.cast(values[k] * multiplier, T.float8_e4m3fn)
            T.pdl_trigger()
            slot = T.alloc_var(T.int32, -1)
            if tx % 32 == 0:
                slot = T.if_then_else(row < m, slots[row], -1)
                slot = T.Select(slot < blocks * block_size, slot, -1)
            slot = T.shfl_sync(slot, 0, 32)
            if slot >= 0:
                block, offset = slot // block_size, slot % block_size
                for k in T.vectorized(16):
                    cache[block, offset * 528 + col + k] = T.reinterpret(quantized[k], T.uint8)
                if tx % 2 == 0:
                    cache[block, offset * 528 + 512 + col // 32] = T.cast(127 - e, T.uint8)

    return _fused_norm_rope_hadamard_cast_store_kvcache


def make_fp8_pair(m: int, k: int, scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn((m, k), dtype=torch.bfloat16, device='cuda') * scale
    x_fp8, x_sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=SF_GRAN, use_packed_ue8m0=True)
    sf = torch.empty_strided((m, k // 128), (1, align(m, 4)), dtype=torch.int32, device='cuda')
    sf.copy_(x_sf)
    return x_fp8, sf


@test_filter(lambda: get_arch_major() == 10)
@torch.no_grad()
def test_v41f_qkv_lora() -> None:
    deep_gemm.set_pdl(True)
    eps = 1e-6
    q_norm_kernel = _q_rmsnorm(eps)
    kv_norm_kernel = _kv_norm_rope_store(CACHE_BLOCK_SIZE, eps)
    q_dim = NUM_HEADS * KV_DIM
    index_q_dim = INDEX_NUM_HEADS * INDEX_HEAD_DIM

    for q_out_dim in (q_dim, q_dim + index_q_dim):
        print(f'Testing Experimental V41F QKV LoRA (q_out_dim={q_out_dim}):')
        for m in (1, 7, 16, 64, 127, 128, 129, 192, 240, 256, 288, 304, 320, 384, 420, 512, 1024, 1025, 1536, 2048, 4096):
            x = make_fp8_pair(m, HIDDEN)
            qkv_weight = make_fp8_pair(Q_RANK + KV_DIM, HIDDEN, scale=HIDDEN ** -0.5)
            q_b_weight = make_fp8_pair(q_out_dim, Q_RANK, scale=Q_RANK ** -0.5)
            q_norm_weight = torch.randn((Q_RANK,), dtype=torch.bfloat16, device='cuda')
            kv_norm_weight = torch.randn((KV_DIM,), dtype=torch.bfloat16, device='cuda')
            positions = torch.randint(0, 512, (m,), dtype=torch.int32, device='cuda')
            angles = torch.randn((512, ROPE_DIM // 2), dtype=torch.float32, device='cuda')
            cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)

            num_blocks = ceil_div(m, CACHE_BLOCK_SIZE) + 1
            token_bytes = KV_DIM + KV_DIM // SF_GRAN
            cache_stride = align(CACHE_BLOCK_SIZE * token_bytes, 512)
            slots = torch.arange(m, dtype=torch.int32, device='cuda')
            slots[::17] = -1
            slots[::31] = num_blocks * CACHE_BLOCK_SIZE

            cache_storage = torch.full((num_blocks, cache_stride), 0x35, dtype=torch.uint8, device='cuda')
            ref_cache_storage = cache_storage.clone()
            cache = cache_storage.as_strided(
                (num_blocks, CACHE_BLOCK_SIZE, 1, token_bytes),
                (cache_stride, token_bytes, token_bytes, 1))
            q = torch.empty((m, q_out_dim), dtype=torch.bfloat16, device='cuda')
            ref_q = torch.empty_like(q)

            qkv_a = torch.empty((m, Q_RANK + KV_DIM), dtype=torch.bfloat16, device='cuda')
            q_a = torch.empty((m, Q_RANK), dtype=torch.float8_e4m3fn, device='cuda')
            q_a_sf = torch.empty_strided((m, Q_RANK // 128), (1, align(m, 4)), dtype=torch.int32, device='cuda')
            q_a_sf_bytes = q_a_sf.mT.view(torch.uint8).view(Q_RANK // 128, m, 4).permute(1, 0, 2)

            def run() -> None:
                deep_gemm.experimental.v41f_qkv_lora(
                    x, qkv_weight, q_b_weight, q_norm_weight, kv_norm_weight,
                    positions, cos_sin, slots, q, cache, eps)

            def run_baseline() -> None:
                deep_gemm.fp8_gemm_nt(x, qkv_weight, qkv_a, recipe=(1, 1, SF_GRAN))
                kv_norm_kernel(qkv_a, kv_norm_weight, positions, cos_sin[:, :ROPE_DIM // 2],
                               cos_sin[:, ROPE_DIM // 2:], slots, ref_cache_storage[:, :CACHE_BLOCK_SIZE * token_bytes])
                q_norm_kernel(qkv_a, q_norm_weight, q_a, q_a_sf_bytes)
                deep_gemm.fp8_gemm_nt((q_a, q_a_sf), q_b_weight, ref_q, recipe=(1, 1, SF_GRAN))

            run()
            run_baseline()

            valid_slots = slots[(slots >= 0) & (slots < num_blocks * CACHE_BLOCK_SIZE)].long()
            block_idx, offset_idx = valid_slots // CACHE_BLOCK_SIZE, valid_slots % CACHE_BLOCK_SIZE
            payload_idx = block_idx[:, None] * cache_stride + offset_idx[:, None] * token_bytes + torch.arange(KV_DIM, device='cuda')

            q_diff = calc_diff(q, ref_q)
            if q_out_dim == q_dim + index_q_dim:
                q_main, index_q = q.split([q_dim, index_q_dim], dim=-1)
                ref_q_main, ref_index_q = ref_q.split([q_dim, index_q_dim], dim=-1)
                assert_check(calc_diff(q_main, ref_q_main) < 1e-3 and calc_diff(index_q, ref_index_q) < 1e-3,
                             f'{m=}: split Q/indexer Q mismatch')
            actual_kv = cache_storage.flatten()[payload_idx].view(torch.float8_e4m3fn).float()
            ref_kv = ref_cache_storage.flatten()[payload_idx].view(torch.float8_e4m3fn).float()
            kv_diff = calc_diff(actual_kv, ref_kv)
            assert_check(q_diff < 1e-3 and kv_diff < 2e-4, f'{q_out_dim=}, {m=}, {q_diff=}, {kv_diff=}')

            expected_cache = ref_cache_storage.flatten().clone()
            expected_cache[payload_idx] = cache_storage.flatten()[payload_idx]
            assert_check(torch.equal(cache_storage.flatten(), expected_cache), f'{q_out_dim=}, {m=}: cache/scales mismatch')

            t = bench_kineto(run, 'v41f_qkv_lora_kernel', suppress_kineto_output=True)
            gemm_t, kv_t, q_t = bench_kineto(
                run_baseline, ('fp8_fp4_gemm_1d1d', '_fused_norm_rope_hadamard_cast_store_kvcache_kernel', '_norm_kernel'),
                suppress_kineto_output=True, with_multiple_kernels=True)
            baseline_t = 2 * gemm_t + kv_t + q_t

            num_flops = 2 * m * ((Q_RANK + KV_DIM) * HIDDEN + q_out_dim * Q_RANK)
            num_bytes = (count_bytes(x, qkv_weight, q_b_weight, q_norm_weight, kv_norm_weight, positions, slots, q) +
                         positions.unique().numel() * ROPE_DIM * cos_sin.element_size() +
                         payload_idx.size(0) * token_bytes)
            print(f' > Perf (m={m:5}): {t * 1e6:5.1f} us | '
                  f'{num_flops / t / 1e12:4.0f} TFLOPS | '
                  f'{num_bytes / 1e9 / t:4.0f} GB/s | '
                  f'{baseline_t / t:4.2f}x | '
                  f'{q_diff=:.2e}, {kv_diff=:.2e}')
        print()


@test_filter(lambda: get_arch_major() == 10)
@torch.no_grad()
def test_v41f_wv_o_proj() -> None:
    print('Testing Experimental V41F Wv + O projection:')
    deep_gemm.set_pdl(True)

    for m in (1, 7, 16, 64, 127, 128, 129, 192, 240, 256, 288, 304, 320, 384, 420, 512, 1024, 1025, 1536, 2048, 4096):
        x_fp8, x_sf = make_fp8_pair(m, NUM_GROUPS * INPUT_RANK)
        x = (x_fp8.view(m, NUM_GROUPS, INPUT_RANK), x_sf.view(m, NUM_GROUPS, INPUT_RANK // 128))

        wv_fp8, wv_sf = make_fp8_pair(NUM_GROUPS * OUTPUT_RANK, INPUT_RANK, scale=INPUT_RANK ** -0.5)
        wv_weight = (
            wv_fp8.view(NUM_GROUPS, OUTPUT_RANK, INPUT_RANK),
            wv_sf.view(NUM_GROUPS, OUTPUT_RANK, INPUT_RANK // 128).mT.contiguous().mT,
        )
        o_weight = make_fp8_pair(HIDDEN, NUM_GROUPS * OUTPUT_RANK, scale=(NUM_GROUPS * OUTPUT_RANK) ** -0.5)
        y = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device='cuda')
        ref_y = torch.empty_like(y)
        mid = torch.empty((m, NUM_GROUPS, OUTPUT_RANK), dtype=torch.float8_e4m3fn, device='cuda')
        mid_sf = torch.empty_strided(
            (m, NUM_GROUPS * OUTPUT_RANK // 128), (1, align(m, 4)), dtype=torch.int32, device='cuda')

        def run() -> None:
            deep_gemm.experimental.v41f_wv_o_proj(x, wv_weight, o_weight, y)

        def run_baseline() -> None:
            deep_gemm.fp8_einsum('bhr,hdr->bhd', x, wv_weight, (mid, mid_sf), recipe=(1, 1, SF_GRAN))
            deep_gemm.fp8_gemm_nt((mid.view(m, -1), mid_sf), o_weight, ref_y, recipe=(1, 1, SF_GRAN))

        run()
        run_baseline()

        diff = calc_diff(y, ref_y)
        assert_check(diff < 1e-3, f'{m=}, {diff=}')

        t = bench_kineto(run, 'v41f_wv_o_proj_kernel', suppress_kineto_output=True)
        baseline_t = 2 * bench_kineto(
            run_baseline, 'fp8_fp4_gemm_1d1d', suppress_kineto_output=True, with_multiple_kernels=True)

        num_flops = 2 * m * NUM_GROUPS * OUTPUT_RANK * (INPUT_RANK + HIDDEN)
        num_bytes = count_bytes(x, wv_weight, o_weight, y)
        print(f' > Perf (m={m:5}): {t * 1e6:5.1f} us | '
              f'{num_flops / t / 1e12:4.0f} TFLOPS | '
              f'{num_bytes / 1e9 / t:4.0f} GB/s | '
              f'{baseline_t / t:4.2f}x | '
              f'{diff=:.2e}')
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    test_v41f_qkv_lora()
    test_v41f_wv_o_proj()
