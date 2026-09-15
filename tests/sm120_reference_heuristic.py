import math


SOURCE_AUTHORITY = '139f504:csrc/jit_kernels/heuristics/sm120.hpp'
SOURCE_BLOB = 'bb7e02fa8e9e13864d856e611fa1ce360a466ccb'


def ceil_div(a, b):
    return (a + b - 1) // b


def predict_dense_fp8(m, n, k, num_sms, gran=128, output_bytes=2,
                      alignment=128, swapped=False):
    assert num_sms > 0 and num_sms % 2 == 0 and min(m, n, k) > 0
    small_n = n <= 32
    if alignment <= 64:
        ms = [64]
    else:
        ms = ([64] if small_n else []) + [128]
        if m <= 64:
            ms.append(64)
        if m <= 128 and ceil_div(m, 128) * ceil_div(n, 64) < num_sms // 8:
            ms.append(64)
    ks = ([64] if m >= 2048 else []) + [128]
    ns = ([16] + ([32] if n > 16 else [])) if small_n else [64, 128]
    candidates = []
    for bm in ms:
        for bk in ks:
            for bn in ns:
                swizzle = 128 if not swapped and bn * output_bytes >= 128 else 0
                per_stage = (bm + bn) * bk + ceil_div(bm * 4, 128) * 128 + ceil_div(bn * 4, 128) * 128
                def stages(store_m):
                    d_bytes = bn * output_bytes * store_m if swizzle and bn * output_bytes % swizzle == 0 else 0
                    return min((101376 - 256 - d_bytes) // per_stage, 16)
                store_m = 64 if swizzle and bm > 64 and stages(64) > stages(bm) else bm
                pipeline = stages(store_m)
                if pipeline < 2:
                    continue
                mn_blocks = ceil_div(m, bm) * ceil_div(n, bn)
                kb = ceil_div(k, bk)
                sf_blocks = 4 * gran // bk
                split = 1
                if mn_blocks < num_sms // 2 and sf_blocks:
                    split = ceil_div(num_sms * 3 // 4, mn_blocks)
                    while split > 1 and (kb % split or (kb // split) % sf_blocks):
                        split -= 1
                    split = min(split, kb // (2 * sf_blocks))
                    split = min(split, max(32 * 1024 * 1024 // (m * n * 4), 1))
                    split = max(split, 1)
                waves = ceil_div(mn_blocks * split, num_sms)
                tma_bytes = (bm + bn) * bk + ceil_div(bm * 4, 128) * 128 + ceil_div(bn * 4, 128) * 128
                cycles = int(waves * ((kb // split if split > 1 else kb) *
                                     (tma_bytes * 0.07 + 120 / math.sqrt(pipeline)) + 2000) +
                             (5000 + 0.01 * m * n if split > 1 else 0))
                candidates.append(dict(block_m=bm, block_n=bn, block_k=bk,
                                       store_m=store_m, swizzle_cd=swizzle, stages=pipeline,
                                       split_k=split, cycles=cycles))
    assert candidates
    best = candidates[0]
    for candidate in candidates[1:]:
        ratio = candidate['cycles'] / best['cycles'] if best['cycles'] else 1
        if ratio < 0.95:
            better = True
        elif ratio > 1.05:
            better = False
        else:
            better = (candidate['block_n'], -candidate['block_k'], candidate['block_m'], -candidate['cycles']) > (
                best['block_n'], -best['block_k'], best['block_m'], -best['cycles'])
        if better:
            best = candidate
    return best


def test_old_split_k_prediction():
    for sms in (80, 96, 128):
        config = predict_dense_fp8(32, 256, 16384, sms)
        assert config['split_k'] > 1
        assert (16384 // config['block_k']) % config['split_k'] == 0
        assert (16384 // config['split_k']) % 512 == 0
        assert predict_dense_fp8(4096, 4096, 4096, sms)['split_k'] == 1


def test_old_subtile_prediction():
    config = predict_dense_fp8(2048, 512, 512, 96)
    assert config['store_m'] == 64 and config['block_m'] == 128
    assert config['swizzle_cd'] == 128
