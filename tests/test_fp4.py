"""
FP4 (E2M1) GEMM correctness test for SM100 MXF4 block-scaled kernel.

Usage:
    python tests/test_fp4.py
"""

import torch
import random
import deep_gemm
from deep_gemm.testing import bench_kineto, count_bytes
from generators import KernelType, get_ue8m0_usage

# ============================================================
# E2M1 FP4 查找表
# ============================================================
E2M1_LUT = torch.tensor([
     0.0,   0.5,   1.0,   1.5,   2.0,   3.0,   4.0,   6.0,   # bits 0-7  (S=0)
    -0.0,  -0.5,  -1.0,  -1.5,  -2.0,  -3.0,  -4.0,  -6.0    # bits 8-15 (S=1)
], dtype=torch.float32)

# ============================================================
# 工具函数
# ============================================================

def pack_fp4_random(m: int, k_fp4: int, device='cuda'):
    """生成随机 E2M1 FP4 数据并打包为 int32。每个 int32 包含 8 个 FP4 值。"""
    assert k_fp4 % 8 == 0
    raw = torch.randint(0, 16, (m, k_fp4), dtype=torch.uint8, device=device)
    packed = torch.zeros(m, k_fp4 // 8, dtype=torch.int32, device=device)
    for i in range(8):
        packed += (raw[:, i::8].to(torch.int32) << (i * 4))
    return packed


def pack_fp4_constant(m: int, k_fp4: int, fp4_bits: int = 0x2, device='cuda'):
    """生成常量 FP4 打包数据。fp4_bits=0x2 -> E2M1 1.0"""
    assert k_fp4 % 8 == 0
    word = 0
    for i in range(8):
        word |= (fp4_bits & 0xF) << (i * 4)
    return torch.full((m, k_fp4 // 8), word, dtype=torch.int32, device=device)


def generate_mxf4_scale_factors(m, n, k_fp4, device='cuda', random_sf=False):
    """为 MXF4 (VS=32) 生成 scale factor (float32 格式)。

    Host 端 C++ API (transform_sf_into_required_layout) 会在调用 kernel 之前
    将 float32 SF 转换为 packed UE8M0 int32:
      1. 从 float32 IEEE 754 中提取 8-bit 指数字段 (bitwise_right_shift(23))
      2. 每 4 个 UE8M0 打包为 1 个 int32
      3. 转置为 MN-major + TMA 对齐
    然后 kernel 通过 TMA 加载到 SMEM, 经 UTCCP 写入 TMEM 供 MMA 使用。

    UE8M0: value = 2^(exp - 127)。每 VS=32 个 FP4 元素共享一个 SF。

    Args:
        random_sf: 如果 True, 生成随机的 2 的幂次 SF (0.25, 0.5, 1.0, 2.0, 4.0)
    """
    VS = 32
    sf_k = ((k_fp4 // VS + 3) // 4) * 4
    if random_sf:
        powers = torch.randint(-2, 3, (m, sf_k), device=device).float()
        sf_a = torch.pow(2.0, powers)
        powers = torch.randint(-2, 3, (n, sf_k), device=device).float()
        sf_b = torch.pow(2.0, powers)
    else:
        sf_a = torch.ones((m, sf_k), dtype=torch.float32, device=device)
        sf_b = torch.ones((n, sf_k), dtype=torch.float32, device=device)
    return sf_a, sf_b


def fp4_reference(a_packed, b_packed, m, n, sf_a=None, sf_b=None):
    """CPU 端 E2M1 FP4 GEMM reference: C = A @ B^T, 支持 block-scaled SF。

    Block-scaled MXF4: C[m,n] = sum_g SF_A[m,g] * SF_B[n,g] * dot(A_g, B_g)
    其中 g 是 VS=32 元素的组。sf_a/sf_b 为 float32 (pre-transform), 每组一个值。
    """
    VS = 32
    a_cpu = a_packed.cpu().to(torch.int64) & 0xFFFFFFFF
    b_cpu = b_packed.cpu().to(torch.int64) & 0xFFFFFFFF
    bits_a = torch.stack([(a_cpu >> (i*4)) & 0xF for i in range(8)], dim=-1).reshape(m, -1)
    bits_b = torch.stack([(b_cpu >> (i*4)) & 0xF for i in range(8)], dim=-1).reshape(n, -1)
    a_float = E2M1_LUT[bits_a.long()]  # [m, k_fp4]
    b_float = E2M1_LUT[bits_b.long()]  # [n, k_fp4]

    if sf_a is None:
        return torch.matmul(a_float, b_float.T)

    k_fp4 = a_float.shape[1]
    sf_a_cpu = sf_a.cpu().float()
    sf_b_cpu = sf_b.cpu().float()
    c = torch.zeros(m, n, dtype=torch.float32)
    num_groups = k_fp4 // VS
    for g in range(num_groups):
        k_start, k_end = g * VS, (g + 1) * VS
        a_g = a_float[:, k_start:k_end]
        b_g = b_float[:, k_start:k_end]
        sf_col = g
        if sf_col < sf_a_cpu.shape[1]:
            sfa_g = sf_a_cpu[:, sf_col].unsqueeze(1)
            sfb_g = sf_b_cpu[:, sf_col].unsqueeze(1)
        else:
            sfa_g = torch.ones(m, 1)
            sfb_g = torch.ones(n, 1)
        c += (sfa_g * sfb_g.T) * torch.matmul(a_g, b_g.T)
    return c


def run_kernel(a_packed, b_packed, sf_a, sf_b, m, n, recipe=(1, 1, 128)):
    """调用 FP4 GEMM kernel (复用 fp8_gemm_nt 入口, int32 dtype 触发 FP4 路径)"""
    duc = not get_ue8m0_usage(KernelType.Kernel1D1D)
    d = torch.empty((m, n), device='cuda', dtype=torch.float32)
    deep_gemm.fp8_gemm_nt((a_packed, sf_a), (b_packed, sf_b), d, c=None,
                          recipe=recipe, disable_ue8m0_cast=duc)
    torch.cuda.synchronize()
    return d


def pack_fp4_random_3d(num_groups: int, n: int, k_fp4: int, device='cuda'):
    """生成 [G, N, K_int32] FP4 packed tensor。"""
    assert k_fp4 % 8 == 0
    raw = torch.randint(0, 16, (num_groups, n, k_fp4), dtype=torch.uint8, device=device)
    packed = torch.zeros(num_groups, n, k_fp4 // 8, dtype=torch.int32, device=device)
    for i in range(8):
        packed += (raw[:, :, i::8].to(torch.int32) << (i * 4))
    return packed


def generate_mxf4_sf_3d(num_groups: int, n: int, k_fp4: int, device='cuda', random_sf=False):
    """生成 SFB [G, N, sf_k] for grouped contiguous."""
    VS = 32
    sf_k = ((k_fp4 // VS + 3) // 4) * 4
    if random_sf:
        powers = torch.randint(-2, 3, (num_groups, n, sf_k), device=device).float()
        return torch.pow(2.0, powers)
    return torch.ones((num_groups, n, sf_k), dtype=torch.float32, device=device)


def fp4_reference_grouped(a_packed, b_packed_grouped, m_indices,
                          n: int, num_groups: int,
                          sf_a=None, sf_b_grouped=None):
    """Per-row grouped FP4 reference: D[i] = A[i] @ B[m_indices[i]].T (with SF scaling).

    Padding rows (m_indices == -1) get D[i] = 0.
    """
    m = a_packed.shape[0]
    c = torch.zeros(m, n, dtype=torch.float32)
    m_idx_cpu = m_indices.cpu()
    for g in range(num_groups):
        rows = (m_idx_cpu == g).nonzero(as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        a_g = a_packed[rows]
        sf_a_g = sf_a[rows] if sf_a is not None else None
        sf_b_g = sf_b_grouped[g] if sf_b_grouped is not None else None
        c_g = fp4_reference(a_g, b_packed_grouped[g], rows.numel(), n, sf_a_g, sf_b_g)
        c[rows] = c_g
    return c


def run_kernel_grouped(a_packed, b_packed, sf_a, sf_b, m_indices, m, n, recipe=(1, 1, 128)):
    """调用 m_grouped_fp8_gemm_nt_contiguous 的 FP4 路径 (int32 dtype 触发)。"""
    duc = not get_ue8m0_usage(KernelType.Kernel1D1D)
    d = torch.empty((m, n), device='cuda', dtype=torch.float32)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (a_packed, sf_a), (b_packed, sf_b), d, m_indices,
        recipe=recipe, disable_ue8m0_cast=duc,
    )
    torch.cuda.synchronize()
    return d


def pack_fp4_random_3d_ga(num_groups: int, max_m: int, k_fp4: int, device='cuda'):
    """生成 [G, max_M, K_int32] FP4 packed tensor (A side for masked variant)."""
    assert k_fp4 % 8 == 0
    raw = torch.randint(0, 16, (num_groups, max_m, k_fp4), dtype=torch.uint8, device=device)
    packed = torch.zeros(num_groups, max_m, k_fp4 // 8, dtype=torch.int32, device=device)
    for i in range(8):
        packed += (raw[:, :, i::8].to(torch.int32) << (i * 4))
    return packed


def generate_mxf4_sfa_3d(num_groups: int, max_m: int, k_fp4: int, device='cuda', random_sf=False):
    """生成 SFA [G, max_M, sf_k] for grouped masked."""
    VS = 32
    sf_k = ((k_fp4 // VS + 3) // 4) * 4
    if random_sf:
        powers = torch.randint(-2, 3, (num_groups, max_m, sf_k), device=device).float()
        return torch.pow(2.0, powers)
    return torch.ones((num_groups, max_m, sf_k), dtype=torch.float32, device=device)


def fp4_reference_masked(a_3d, b_3d, masked_m_cpu, max_m, n, num_groups,
                         sf_a_3d=None, sf_b_3d=None):
    """Reference for masked variant: D[g, :masked_m[g], :] = A[g, :masked_m[g], :] @ B[g, :, :].T.

    Padding rows beyond masked_m[g] in D are not checked (kernel may leave any value).
    Returns d_ref of shape [G, max_M, N] with valid rows filled, padding rows zeroed.
    """
    d_ref = torch.zeros(num_groups, max_m, n, dtype=torch.float32)
    for g in range(num_groups):
        mg = int(masked_m_cpu[g].item())
        if mg == 0:
            continue
        a_g = a_3d[g, :mg]
        sf_a_g = sf_a_3d[g, :mg] if sf_a_3d is not None else None
        sf_b_g = sf_b_3d[g] if sf_b_3d is not None else None
        c_g = fp4_reference(a_g, b_3d[g], mg, n, sf_a_g, sf_b_g)
        d_ref[g, :mg] = c_g
    return d_ref


def run_kernel_grouped_masked(a_3d, b_3d, sf_a_3d, sf_b_3d, masked_m, num_groups,
                              max_m, n, expected_m, recipe=(1, 1, 128)):
    """调用 m_grouped_fp8_gemm_nt_masked FP4 路径。"""
    duc = not get_ue8m0_usage(KernelType.Kernel1D1D)
    d = torch.empty((num_groups, max_m, n), device='cuda', dtype=torch.float32)
    deep_gemm.m_grouped_fp8_gemm_nt_masked(
        (a_3d, sf_a_3d), (b_3d, sf_b_3d), d, masked_m, expected_m,
        recipe=recipe, disable_ue8m0_cast=duc,
    )
    torch.cuda.synchronize()
    return d


# ============================================================
# 测试用例
# ============================================================

def test_constant():
    """全 1.0 常量测试: C[i,j] = K (因为 1.0 * 1.0 * K 个元素)"""
    print('Test: constant values (all E2M1 1.0)')
    configs = [
        # (M, N, K_fp4) — K 是 FP4 元素个数
        # 单 stage (K <= 256)
        (32,   64,  256),
        (128, 128,  128),
        (128, 256,  256),
        # 多 stage (K > 256)
        (128, 128,  512),
        (256, 256,  512),
        (128, 128, 1024),
        (128, 256, 1024),
        # 大 M (multi-wave: BLOCK_M=128, 所以 M>128 需要多个 wave)
        (256, 128,  256),
        (256, 256, 1024),
    ]
    all_pass = True
    for m, n, k in configs:
        a = pack_fp4_constant(m, k, fp4_bits=0x2)
        b = pack_fp4_constant(n, k, fp4_bits=0x2)
        sf_a, sf_b = generate_mxf4_scale_factors(m, n, k)
        d = run_kernel(a, b, sf_a, sf_b, m, n)
        expected = float(k)
        ok = (d.cpu() == expected).all().item()
        if not ok:
            all_pass = False
        print(f'  M={m:4d} N={n:4d} K={k:4d}: expected={expected:.0f} got={d.cpu()[0,0].item():.0f} {"PASS" if ok else "FAIL"}')
    return all_pass


def test_random():
    """随机数据测试 (SF=1.0): 对比 GPU kernel 与 CPU reference"""
    print('Test: random data (vs CPU reference, SF=1.0)')
    configs = [
        (32,   64,  256),
        (128,  128, 128),
        (128,  256, 256),
        # 多 stage
        (128,  128, 512),
        (256,  256, 512),
        (128,  128, 1024),
        # 大 M
        (256,  128, 256),
        (256,  256, 1024),
        # 较大 N
        (128,  512, 256),
    ]
    all_pass = True
    for m, n, k in configs:
        a = pack_fp4_random(m, k)
        b = pack_fp4_random(n, k)
        sf_a, sf_b = generate_mxf4_scale_factors(m, n, k)
        d = run_kernel(a, b, sf_a, sf_b, m, n)
        ref = fp4_reference(a, b, m, n, sf_a, sf_b)
        max_diff = torch.abs(d.cpu().float() - ref.float()).max().item()
        ok = max_diff < 1.0
        if not ok:
            all_pass = False
        print(f'  M={m:4d} N={n:4d} K={k:4d}: max_diff={max_diff:.4f} {"PASS" if ok else "FAIL"}')
    return all_pass


def test_value_sweep():
    """不同 FP4 值测试: 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0"""
    print('Test: FP4 value sweep')
    m, n, k = 128, 128, 256
    all_pass = True
    for bits, val in [(0x1, 0.5), (0x2, 1.0), (0x3, 1.5), (0x4, 2.0),
                      (0x5, 3.0), (0x6, 4.0), (0x7, 6.0)]:
        a = pack_fp4_constant(m, k, fp4_bits=bits)
        b = pack_fp4_constant(n, k, fp4_bits=bits)
        sf_a, sf_b = generate_mxf4_scale_factors(m, n, k)
        d = run_kernel(a, b, sf_a, sf_b, m, n)
        expected = float(k) * val * val
        actual = d.cpu()[0, 0].item()
        ok = abs(actual - expected) < 1.0
        if not ok:
            all_pass = False
        print(f'  FP4={val:4.1f} (bits=0x{bits:X}): expected={expected:.1f} got={actual:.1f} {"PASS" if ok else "FAIL"}')
    return all_pass


def test_uniform_sf():
    """Uniform SF 测试: 验证 UTCCP SF->TMEM 路径在不同 SF 值下正确"""
    print('Test: uniform scale factors (UTCCP path)')
    configs = [
        (128, 256, 256),
        (256, 128, 256),
        (128, 128, 512),
    ]
    all_pass = True
    for m, n, k in configs:
        for sf_val in [0.25, 0.5, 1.0, 2.0, 4.0]:
            a = pack_fp4_random(m, k)
            b = pack_fp4_random(n, k)
            sf_a, sf_b = generate_mxf4_scale_factors(m, n, k)
            sf_a.fill_(sf_val)
            sf_b.fill_(sf_val)
            d = run_kernel(a, b, sf_a, sf_b, m, n)
            # uniform SF: kernel result = unscaled_result * sf_a * sf_b
            ref = fp4_reference(a, b, m, n, sf_a, sf_b)
            max_diff = torch.abs(d.cpu().float() - ref.float()).max().item()
            ok = max_diff < 1.0
            if not ok:
                all_pass = False
            print(f'  M={m:4d} N={n:4d} K={k:4d} SF={sf_val:5.2f}: max_diff={max_diff:.4f} {"PASS" if ok else "FAIL"}')
    return all_pass


def test_asymmetric_values():
    """A 和 B 使用不同 FP4 值"""
    print('Test: asymmetric A/B values')
    m, n, k = 128, 128, 256
    all_pass = True
    cases = [
        (0x2, 0x4, 1.0, 2.0),   # A=1.0, B=2.0
        (0x1, 0x6, 0.5, 4.0),   # A=0.5, B=4.0
        (0x4, 0x1, 2.0, 0.5),   # A=2.0, B=0.5
    ]
    for bits_a, bits_b, val_a, val_b in cases:
        a = pack_fp4_constant(m, k, fp4_bits=bits_a)
        b = pack_fp4_constant(n, k, fp4_bits=bits_b)
        sf_a, sf_b = generate_mxf4_scale_factors(m, n, k)
        d = run_kernel(a, b, sf_a, sf_b, m, n)
        expected = float(k) * val_a * val_b
        actual = d.cpu()[0, 0].item()
        ok = abs(actual - expected) < 1.0
        if not ok:
            all_pass = False
        print(f'  A={val_a}, B={val_b}: expected={expected:.1f} got={actual:.1f} {"PASS" if ok else "FAIL"}')
    return all_pass


def test_random_sf():
    """随机数据 + 随机 per-group SF (powers of 2)"""
    print('Test: random data + random scale factors')
    configs = [
        (128, 128, 256),
        (128, 256, 256),
        (256, 256, 512),
        (128, 128, 512),
        (128, 128, 1024),
        (256, 128, 256),
        (256, 256, 1024),
        (128, 256, 1024),
    ]
    all_pass = True
    for m, n, k in configs:
        a = pack_fp4_random(m, k)
        b = pack_fp4_random(n, k)
        sf_a, sf_b = generate_mxf4_scale_factors(m, n, k, random_sf=True)
        d = run_kernel(a, b, sf_a, sf_b, m, n)
        ref = fp4_reference(a, b, m, n, sf_a, sf_b)
        max_diff = torch.abs(d.cpu().float() - ref.float()).max().item()
        ok = max_diff < 1.0
        if not ok:
            all_pass = False
        print(f'  M={m:4d} N={n:4d} K={k:4d}: max_diff={max_diff:.4f} {"PASS" if ok else "FAIL"}')
    return all_pass


def test_m_grouped_contiguous():
    """M-grouped contiguous FP4 GEMM (MoE forward shape).

    A=[M_total, K] @ B=[G, N, K].T → D=[M_total, N]
    m_indices[M_total] selects which group each row belongs to (-1 = padding).
    """
    print('Test: m-grouped contiguous (MoE-style)')
    BLOCK_M = 128

    # Small/debug shapes — fast CPU LUT reference, exercise padding-row case.
    debug_configs = [
        (2, 128, 128,  256),
        (4, 128, 128,  256),
        (4, 128, 256,  512),
        (4, 256, 128, 1024),
        (8, 128, 256,  512),
        # Uneven actual M per group, padded to BLOCK_M (exercises padding rows).
        (4,  90, 128,  256),
        (4, 200, 256,  512),
        # Extra N×K variety to exercise more BLOCK_N choices and SF transform paths.
        (4, 128,  512,  256),
        (4, 128,  768,  256),
    ]

    # Production MoE shapes — mirror DeepGEMM official FP8 grouped (generators.py:105).
    # CPU LUT reference takes a few seconds per shape at this size; OK for nightly.
    prod_configs = [
        (4, 8192, 4096, 7168),  # EP4, MoE up-projection
        (4, 8192, 7168, 2048),  # EP4, MoE down-projection
        (8, 4096, 4096, 7168),  # EP8, MoE up-projection
        (8, 4096, 7168, 2048),  # EP8, MoE down-projection
        # Extra (n, k) variants from FP8 enumerate_normal for N/K coverage:
        (4, 8192, 24576, 1536),
        (4, 8192, 32768,  512),
    ]

    # Mirror FP8 masked-test pattern: multiple random-data iterations per shape to
    # catch flaky bugs that only show up with certain RNG seeds. Debug shapes get
    # fewer iters (kept fast); prod shapes get 3 iters each.
    NUM_ITERS = {'debug': 2, 'prod': 3}

    all_pass = True
    for label, configs in [('debug', debug_configs), ('prod', prod_configs)]:
        for num_groups, m_per_group, n, k in configs:
            aligned_m = ((m_per_group + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
            m_total = aligned_m * num_groups

            worst_diff = 0.0
            for _ in range(NUM_ITERS[label]):
                # Build A, B, m_indices fresh per iteration
                a = pack_fp4_random(m_total, k)
                b = pack_fp4_random_3d(num_groups, n, k)
                sf_a, _ = generate_mxf4_scale_factors(m_total, n, k, random_sf=True)
                sf_b = generate_mxf4_sf_3d(num_groups, n, k, random_sf=True)
                m_indices = torch.empty(m_total, dtype=torch.int32, device='cuda')
                for g in range(num_groups):
                    start = g * aligned_m
                    actual_end = start + m_per_group
                    aligned_end = start + aligned_m
                    m_indices[start:actual_end] = g
                    m_indices[actual_end:aligned_end] = -1

                d = run_kernel_grouped(a, b, sf_a, sf_b, m_indices, m_total, n)
                d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)

                ref = fp4_reference_grouped(a, b, m_indices, n, num_groups, sf_a, sf_b)
                d_max = torch.abs(d.cpu().float() - ref.float()).max().item()
                if d_max > worst_diff:
                    worst_diff = d_max

            ok = worst_diff < 1.0
            if not ok:
                all_pass = False
            print(f'  [{label}×{NUM_ITERS[label]}] G={num_groups} m_per_group={m_per_group:5d} '
                  f'(aligned={aligned_m:5d}) N={n:5d} K_fp4={k:5d}: '
                  f'max_diff={worst_diff:.4f} {"PASS" if ok else "FAIL"}')

    # Perf section — bench_kineto on production shapes only.
    # Filter to the FP4 GEMM kernel name; works for both dense and grouped wrappers
    # because the device kernel function is sm100_fp4_gemm_1d1d_impl in both cases.
    print('\nPerf: m-grouped contiguous (production MoE shapes)')
    duc = not get_ue8m0_usage(KernelType.Kernel1D1D)
    for num_groups, m_per_group, n, k in prod_configs:
        aligned_m = ((m_per_group + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
        m_total = aligned_m * num_groups

        a = pack_fp4_random(m_total, k)
        b = pack_fp4_random_3d(num_groups, n, k)
        sf_a, _ = generate_mxf4_scale_factors(m_total, n, k, random_sf=True)
        sf_b = generate_mxf4_sf_3d(num_groups, n, k, random_sf=True)
        m_indices = torch.empty(m_total, dtype=torch.int32, device='cuda')
        for g in range(num_groups):
            start = g * aligned_m
            m_indices[start:start + m_per_group] = g
            m_indices[start + m_per_group:start + aligned_m] = -1
        d = torch.empty((m_total, n), device='cuda', dtype=torch.float32)

        def fn():
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (a, sf_a), (b, sf_b), d, m_indices,
                recipe=(1, 1, 128), disable_ue8m0_cast=duc,
            )

        t = bench_kineto(fn, 'sm100_fp4_gemm', suppress_kineto_output=True)
        # FLOPs: 2 * M_total * N * K_fp4 (k here is K_fp4 element count).
        # Bytes: A (int32, 4B/elem), B (int32, 4B/elem), D (fp32, 4B/elem), plus SF.
        tflops = 2 * m_total * n * k / t / 1e12
        gbps = count_bytes(a, b, d) / 1e9 / t
        print(f'  G={num_groups} m_per_group={m_per_group:5d} N={n:5d} K_fp4={k:5d}: '
              f'{t * 1e6:6.1f} us | {tflops:6.0f} TFLOPS | {gbps:5.0f} GB/s')

    return all_pass


def test_m_grouped_trtllm_comparable():
    """trtllm-gen apples-to-apples: DeepSeek-R1 MoE setup with 256 experts × topK=8.

    Setup mirrors gitlab/trtllm BatchedGemm baseline:
      num_groups = 256 (numExperts)
      total_actual_M = numTokens * topK = numTokens * 8
      Uniform routing → rows_per_expert = total_actual_M / 256
      Each expert tile padded to BLOCK_M=128

    CAVEAT: trtllm-gen's BatchedGemmFp4LowLatency uses batch=N (grouped-along-N);
    DG here uses batch=M. Memory access patterns differ but useful FLOPs compare directly.

    trtllm-gen baseline on B200 (from memory/fp4_grouped_gemm_perf_work.md):

      FC2 (N=7168, K=2048):
        tokens=32   → 0.37 ms |  20.4 TFLOPS | 5.74 TB/s
        tokens=64   → 0.37 ms |  40.7 TFLOPS | 5.75 TB/s
        tokens=128  → 0.37 ms |  80.8 TFLOPS | 5.72 TB/s
        tokens=256  → 0.37 ms | 161.9 TFLOPS | 5.78 TB/s
        tokens=512  → 0.55 ms | 219.6 TFLOPS | 3.98 TB/s
        tokens=1024 → 0.51 ms | 472.3 TFLOPS | 4.40 TB/s
        tokens=2048 → 0.51 ms | 941.2 TFLOPS | 4.63 TB/s

      FC1 (N=4096, K=7168, fusedAct=swiglu, routeAct=tma) — note batch=N caveat:
        tokens=32   → 0.76 ms |  19.9 TFLOPS | 5.59 TB/s
        tokens=64   → 0.75 ms |  39.8 TFLOPS | 5.61 TB/s
        tokens=128  → 0.76 ms |  78.7 TFLOPS | 5.54 TB/s
        tokens=256  → 0.76 ms | 157.5 TFLOPS | 5.55 TB/s
        tokens=512  → 0.78 ms | 310.1 TFLOPS | 5.48 TB/s
        tokens=1024 → 0.82 ms | 590.0 TFLOPS | 5.25 TB/s
        tokens=2048 → 1.19 ms | 810.9 TFLOPS | 3.65 TB/s
    """
    print('Test: m-grouped contiguous (trtllm-gen comparable, 256 experts)')
    BLOCK_M = 128
    NUM_GROUPS = 256
    TOP_K = 8

    # Latency regime (trtllm-gen batch=N baseline): tokens 32..2048, per_group_M < BLOCK_M
    # Throughput regime (trtllm-gen batch=M target): tokens 4096..8192, per_group_M >= BLOCK_M
    # 4096: rows/grp=128 (= BLOCK_M, zero padding) — first apples-to-apples vs trtllm batch=M
    # 8192: rows/grp=256 (= 2×BLOCK_M, 2 tiles per group)
    token_sweep = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    shape_configs = [
        ('FC2', 7168, 2048),
        ('FC1', 4096, 7168),
    ]

    duc = not get_ue8m0_usage(KernelType.Kernel1D1D)
    all_pass = True

    for fc_name, n, k in shape_configs:
        print(f'\n--- {fc_name}: N={n}, K_fp4={k}, num_groups={NUM_GROUPS} ---')
        # Allocate B/SFB once per shape (large, reuse across token sweep — independent of tokens)
        b = pack_fp4_random_3d(NUM_GROUPS, n, k)
        sf_b = generate_mxf4_sf_3d(NUM_GROUPS, n, k, random_sf=True)

        for num_tokens in token_sweep:
            total_actual_m = num_tokens * TOP_K
            rows_per_group = max(1, total_actual_m // NUM_GROUPS)
            # Per-group rows aligned UP to BLOCK_M (may exceed BLOCK_M for tokens >= 4096)
            aligned_per_group = ((rows_per_group + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
            total_padded_m = NUM_GROUPS * aligned_per_group

            # A/SF_A/D scale with total_padded_m, so allocate per iteration.
            a = pack_fp4_random(total_padded_m, k)
            sf_a, _ = generate_mxf4_scale_factors(total_padded_m, n, k, random_sf=True)
            d = torch.empty((total_padded_m, n), device='cuda', dtype=torch.float32)

            # m_indices: each group gets aligned_per_group rows, first rows_per_group are valid (g),
            # remainder padding (-1).
            m_indices = torch.full((total_padded_m,), -1, dtype=torch.int32, device='cuda')
            for g in range(NUM_GROUPS):
                start = g * aligned_per_group
                m_indices[start:start + rows_per_group] = g
                # rows [start + rows_per_group : start + aligned_per_group) stay -1

            # Run kernel
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (a, sf_a), (b, sf_b), d, m_indices,
                recipe=(1, 1, 128), disable_ue8m0_cast=duc,
            )
            torch.cuda.synchronize()
            d_clean = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)

            # Correctness: only for smaller tokens (CPU LUT reference for large M is slow)
            if num_tokens <= 256:
                ref = fp4_reference_grouped(a, b, m_indices, n, NUM_GROUPS, sf_a, sf_b)
                max_diff = torch.abs(d_clean.cpu().float() - ref.float()).max().item()
                ok = max_diff < 1.0
                if not ok:
                    all_pass = False
                correctness = f'diff={max_diff:.4f} {"✓" if ok else "✗"}'
            else:
                correctness = '(skip)'

            # Perf
            def fn():
                deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                    (a, sf_a), (b, sf_b), d, m_indices,
                    recipe=(1, 1, 128), disable_ue8m0_cast=duc,
                )
            t = bench_kineto(fn, 'sm100_fp4_gemm', suppress_kineto_output=True)

            # Useful FLOPS = 2 * useful_M * N * K (matches trtllm-gen convention)
            useful_m = NUM_GROUPS * rows_per_group
            useful_tflops = 2 * useful_m * n * k / t / 1e12
            # Bytes: A loaded once (total_padded_m), B loaded once, D written once
            gbps = count_bytes(a, b, d) / 1e9 / t
            print(f'  tokens={num_tokens:5d} useful_M={useful_m:6d} '
                  f'rows/grp={rows_per_group:4d}/{aligned_per_group:4d}: '
                  f'{t * 1e3:6.3f} ms | {useful_tflops:7.1f} TFLOPS | '
                  f'{gbps/1000:5.2f} TB/s | {correctness}')

    return all_pass


def test_m_grouped_masked():
    """M-grouped masked FP4 GEMM (MoE decode shape).

    A=[G, max_M, K] @ B=[G, N, K].T → D=[G, max_M, N]
    masked_m[G] indicates the number of valid rows per group (rest is padding).
    """
    print('Test: m-grouped masked (MoE decode)')
    BLOCK_M = 128

    # (num_groups, max_m, expected_m, n, k_fp4)
    # Small/debug shapes (CPU reference fast) + production shapes mirroring FP8 enumerate_m_grouped_masked
    debug_configs = [
        (2,  128,  64, 128,  256),    # 50% util
        (4,  128,  32, 128,  256),    # heavy padding
        (4,  256, 200, 128,  256),    # multi-tile per group
        (4,  128, 128, 256,  512),    # full
        (8,  128,  64, 256,  512),
    ]
    prod_configs = [
        # max_m=4096 (matching FP8 enumerate_m_grouped_masked max_m), varying num_groups & m
        (1, 4096, 1024, 4096, 7168),  # FP8 (1, 1024)
        (2, 4096,  512, 4096, 7168),  # FP8 (2, 512)
        (4, 4096,  256, 4096, 7168),  # FP8 (4, 256)
        (1, 4096, 1024, 7168, 2048),
        (2, 4096,  512, 7168, 2048),
        (4, 4096,  256, 7168, 2048),
    ]

    # Mirror FP8 test_m_grouped_gemm_masked: 10 random-data iterations per shape on
    # production shapes to catch flaky bugs (different masked_m distribution each time).
    NUM_ITERS = {'debug': 3, 'prod': 10}

    all_pass = True
    duc = not get_ue8m0_usage(KernelType.Kernel1D1D)
    for label, configs in [('debug', debug_configs), ('prod', prod_configs)]:
        for num_groups, max_m, expected_m, n, k in configs:
            worst_diff = 0.0
            for _ in range(NUM_ITERS[label]):
                # Fresh masked_m + tensors per iteration (matches FP8 pattern)
                masked_m_cpu = torch.tensor([
                    max(1, min(max_m, int(expected_m * random.uniform(0.7, 1.3))))
                    for _ in range(num_groups)
                ], dtype=torch.int32)
                masked_m = masked_m_cpu.cuda()

                a = pack_fp4_random_3d_ga(num_groups, max_m, k)
                b = pack_fp4_random_3d(num_groups, n, k)
                sf_a = generate_mxf4_sfa_3d(num_groups, max_m, k, random_sf=True)
                sf_b = generate_mxf4_sf_3d(num_groups, n, k, random_sf=True)

                d = run_kernel_grouped_masked(a, b, sf_a, sf_b, masked_m, num_groups,
                                              max_m, n, expected_m)
                ref = fp4_reference_masked(a, b, masked_m_cpu, max_m, n, num_groups, sf_a, sf_b)

                # Only compare valid rows per group
                for g in range(num_groups):
                    mg = int(masked_m_cpu[g].item())
                    if mg == 0:
                        continue
                    diff = torch.abs(d[g, :mg].cpu().float() - ref[g, :mg].float()).max().item()
                    if diff > worst_diff:
                        worst_diff = diff

            ok = worst_diff < 1.0
            if not ok:
                all_pass = False
            print(f'  [{label}×{NUM_ITERS[label]}] G={num_groups} max_m={max_m:5d} '
                  f'expected_m={expected_m:5d} N={n:5d} K_fp4={k:5d}: '
                  f'max_diff={worst_diff:.4f} {"PASS" if ok else "FAIL"}')

    return all_pass


def test_multicast():
    """大 M 测试：触发 B-multicast (M>=512, 2CTA along M, UMMA_M=256)"""
    print('Test: B-multicast (M>=512, 2CTA)')
    configs = [
        (512,  128,  256),
        (512,  128,  512),
        (512,  128, 1024),
        (1024, 128,  256),
        (1024, 128,  512),
    ]
    all_pass = True
    for m, n, k in configs:
        a = pack_fp4_random(m, k)
        b = pack_fp4_random(n, k)
        sf_a, sf_b = generate_mxf4_scale_factors(m, n, k, random_sf=True)
        d = run_kernel(a, b, sf_a, sf_b, m, n)
        ref = fp4_reference(a, b, m, n, sf_a, sf_b)
        max_diff = torch.abs(d.cpu().float() - ref.float()).max().item()
        ok = max_diff < 1.0
        if not ok:
            all_pass = False
        print(f'  M={m:4d} N={n:4d} K={k:4d}: max_diff={max_diff:.4f} {"PASS" if ok else "FAIL"}')
    return all_pass


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print(f'Library: {deep_gemm.__path__}\n')

    results = [
        ('constant',       test_constant()),
        ('random',         test_random()),
        ('sweep',          test_value_sweep()),
        ('asymmetric',     test_asymmetric_values()),
        ('uniform_sf',     test_uniform_sf()),
        ('random_sf',      test_random_sf()),
        ('multicast',      test_multicast()),
        ('m_grouped',      test_m_grouped_contiguous()),
        ('m_grouped_masked', test_m_grouped_masked()),
        ('trtllm_cmp',     test_m_grouped_trtllm_comparable()),
    ]

    print()
    passed = all(r for _, r in results)
    for name, ok in results:
        print(f'  {name}: {"PASS" if ok else "FAIL"}')
    print(f'\n{"ALL FP4 TESTS PASSED" if passed else "SOME TESTS FAILED"}')
    if not passed:
        exit(1)
