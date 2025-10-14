import copy
import random
import time
import torch
import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json

import deep_gemm
from deep_gemm.testing import (
    bench, bench_kineto,
    calc_diff, count_bytes
)

from generators import (
    KernelType, get_ue8m0_usage,
    enumerate_normal, enumerate_m_grouped_contiguous, enumerate_m_grouped_masked, enumerate_k_grouped_contiguous,
    generate_normal, generate_m_grouped_contiguous, generate_m_grouped_masked, generate_k_grouped_contiguous
)


@dataclass
class GemmTestResult:
    """单次GEMM测试的结果"""
    # 测试配置
    m: int
    n: int
    k: int
    kernel_type: str  # '1D1D' or '1D2D'
    layout: str       # 'NT', 'TN', 'NN', 'TT'
    out_dtype: str    # 'FP32' or 'BF16'
    accumulate: bool
    
    # 性能指标
    launch_time_us: float    # 启动时间 (微秒)
    execution_time_us: float # 执行时间 (微秒)
    tflops: float           # TFLOPS
    bandwidth_gb_s: float   # 带宽 (GB/s)
    
    # 正确性验证
    diff: float             # 与参考结果的差异
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'config': {
                'm': int(self.m),
                'n': int(self.n),
                'k': int(self.k),
                'kernel_type': str(self.kernel_type),
                'layout': str(self.layout),
                'out_dtype': str(self.out_dtype),
                'accumulate': bool(self.accumulate)
            },
            'performance': {
                'launch_time_us': float(self.launch_time_us),
                'execution_time_us': float(self.execution_time_us),
                'tflops': float(self.tflops),
                'bandwidth_gb_s': float(self.bandwidth_gb_s)
            },
            'correctness': {
                'diff': float(self.diff)
            }
        }


def test_gemm(collect_results: bool = False) -> List[GemmTestResult]:
    """
    测试GEMM函数
    
    Args:
        collect_results: 是否收集测试结果而不是打印
    
    Returns:
        如果collect_results为True，返回测试结果列表；否则返回空列表
    """
    
    results = []
    
    for kernel_type, m, n, k, major_a, major_b, accumulate, out_dtype in enumerate_normal():
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        out_opt    = 'FP32' if out_dtype == torch.float else 'BF16'
        acc_opt    = f'acc={int(accumulate)}'
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0

        # 测试正确性
        max_diff = 0.0
        for test_alias in (False, True):
            a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, use_ue8m0=use_ue8m0)
            func_name = f'fp8_gemm_{major_opt.lower() if test_alias else "nt"}'
            if test_alias:
                a = a if major_a.is_k_major() else (a[0].T, a[1].T)
                b = b if major_b.is_k_major() else (b[0].T, b[1].T)
                assert a[0].is_contiguous() and b[0].is_contiguous()
            getattr(deep_gemm, func_name)(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast)
            diff = calc_diff(d, ref_d)
            max_diff = max(max_diff, diff)
            assert diff < 0.001, (f'{m=}, {n=}, {k=}, {kernel_opt}, {major_opt=}, {accumulate=}, {out_dtype=}, '
                                  f'{diff:.5f}, alias={test_alias}')
        
        # 性能测试
        a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, use_ue8m0=use_ue8m0)

        # Test launch overhead
        launch_start_t = time.time_ns()
        deep_gemm.fp8_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast)
        launch_end_t = time.time_ns()
        torch.cuda.synchronize()

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.fp8_gemm_nt(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=False, trace_path="trace.json")
        
        # 计算性能指标
        launch_time_us = (launch_end_t - launch_start_t) / 1e3
        execution_time_us = t * 1e6
        tflops = 2 * m * n * k / t / 1e12
        bandwidth_gb_s = (count_bytes(a, b, d) + count_bytes(c) * int(accumulate)) / 1e9 / t
        
        if collect_results:
            # 收集结果
            result = GemmTestResult(
                m=m, n=n, k=k,
                kernel_type=kernel_opt,
                layout=major_opt,
                out_dtype=out_opt,
                accumulate=accumulate,
                launch_time_us=launch_time_us,
                execution_time_us=execution_time_us,
                tflops=tflops,
                bandwidth_gb_s=bandwidth_gb_s,
                diff=max_diff
            )
            results.append(result)
        
        print(f' > Perf (m={m:5}, n={n:5}, k={k:5}, {kernel_opt}, layout={major_opt}, {out_opt}, {acc_opt}): '
                f'launch {launch_time_us:4.0f} us | {execution_time_us:4.0f} us | '
                f'{tflops:4.0f} TFLOPS | '
                f'{bandwidth_gb_s:4.0f} GB/s')
    
    return results


def test_image_configs(save_results: bool = True, results_file: str = "gps_test_results.json"):
    """
    测试不同的图片配置值
    
    Args:
        save_results: 是否保存结果到文件
        results_file: 结果文件名
    
    Returns:
        GPSTestResultManager: 包含所有测试结果的管理器
    """
    print('Testing GPS Image Configurations:')
    
    # 有效的图片配置n值
    # 只包含满足FP8缩放约束的值 (当BLOCK_K=128时)
    # 约束条件: ceil_div(BLOCK_N, BLOCK_K) == 1 or (gcd(BLOCK_N, BLOCK_K) == BLOCK_N - BLOCK_K)
    # 对于BLOCK_K=128，满足条件的N值主要是 <= 128 的值
    valid_n_values = [
        0, 16, 32, 48, 64, 80, 96, 112, 128, 
            144, 160, 176, 192, 208, 224, 240, 256,
            272, 288, 304, 320, 336, 352, 368, 384, 
            400, 416, 432, 448, 464, 480, 496, 512
    ]

    valid_m_values = [64, 128, 256, 512]
    
    # 保存原始环境变量值
    original_n_value = os.environ.get('DG_GPS_CONFIG_N', '8')
    original_m_value = os.environ.get('DG_GPS_CONFIG_M', '64')
    
    try:
        total_configs = len(valid_m_values) * len(valid_n_values)
        current_config = 0
        
        for m_value in valid_m_values:
            os.environ['DG_GPS_CONFIG_M'] = str(m_value)
            for n_value in valid_n_values:
                current_config += 1
                print(f'\n=== Testing GPS Config [{current_config}/{total_configs}]: n={n_value}, m={m_value} ===')
                
                # 直接设置环境变量
                os.environ['DG_GPS_CONFIG_N'] = str(n_value)
            
                try:
                    # 运行测试并收集结果
                    test_results = test_gemm(collect_results=True)
                    
                    # 保存test_results到文件
                    with open("test_results.json", "a") as f:
                        result_data = {
                            "m_value": m_value,
                            "n_value": n_value,
                            "test_results": [result.to_dict() for result in test_results]
                        }
                        f.write(json.dumps(result_data) + "\n")
                        
                except Exception as e:
                    error_msg = str(e)
                    if "Too much B scales in a single block" in error_msg:
                        print(f'    Skipping config n={n_value}, m={m_value}: FP8 scaling constraint violation')
                    elif "JSON serializable" in error_msg:
                        print(f'    Skipping config n={n_value}, m={m_value}: JSON serialization error')
                    else:
                        print(f'    Error testing config n={n_value}, m={m_value}: {e}')
                    continue
            
    finally:
        # 恢复原始环境变量值
        os.environ['DG_GPS_CONFIG_N'] = original_n_value
        os.environ['DG_GPS_CONFIG_M'] = original_m_value
    
    print('\nGPS configuration testing completed!')


def test_m_grouped_gemm_contiguous() -> None:
    print('Testing m-grouped contiguous GEMM:')

    for kernel_type, num_groups, expected_m_per_group, n, k, major_a, major_b in enumerate_m_grouped_contiguous(): # sm90 返回的major_a, major_b是(MajorTypeAB.KMajor,  MajorTypeAB.KMajor)
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type) #sm90 就会返回false
        disable_ue8m0_cast = not use_ue8m0

        for test_alias in (False, True):
            m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b, use_ue8m0=use_ue8m0)
            func_name = f"m_grouped_fp8_gemm_{(major_opt.lower() if test_alias else 'nt')}_contiguous" #sm90 返回的是m_grouped_fp8_gemm_nt_contiguous
            if test_alias:
                assert major_a.is_k_major()
                b = b if major_b.is_k_major() else (b[0].mT, b[1].mT)
                assert a[0].is_contiguous() and b[0].is_contiguous()
            getattr(deep_gemm, func_name)(a, b, d, m_indices, disable_ue8m0_cast=disable_ue8m0_cast)
            d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
            diff = calc_diff(d, ref_d)
            assert diff < 0.001, f'{m=}, {n=}, {k=}, {major_opt}, {kernel_opt}, {diff:.5f}, alias={test_alias}'
        m, a, b, m_indices, d, ref_d = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b, use_ue8m0=use_ue8m0)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a, b, d, m_indices, disable_ue8m0_cast=disable_ue8m0_cast)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, m={m:5}, n={n:5}, k={k:5}, {kernel_opt}, layout={major_opt}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_masked() -> None:
    print('Testing m-grouped masked GEMM:')

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    for kernel_type, num_groups, max_m, expected_m_per_group, n, k in enumerate_m_grouped_masked():
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type)
        disable_ue8m0_cast = not use_ue8m0

        # Test correctness
        for i in range(10):
            a, b, masked_m, d, ref_d = generate_m_grouped_masked(num_groups, max_m, expected_m_per_group, n, k, use_ue8m0=use_ue8m0)
            deep_gemm.m_grouped_fp8_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group, disable_ue8m0_cast=disable_ue8m0_cast)
            for j in range(num_groups):
                diff = calc_diff(d[j, :masked_m[j].item()], ref_d[j, :masked_m[j].item()])
                assert diff < 0.001, f'{max_m=}, {n=}, {k=}, {j=}, masked_m={masked_m[j]}, {kernel_opt}, {num_groups=}, {diff:.5f}'

        # Construct full cases
        a, b, masked_m, d, ref_d = generate_m_grouped_masked(num_groups, max_m, expected_m_per_group, n, k, use_ue8m0=use_ue8m0)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_fp8_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group, disable_ue8m0_cast=disable_ue8m0_cast)

        # Test performance with fixed shapes
        valid_m = masked_m.sum().item()
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}, {kernel_opt}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * valid_m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, d) * valid_m / (max_m * num_groups) + count_bytes(b)) / 1e9 / t:4.0f} GB/s')
    print()


def test_k_grouped_gemm_contiguous() -> None:
    print('Testing k-grouped contiguous GEMM:')

    for num_groups, m, n, ks, expected_k_per_group in enumerate_k_grouped_contiguous():
        use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D)

        for test_empty_groups in (False, True):
            new_ks = copy.deepcopy(ks)
            if test_empty_groups:
                new_ks[random.randint(0, num_groups - 1)] = 0
            k, a, b, c, d, ref_d = generate_k_grouped_contiguous(num_groups, m, n, new_ks, use_ue8m0=use_ue8m0)
            new_ks_tensor = torch.tensor(new_ks, dtype=torch.int, device='cuda')
            deep_gemm.k_grouped_fp8_gemm_tn_contiguous(a, b, d, new_ks, new_ks_tensor, c=c)
            diff = calc_diff(d, ref_d)
            assert diff < 0.001, f'{m=}, {n=}, {k=}, {i=}, {diff:.5f}'

        # Test performance
        k, a, b, c, d, ref_d = generate_k_grouped_contiguous(num_groups, m, n, ks, use_ue8m0=use_ue8m0)
        ks_tensor = torch.tensor(ks, dtype=torch.int, device='cuda')

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.k_grouped_fp8_gemm_tn_contiguous(a, b, d, ks, ks_tensor, c=c)

        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=:2}, m={m:5}, n={n:5}, k={k:5}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, c, d) / 1e9 / t:4.0f} GB/s')
    print()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    # 检查命令行参数
    if '--gps-configs' in sys.argv or '--image-configs' in sys.argv:
        # 运行GPS配置测试
        results_file = "gps_test_results.json"
        if '--output' in sys.argv:
            idx = sys.argv.index('--output')
            if idx + 1 < len(sys.argv):
                results_file = sys.argv[idx + 1]
        
        # manager = test_image_configs(save_results=True, results_file=results_file)
        test_image_configs(save_results=True, results_file=results_file)
        
    elif '--analyze' in sys.argv:
        # 分析已有结果
        results_file = "gps_test_results.json"
        if '--file' in sys.argv:
            idx = sys.argv.index('--file')
            if idx + 1 < len(sys.argv):
                results_file = sys.argv[idx + 1]
        analyze_gps_results(results_file)
        
    elif '--compare' in sys.argv:
        # 比较两个配置
        if len(sys.argv) >= 6:
            try:
                m1, n1 = int(sys.argv[2]), int(sys.argv[3])
                m2, n2 = int(sys.argv[4]), int(sys.argv[5])
                results_file = sys.argv[6] if len(sys.argv) > 6 else "gps_test_results.json"
                compare_gps_configs((m1, n1), (m2, n2), results_file)
            except ValueError:
                print("Usage: python test_fp8.py --compare M1 N1 M2 N2 [results_file]")
        else:
            print("Usage: python test_fp8.py --compare M1 N1 M2 N2 [results_file]")
            
    else:
        # 运行标准测试
        test_gemm()
        test_m_grouped_gemm_contiguous()
        test_m_grouped_gemm_masked()
        test_k_grouped_gemm_contiguous()
