import random
import torch
from typing import Tuple

import adaptive_gemm
from adaptive_gemm import bench_kineto, calc_diff, ceil_div, get_col_major_tma_aligned_tensor


def generate_random_list(length, total_sum):
    # 生成一个长度为length的列表，元素之和为total_sum
    # 先生成一个平均分配的列表
    avg = total_sum // length
    remainder = total_sum % length
    lst = [avg] * length

    # 随机调整数值，确保总和不变
    for i in range(length):
        # 随机选择两个不同的位置
        lst[i] = random.randint(0, int(avg))
    return lst

def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y

def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)

def per_channel_cast_to_fp8(x: torch.Tensor, dtype = torch.float8_e4m3fn) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, n)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    if dtype == torch.float8_e4m3fn:
        fmax = torch.finfo(torch.float8_e4m3fn).max
        return (x_view * (fmax / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / fmax).view(m, -1)
    else:
        fmax = torch.finfo(torch.float8_e5m2).max
        return (x_view * (fmax / x_amax.unsqueeze(2))).to(torch.float8_e5m2).view(m, n), (x_amax / fmax).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))

def per_expert_cast_to_fp8(x: torch.Tensor, dtype = torch.float8_e4m3fn) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 3
    num_groups, m, n = x.shape
    x_padded = torch.zeros((num_groups, ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:, :m, :n] = x
    x_view = x_padded.view(num_groups, m, 1, n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    if dtype == torch.float8_e4m3fn:
        fmax = torch.finfo(torch.float8_e4m3fn).max
        x_scaled = (x_view * (fmax / x_amax)).to(torch.float8_e4m3fn)
        return x_scaled.view_as(x_padded)[:, :m, :n].contiguous(), (x_amax / fmax).view(x_view.size(0), x_view.size(2))
    else:
        fmax = torch.finfo(torch.float8_e5m2).max
        x_scaled = (x_view * (fmax / x_amax)).to(torch.float8_e5m2)
        return x_scaled.view_as(x_padded)[:, :m, :n].contiguous(), (x_amax / fmax).view(x_view.size(0), x_view.size(2))


def gen_data_fwd(M, N, K, tokens_per_expert, dtype_out = torch.bfloat16, dtype_a = torch.float8_e4m3fn, dtype_b = torch.float8_e4m3fn):
    ref_dw = torch.empty(M, N, device = "cuda", dtype = dtype_out)
    x = torch.randn(M, K, device = "cuda", dtype = torch.bfloat16)

    num_groups = len(tokens_per_expert)

    y = torch.randn(num_groups, N, K, device = "cuda", dtype = torch.bfloat16)
    y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn), torch.empty((num_groups, (N + 127) // 128, K // 128), device='cuda', dtype=torch.float))

    x_fp8, x_scale = per_token_cast_to_fp8(x)

    for i in range(num_groups):
        y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])

    # prepare data
    t_start = 0
    for i, tokens in enumerate(tokens_per_expert):
        tokens = int(tokens)
        x_tmp = x[t_start: t_start+tokens]; 
        weight = y[i]

        ref_dw[t_start: t_start+tokens] = (x_tmp @ weight.T)

        t_start += tokens

    # breakpoint()
    return x_fp8, x_scale, y_fp8[0], y_fp8[1], ref_dw

if __name__=='__main__':
    from typing import Tuple
    import random
    
    print('Testing grouped contiguous GEMM:')

    dtype_a = torch.float8_e5m2
    dtype_b = torch.float8_e4m3fn

    dtype_out = torch.bfloat16

    for (N, K) in ((6144, 5120), (5120, 3072), (7168, 4096), (7168, 2048), (7168, 16384)):
        tokens_per_expert = generate_random_list(8, 16384)
        # tokens_per_expert = [2048] * 4
        tokens_per_expert = torch.tensor(tokens_per_expert, device='cuda', dtype=torch.long)
        num_groups = len(tokens_per_expert)
        print(tokens_per_expert)
        M = sum(tokens_per_expert)

        x_fp8, x_scale, weights_fp8, weights_scale, ref_fwd = gen_data_fwd(M, N, K, tokens_per_expert, dtype_out = dtype_out, dtype_a = dtype_a, dtype_b = dtype_b)
        size_per_group = torch.tensor(tokens_per_expert, device='cuda', dtype=torch.long)
        
        for i in range(3):
            output_tensor = adaptive_gemm.m_grouped_varlen_gemm_fp8_fp8_bf16_nt_contiguous((x_fp8, x_scale), (weights_fp8, weights_scale), size_per_group)

        

        def test_func():
            output_tensor = adaptive_gemm.m_grouped_varlen_gemm_fp8_fp8_bf16_nt_contiguous((x_fp8, x_scale), (weights_fp8, weights_scale), size_per_group)
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        print(f' > Performance ({num_groups=}, {M=}, {N=}, {K=}, {t * 1e6:4.0f} us | '
                f'throughput: {2 * M * N * K / t / 1e12:4.0f} TFLOPS, '
                f'{(num_groups * K * N + M * K +  M * N * 2) / 1e9 / t:4.0f} GB/s')
        amax = max(output_tensor.abs().max(), ref_fwd.abs().max())
        adiffmax = (output_tensor - ref_fwd).abs().max()
        rdiffmax = adiffmax / amax
        print(f"    max relative difference of the layer is {rdiffmax}")


    # from torch.profiler import ProfilerActivity, profile, record_function
    # with profile(
    #         activities=[
    #                 ProfilerActivity.CPU, ProfilerActivity.CUDA
    #         ],
    #         # with_stack = True,
    #         # with_modules = True,
    #         record_shapes=True,) as prof:
    #     output_tensor = adaptive_gemm.m_grouped_varlen_gemm_fp8_fp8_bf16_nt_contiguous((x_fp8, x_scale), (weights_fp8, weights_scale), size_per_group)
    
    # trace = f'./grouped_m_gemm.json'
    # prof.export_chrome_trace(trace)
    # # Get time from trace
    # import json
    # with open(trace, "r") as file:
    #     data = json.load(file)
    
    # kernel_time = 0
    # for event in data["traceEvents"]:
    #     if "fp8_gemm_kernel" in event["name"]:
    #         kernel_time += event["dur"] / 1000
    # print(f"\nPure kernel Elapsed time {round((kernel_time), 1)} ms, {round((2 * M.item() * N * K)/(kernel_time)/10**9, 0)} tflops")