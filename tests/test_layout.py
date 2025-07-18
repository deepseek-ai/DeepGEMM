import time
import torch
import random
from deep_gemm.testing import bench_kineto, count_bytes
from deep_gemm.utils import (
    align, ceil_div,
    per_token_cast_to_fp8, per_channel_cast_to_fp8,
    get_tma_aligned_size,
    get_mn_major_tma_aligned_packed_ue8m0_tensor,
    get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor
)

from generators import (
    enumerate_sf_layout,
    enumerate_k_grouped_sf_layout
)


def get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float and x.dim() in (2, 3)

    # First, convert into UE8M0 `uint8_t`
    ue8m0_tensor = (x.view(torch.int) >> 23).to(torch.uint8)

    # Second, make padded packed tensors
    mn, k = x.shape[-2], x.shape[-1]
    remove_dim = False
    if x.dim() == 2:
        x, remove_dim = x.unsqueeze(0), True
    b = x.shape[0]
    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = align(k, 4)
    padded = torch.zeros((b, aligned_mn, aligned_k), device=x.device, dtype=torch.uint8)
    padded[:, :mn, :k] = ue8m0_tensor
    padded = padded.view(-1).view(dtype=torch.int).view(b, aligned_mn, aligned_k // 4)

    # Finally, transpose
    transposed = torch.zeros((b, aligned_k // 4, aligned_mn), device=x.device, dtype=torch.int).mT
    transposed[:, :, :] = padded
    aligned_x = transposed[:, :mn, :]
    return aligned_x.squeeze(0) if remove_dim else aligned_x


def test_sf_layout_kernels() -> None:
    print('Testing SF layout kernels:')
    for mn, k, with_transpose, num_groups in enumerate_sf_layout():
        x = torch.randn((num_groups * mn, k), dtype=torch.bfloat16, device='cuda')
        x, fp32_sf = per_token_cast_to_fp8(x, use_ue8m0=True)
        fp32_sf = fp32_sf if num_groups == 1 else fp32_sf.view(num_groups, mn, -1)
        fp32_sf = fp32_sf if with_transpose else fp32_sf.transpose(-1, -2).contiguous().transpose(-1, -2)

        # Correctness
        packed_sf = get_mn_major_tma_aligned_packed_ue8m0_tensor(fp32_sf)
        ref_packed_sf = get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl(fp32_sf)
        assert torch.equal(packed_sf, ref_packed_sf), f'{mn=}, {k=}, {with_transpose=}, {num_groups=}'
        assert packed_sf.shape == ref_packed_sf.shape
        assert all([packed_sf.stride(i) == ref_packed_sf.stride(i) for i in range(packed_sf.dim())])

        # Test launch overhead
        launch_start_t = time.time_ns()
        get_mn_major_tma_aligned_packed_ue8m0_tensor(fp32_sf)
        launch_end_t = time.time_ns()

        # Performance
        t = bench_kineto(lambda: get_mn_major_tma_aligned_packed_ue8m0_tensor(fp32_sf), 'pack_fp32_into_ue8m0')
        print(f' > Perf ({num_groups=:2}, {mn=:5}, {k=:5}, transpose={int(with_transpose)}): '
              f'launch {(launch_end_t - launch_start_t) / 1e3:3.0f} us | {t * 1e6:4.0f} us | '
              f'{count_bytes(fp32_sf, packed_sf) / 1e9 / t:4.0f} GB/s')
    print()


def test_k_grouped_sf_layout_kernels() -> None:
    print('Testing k-grouped SF layout kernels:')
    for mn, ks, num_groups in enumerate_k_grouped_sf_layout():
        sf_ks = [k // 128 for k in ks]
        packed_sf_ks = [ceil_div(k, 512) for k in ks]
        ks_tensor = torch.tensor(ks, dtype=torch.int, device='cuda')
        x = torch.randn((sum(ks), mn), dtype=torch.bfloat16, device='cuda')
        x, fp32_sf = per_channel_cast_to_fp8(x, use_ue8m0=True)

        # Correctness
        packed_sf = get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(fp32_sf, ks_tensor, ks)
        split_packed_sf = packed_sf.split(packed_sf_ks)
        split_fp32_sf = fp32_sf.split(sf_ks)
        for i in range(num_groups):
            ref_packed_sf = get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl(split_fp32_sf[i].T).T
            assert torch.equal(split_packed_sf[i], ref_packed_sf), f'{i=}'

        # Performance
        t = bench_kineto(lambda: get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(fp32_sf, ks_tensor, ks), 'pack_fp32_into_ue8m0')
        print(f' > Perf ({num_groups=:3}, {mn=:5}, sum_k={sum(ks):5}):'
              f'{t * 1e6:4.0f} us | '
              f'{count_bytes(fp32_sf, packed_sf, ks_tensor) / 1e9 / t:4.0f} GB/s')
    print()


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(1)
    random.seed(1)

    test_sf_layout_kernels()
    test_k_grouped_sf_layout_kernels()
