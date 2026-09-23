import torch
import random
from deep_gemm.testing import bench_kineto, count_bytes, get_arch_major
from deep_gemm.utils import (
    align, ceil_div, ceil_to_ue8m0,
    per_token_cast_to_fp8, per_channel_cast_to_fp8,
    get_tma_aligned_size,
    get_mn_major_tma_aligned_tensor,
    get_mn_major_tma_aligned_packed_ue8m0_tensor,
    get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor,
)

from generators import (
    enumerate_sf_layout,
    enumerate_k_grouped_sf_layout,
    enumerate_k_grouped_psum_sf_layout,
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
    for mn, k, with_transpose, use_ue8m0, num_groups, gran_k in enumerate_sf_layout():
        x = torch.randn((num_groups * mn, k), dtype=torch.bfloat16, device='cuda')
        x, fp32_sf = per_token_cast_to_fp8(x, use_ue8m0=use_ue8m0, gran_k=gran_k)
        fp32_sf = fp32_sf if num_groups == 1 else fp32_sf.view(num_groups, mn, -1)
        fp32_sf = fp32_sf if with_transpose else fp32_sf.transpose(-1, -2).contiguous().transpose(-1, -2)

        # Correctness
        if use_ue8m0:
            impl, name = get_mn_major_tma_aligned_packed_ue8m0_tensor, 'pack_fp32_into_ue8m0'
            packed_sf = get_mn_major_tma_aligned_packed_ue8m0_tensor(fp32_sf)
            ref_packed_sf = get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl(fp32_sf)
            assert torch.equal(packed_sf, ref_packed_sf), f'{mn=}, {k=}, {with_transpose=}, {num_groups=}'
            assert packed_sf.shape == ref_packed_sf.shape
            assert all([packed_sf.stride(i) == ref_packed_sf.stride(i) for i in range(packed_sf.dim())])
        else:
            impl, name = get_mn_major_tma_aligned_tensor, 'transpose'
            transposed_sf = get_mn_major_tma_aligned_tensor(fp32_sf)
            tma_aligned_mn, sf_k = get_tma_aligned_size(mn, fp32_sf.element_size()), ceil_div(k, gran_k)
            if num_groups > 1:
                assert transposed_sf.size(0) == num_groups
                assert transposed_sf.stride(0) == tma_aligned_mn * sf_k
            assert transposed_sf.shape[-2:] == (mn, sf_k)
            assert transposed_sf.stride()[-2:] == (1, tma_aligned_mn)
            assert torch.equal(fp32_sf, transposed_sf)

        # Performance
        try:
            t = bench_kineto(lambda: impl(fp32_sf), name)
        except AssertionError as e:
            # Some cases may fallback to PyTorch impl
            t = 0
        print(f' > Perf ({num_groups=:2}, {mn=:5}, {k=:5}, transpose={int(with_transpose)}, use_ue8m0={int(use_ue8m0)}, gran_k={gran_k:3}): '
              f'{t * 1e6:4.0f} us | {count_bytes(fp32_sf, impl(fp32_sf)) / 1e9 / t if t else 0:4.0f} GB/s')
    print()


def test_k_grouped_sf_layout_kernels() -> None:
    print('Testing k-grouped SF layout kernels:')
    if get_arch_major() != 10:
        print(' > Skipped (packed k-grouped SF only supported on SM100)')
        return

    for mn, ks_cpu, num_groups, gran_k, k_alignment in enumerate_k_grouped_sf_layout():
        sf_ks = [k // gran_k for k in ks_cpu]
        packed_sf_ks = [ceil_div(sf_k, 4) for sf_k in sf_ks]
        grouped_layout = torch.tensor(ks_cpu, dtype=torch.int, device='cuda')
        x = torch.randn((sum(ks_cpu), mn), dtype=torch.bfloat16, device='cuda')
        _, fp32_sf = per_channel_cast_to_fp8(x, use_ue8m0=True, gran_k=gran_k)

        # Correctness: each group starts at a new packed row and independently
        # zero-fills unused slots in its final row.
        packed_sf = get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(fp32_sf, grouped_layout, ks_cpu, gran_k, k_alignment)
        assert packed_sf.shape == (sum(packed_sf_ks), mn)
        split_packed_sf = packed_sf.split(packed_sf_ks)
        split_fp32_sf = fp32_sf.split(sf_ks)
        for group_idx in range(num_groups):
            ref_packed_sf = get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl(split_fp32_sf[group_idx].T).T
            assert torch.equal(split_packed_sf[group_idx], ref_packed_sf), f'{group_idx=}'
            if sf_ks[group_idx] % 4:
                packed_bytes = split_packed_sf[group_idx][-1].view(torch.uint8).view(mn, 4)
                assert torch.count_nonzero(packed_bytes[:, sf_ks[group_idx] % 4:]) == 0

        # Performance
        t = bench_kineto(lambda: get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(fp32_sf, grouped_layout, ks_cpu, gran_k, k_alignment), 'pack_fp32_into_ue8m0')
        print(f' > Perf ({num_groups=:3}, {mn=:5}, sum_k={sum(ks_cpu):5}, '
              f'gran_k={gran_k:3}, k_alignment={k_alignment:3}): '
              f'{t * 1e6:4.0f} us | '
              f'{count_bytes(fp32_sf, packed_sf, grouped_layout) / 1e9 / t:4.0f} GB/s')
    print()


def test_k_grouped_psum_sf_layout_kernels() -> None:
    print('Testing k-grouped psum SF layout kernels:')
    if get_arch_major() != 10:
        print(' > Skipped (packed k-grouped SF only supported on SM100)')
        return

    for mn, real_ks_cpu, aligned_ks_cpu, psum_layout, num_groups, gran_k, k_alignment in enumerate_k_grouped_psum_sf_layout():
        grouped_layout = torch.tensor(psum_layout, dtype=torch.int, device='cuda')

        sf_ks = [align(k, k_alignment) // gran_k for k in real_ks_cpu]
        packed_sf_ks = [ceil_div(sf_k, 4) for sf_k in sf_ks]
        num_sf_rows = sum(sf_ks)
        fp32_sf = ceil_to_ue8m0(torch.rand((num_sf_rows, mn), dtype=torch.float, device='cuda') + 0.5)
        ref_packed_sf = torch.cat([
            get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl(group_sf.T).T
            for group_sf in fp32_sf.split(sf_ks) if group_sf.size(0) > 0
        ])

        # Psum layout provides raw K; SF covers each group's aligned K span and
        # packing independently pads every group to a packed row boundary.
        packed_sf = get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(fp32_sf, grouped_layout, aligned_ks_cpu, gran_k, k_alignment, use_psum_layout=True)
        assert torch.equal(packed_sf, ref_packed_sf)

        # Without synchronized aligned K sizes, allocation uses a safe upper bound.
        unsynced_packed_sf = get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
            fp32_sf, grouped_layout, None, gran_k, k_alignment, use_psum_layout=True)
        assert torch.equal(unsynced_packed_sf[:sum(packed_sf_ks)], ref_packed_sf)
        empty_ks_packed_sf = get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
            fp32_sf, grouped_layout, [], gran_k, k_alignment, use_psum_layout=True)
        assert torch.equal(empty_ks_packed_sf[:sum(packed_sf_ks)], ref_packed_sf)

        # Performance
        t = bench_kineto(lambda: get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(fp32_sf, grouped_layout, aligned_ks_cpu, gran_k, k_alignment, use_psum_layout=True), 'pack_fp32_into_ue8m0')
        print(f' > Perf ({num_groups=:3}, {mn=:5}, sum_k={sum(real_ks_cpu):5}, gran_k={gran_k:3}, k_alignment={k_alignment:3}):'
              f'{t * 1e6:4.0f} us | '
              f'{count_bytes(fp32_sf, packed_sf, grouped_layout) / 1e9 / t:4.0f} GB/s')
    print()


if __name__ == '__main__':
    torch.manual_seed(1)
    random.seed(1)

    test_sf_layout_kernels()
    test_k_grouped_sf_layout_kernels()
    test_k_grouped_psum_sf_layout_kernels()
