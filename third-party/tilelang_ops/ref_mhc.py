import torch

import deep_gemm

from deep_gemm.utils import align, per_token_cast_to_fp8


_BASELINE_IMPORT_ERROR = None
try:
    from tile_kernels.modeling.mhc.ops import (
        MHCGemmWithSqrsum,
        mhc_post,
        mhc_pre_apply_mix,
        mhc_pre_big_fuse,
        mhc_pre_split_mixes,
        sinkhorn_normalize,
    )
    from tile_kernels.modeling.mhc.ops.norm_fn import MHCReducePartialsAndRmsnorm

    from .norm import norm

    def deep_gemm_hc_gemm_with_sqr_sum(
        x: torch.Tensor, fn: torch.Tensor, transpose_fn: bool, num_splits: int,
    ):
        assert not transpose_fn
        x_flat = x.view(-1, x.size(-2) * x.size(-1))
        output_shape = (num_splits, x_flat.size(0))
        gemm_out = torch.empty(
            (*output_shape, fn.size(0)), dtype=torch.float32, device=x.device)
        sqr_sum = torch.empty(output_shape, dtype=torch.float32, device=x.device)
        deep_gemm.tf32_hc_prenorm_gemm(
            x_flat, fn, gemm_out, sqr_sum, num_splits=num_splits)
        return gemm_out.unsqueeze(2), sqr_sum.unsqueeze(2)

    # Keep the fused Normal post-GEMM path, replacing only its GEMM+sqr-sum producer.
    MHCGemmWithSqrsum.apply = staticmethod(deep_gemm_hc_gemm_with_sqr_sum)
except Exception as ex:
    _BASELINE_IMPORT_ERROR = ex


NUM_SPLITS = 16
SHIFTED_KERNELS = (
    'mhc_post_fwd',
    'mhc_pre_apply_mix_fwd',
    'tf32_hc_prenorm_gemm',
    'mhc_reduce_partials_and_rmsnorm_fwd',
    'mhc_pre_split_mixes_fwd',
    'mhc_sinkhorn_fwd',
    '_norm_kernel',
)
NORMAL_KERNELS = (
    'mhc_post_fwd',
    'tf32_hc_prenorm_gemm',
    'mhc_pre_big_fuse_kernel',
    '_norm_kernel',
)


def has_baseline() -> bool:
    if _BASELINE_IMPORT_ERROR is None:
        return True
    print(f'Failed to load baseline code: {_BASELINE_IMPORT_ERROR}, skip baseline tests')
    return False


def extra_sf_rows(x: torch.Tensor, block_m: int) -> torch.Tensor:
    token_idx = torch.arange(x.size(0), device=x.device)
    index_in_block = token_idx % block_m
    return (
        token_idx // block_m * align(block_m, 128)
        + index_in_block // 128 * 128
        + index_in_block % 32 * 4
        + index_in_block % 128 // 32
    )


def _allocate_extra_sf(x: torch.Tensor, block_m: int) -> torch.Tensor:
    num_tokens, hidden = x.shape
    num_rows = (num_tokens + block_m - 1) // block_m * align(block_m, 128)
    return torch.empty_strided(
        (num_rows, hidden // 128), (1, num_rows), dtype=torch.int32, device=x.device)


def _norm_outputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    rmsnorm_scale: float,
    sf_layout: str,
    shared_sf_block_m: int,
) -> dict[str, torch.Tensor]:
    if sf_layout == 'bf16':
        return {'y_bf16': norm(x, weight, eps, rmsnorm_scale)}

    extra_storage = _allocate_extra_sf(x, shared_sf_block_m) if sf_layout == 'extra' else None
    y_bf16, y_fp8, y_fp8_sf = norm(
        x,
        weight,
        eps,
        rmsnorm_scale,
        sf_layout,
        extra_storage[:x.size(0)] if extra_storage is not None else None,
        shared_sf_block_m,
    )
    sf_name = 'y_gemm_sf' if sf_layout == 'col' else 'y_routed_sf'
    result = {'y_bf16': y_bf16, 'y_fp8': y_fp8, sf_name: y_fp8_sf}
    if extra_storage is not None:
        result['y_shared_sf_storage'] = extra_storage
    return result


@torch.no_grad()
def mhc_baseline(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    mix_scales: torch.Tensor,
    mix_bases: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    hc_mult: int,
    hc_norm_eps: float,
    hc_pre_eps: float,
    hc_post_scale: float,
    sinkhorn_eps: float,
    num_sinkhorn_iters: int,
    rmsnorm_eps: float,
    rmsnorm_scale: float,
    sf_layout: str = 'bf16',
    shared_sf_block_m: int = 0,
    shifted_prev_mix: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    new_residual = mhc_post(
        x.unsqueeze(0), residual.unsqueeze(0),
        post_mix.unsqueeze(0), comb_res_mix.unsqueeze(0),
    ).squeeze(0)

    if shifted_prev_mix is None:
        new_post_mix, new_comb_res_mix, norm_input = mhc_pre_big_fuse(
            new_residual, fn, mix_scales, mix_bases,
            hc_norm_eps, hc_pre_eps, sinkhorn_eps, hc_post_scale,
            num_sinkhorn_iters, n_splits=NUM_SPLITS,
        )
        result = {
            'new_residual': new_residual,
            'new_post_mix': new_post_mix,
            'new_comb_res_mix': new_comb_res_mix,
        }
    else:
        norm_input = mhc_pre_apply_mix(
            new_residual.unsqueeze(0), shifted_prev_mix.unsqueeze(0),
        ).squeeze(0)
        gemm_out, hc_sqr_sum = MHCGemmWithSqrsum.apply(
            new_residual.unsqueeze(0), fn, False, NUM_SPLITS,
        )
        mixes = MHCReducePartialsAndRmsnorm.apply(
            gemm_out, hc_sqr_sum, hc_norm_eps, hc_mult * x.size(1),
        )
        new_prev_mix, new_post_mix, new_comb_res_mix = mhc_pre_split_mixes(
            mixes.unsqueeze(0), mix_scales, mix_bases,
            hc_mult, hc_post_scale, hc_pre_eps,
        )
        new_comb_res_mix = sinkhorn_normalize(
            new_comb_res_mix, repeat=num_sinkhorn_iters, eps=sinkhorn_eps,
        )
        result = {
            'new_residual': new_residual,
            'new_prev_mix': new_prev_mix.squeeze(0),
            'new_post_mix': new_post_mix.squeeze(0),
            'new_comb_res_mix': new_comb_res_mix.squeeze(0),
        }

    result.update(_norm_outputs(
        norm_input, rmsnorm_weight, rmsnorm_eps, rmsnorm_scale,
        sf_layout, shared_sf_block_m,
    ))
    return result


def _sinkhorn_reference(value: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    value = value.softmax(dim=-1) + eps
    value = value / (value.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        value = value / (value.sum(dim=-1, keepdim=True) + eps)
        value = value / (value.sum(dim=-2, keepdim=True) + eps)
    return value


@torch.no_grad()
def mhc_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    mix_scales: torch.Tensor,
    mix_bases: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    hc_mult: int,
    hc_norm_eps: float,
    hc_pre_eps: float,
    hc_post_scale: float,
    sinkhorn_eps: float,
    num_sinkhorn_iters: int,
    rmsnorm_eps: float,
    rmsnorm_scale: float,
    sf_layout: str = 'bf16',
    shared_sf_block_m: int = 0,
    shifted_prev_mix: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    new_residual = x.float().unsqueeze(1) * post_mix
    new_residual += torch.einsum(
        'tij,tih->tjh', comb_res_mix, residual.float())
    new_residual = new_residual.bfloat16()
    norm_input = None if shifted_prev_mix is None else (
        new_residual.float() * shifted_prev_mix).sum(dim=1).bfloat16()

    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        mixes = new_residual.flatten(1).float() @ fn.mT
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
    hc_sqr_sum = new_residual.float().square().sum((1, 2))
    mixes *= torch.rsqrt(
        hc_sqr_sum / (hc_mult * x.size(1)) + hc_norm_eps).unsqueeze(1)
    scales = torch.cat((
        mix_scales[0].expand(hc_mult),
        mix_scales[1].expand(hc_mult),
        mix_scales[2].expand(hc_mult * hc_mult),
    ))
    mixes = mixes * scales + mix_bases
    new_prev_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(2) + hc_pre_eps
    new_post_mix = (
        mixes[:, hc_mult:2 * hc_mult].sigmoid() * hc_post_scale
    ).unsqueeze(2)
    new_comb_res_mix = _sinkhorn_reference(
        mixes[:, 2 * hc_mult:].view(-1, hc_mult, hc_mult),
        num_sinkhorn_iters,
        sinkhorn_eps,
    )
    if norm_input is None:
        norm_input = (new_residual.float() * new_prev_mix).sum(dim=1).bfloat16()

    norm_input_float = norm_input.float()
    y_bf16 = (
        norm_input_float
        * torch.rsqrt(norm_input_float.square().mean(1) + rmsnorm_eps).unsqueeze(1)
        * (rmsnorm_weight.float() * rmsnorm_scale)
    ).bfloat16()
    result = {
        'new_residual': new_residual,
        'new_post_mix': new_post_mix,
        'new_comb_res_mix': new_comb_res_mix,
        'y_bf16': y_bf16,
    }
    if shifted_prev_mix is not None:
        result['new_prev_mix'] = new_prev_mix
    if sf_layout == 'bf16':
        return result

    y_fp8, y_fp8_sf = per_token_cast_to_fp8(
        y_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    if sf_layout == 'col':
        col_major_sf = torch.empty_strided(
            y_fp8_sf.shape, (1, align(x.size(0), 4)),
            dtype=torch.int32, device=x.device)
        col_major_sf.copy_(y_fp8_sf)
        y_fp8_sf = col_major_sf
    sf_name = 'y_gemm_sf' if sf_layout == 'col' else 'y_routed_sf'
    result.update(y_fp8=y_fp8, **{sf_name: y_fp8_sf})
    if sf_layout == 'extra':
        storage = _allocate_extra_sf(x, shared_sf_block_m)
        storage[extra_sf_rows(x, shared_sf_block_m)] = y_fp8_sf
        result['y_shared_sf_storage'] = storage
    return result
