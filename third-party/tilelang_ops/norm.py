# Adapted from vLLM's TileLang norm kernel for the Mega mHC baseline.
import tilelang
import tilelang.language as T
import torch


FP8_AMAX_MARGIN = 1e-4
FLOAT8E4M3_MAX = 448.0


@T.macro
def ue8m0_exp_scale(scale: T.float32) -> T.int32:
    scale_bits = T.reinterpret("uint32", scale)
    exponent = (scale_bits >> 23) & 0xFF
    has_mantissa = scale_bits & ((1 << 23) - 1) != 0
    return -T.reinterpret("int32", exponent - 127 + has_mantissa)


@tilelang.jit(
    out_idx=None,
    target="cuda",
    pass_configs={"tl.disable_tma_lower": True},
)
def norm_tl(
    hidden: int,
    eps: float,
    out_scale: float,
    sf_layout: str,
    shared_sf_block_m: int,
):
    assert hidden in (4096, 7168)
    assert sf_layout in ("bf16", "col", "extra")

    do_fp8_cast = sf_layout != "bf16"
    num_tokens = T.dynamic("num_tokens")
    block_hidden = tilelang.next_power_of_2(hidden)
    threads = 256
    vec_size = 8 if hidden % 2048 == 0 else 4
    values_per_thread = block_hidden // threads

    def local_layout(i: int, j: int) -> tuple[int, int]:
        thread_idx = j % (threads * vec_size) // vec_size
        local_idx = i * values_per_thread + j // (threads * vec_size) * vec_size + j % vec_size
        return thread_idx, local_idx

    num_scales = hidden // 32
    block_num_scales = block_hidden // 32
    output_dtype = T.float8_e4m3fn if do_fp8_cast else "bfloat16"
    if sf_layout == "col":
        num_scale_packs = num_scales // 4
        aligned_tokens = (num_tokens + 3) // 4 * 4
        sf_type = "uint8"
        sf_shape = (num_tokens, num_scale_packs, 4)
        sf_stride = (4, aligned_tokens * 4, 1)
    elif sf_layout == "extra":
        sf_type = "uint8"
        sf_shape = (num_tokens, num_scales)
        sf_stride = (num_scales, 1)
    else:
        sf_type = "bfloat16"
        sf_shape = (num_tokens, hidden)
        sf_stride = (hidden, 1)

    if sf_layout == "extra":
        aligned_shared_block_m = (shared_sf_block_m + 127) // 128 * 128
        shared_sf_rows = T.dynamic("shared_sf_rows")
        shared_sf_type = "uint8"
        shared_sf_shape = (shared_sf_rows, hidden // 128, 4)
        shared_sf_stride = (4, shared_sf_rows * 4, 1)

        def shared_sf_row(token_idx):
            index_in_block = token_idx % shared_sf_block_m
            return (
                token_idx // shared_sf_block_m * aligned_shared_block_m
                + index_in_block // 128 * 128
                + index_in_block % 32 * 4
                + index_in_block % 128 // 32
            )
    else:
        shared_sf_type = "bfloat16"
        shared_sf_shape = (num_tokens, hidden)
        shared_sf_stride = (hidden, 1)

    @T.prim_func
    def _norm(
        x: T.Tensor((num_tokens, hidden), "bfloat16"),  # type: ignore
        y: T.Tensor((num_tokens, hidden), output_dtype),  # type: ignore
        weight: T.Tensor((hidden,), "bfloat16"),  # type: ignore
        sf: T.StridedTensor(sf_shape, sf_stride, sf_type),  # type: ignore
        shared_sf: T.StridedTensor(shared_sf_shape, shared_sf_stride, shared_sf_type),  # type: ignore
        y_bf16: T.Tensor((num_tokens, hidden), "bfloat16"),  # type: ignore
    ) -> None:
        with T.Kernel(num_tokens, threads=threads) as (token_idx):
            x_local = T.alloc_fragment((1, block_hidden), "float")
            y_local = T.alloc_fragment((1, block_hidden), "float")
            square_local = T.alloc_fragment((1, block_hidden), "float")
            rstd_local = T.alloc_fragment((1,), "float")
            weight_local = T.alloc_fragment((block_hidden,), "bfloat16")

            if do_fp8_cast:
                y_quant_local = T.alloc_fragment((1, block_hidden), "float")
                y_fp8_local = T.alloc_fragment((1, block_hidden), T.float8_e4m3fn)
                amax_local = T.alloc_fragment((1, block_num_scales), "float")
                scale_local = T.alloc_fragment((1, block_num_scales), "float")
                sf_local = T.alloc_fragment((1, block_num_scales), "uint8")
                T.annotate_layout({
                    x_local: T.Fragment(x_local.shape, forward_fn=local_layout),
                    y_quant_local: T.Fragment(y_quant_local.shape, forward_fn=local_layout),
                    y_fp8_local: T.Fragment(y_fp8_local.shape, forward_fn=local_layout),
                })
            else:
                T.annotate_layout({
                    x_local: T.Fragment(x_local.shape, forward_fn=local_layout),
                })

            T.copy(weight, weight_local[:hidden])
            T.pdl_sync()
            T.copy(x[token_idx, :], x_local[0, :hidden])
            for j in T.Parallel(block_hidden):
                square_local[0, j] = x_local[0, j] * x_local[0, j] if j < hidden else 0
            T.reduce_sum(square_local, rstd_local, dim=1)
            rstd_local[0] = T.rsqrt(rstd_local[0] / hidden + eps)
            for j in T.Parallel(block_hidden):
                if j < hidden:
                    y_local[0, j] = x_local[0, j] * rstd_local[0] * (weight_local[j] * out_scale)

            if not do_fp8_cast:
                T.pdl_trigger()
                T.copy(y_local[0, :hidden], y[token_idx, :])
            else:
                T.copy(y_local[0, :hidden], y_bf16[token_idx, :])
                for j in T.Parallel(block_hidden):
                    y_quant_local[0, j] = T.cast(T.cast(y_local[0, j], "bfloat16"), "float")
                y_quant_reshaped = T.reshape(y_quant_local, [1, block_num_scales, 32])
                T.reduce_absmax(y_quant_reshaped, amax_local, dim=2)

                for j in T.Parallel(block_num_scales):
                    amax_local[0, j] = T.max(amax_local[0, j], FP8_AMAX_MARGIN)
                    exponent = ue8m0_exp_scale(amax_local[0, j] / FLOAT8E4M3_MAX)
                    scale_local[0, j] = T.reinterpret("float32", (exponent + 127) << 23)
                    sf_local[0, j] = T.cast(-exponent + 127, "uint8")

                for j in T.Parallel(block_hidden):
                    y_fp8_local[0, j] = T.cast(
                        y_quant_local[0, j] * scale_local[0, j // 32], T.float8_e4m3fn)
                    y[token_idx, j] = y_fp8_local[0, j]

                T.pdl_trigger()
                if sf_layout == "col":
                    for j in T.Parallel(block_num_scales):
                        if j < num_scales:
                            sf[token_idx, j // 4, j % 4] = sf_local[0, j]
                else:
                    T.copy(sf_local[0, :num_scales], sf[token_idx, :])
                    for j in T.Parallel(block_num_scales):
                        if j < num_scales:
                            shared_sf[shared_sf_row(token_idx), j // 4, j % 4] = sf_local[0, j]

    return _norm


def norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out_scale: float,
    sf_layout: str = "bf16",
    shared_sf_out: torch.Tensor | None = None,
    shared_sf_block_m: int = 0,
):
    assert sf_layout in ("bf16", "col", "extra")
    assert x.ndim == 2 and x.dtype == torch.bfloat16 and x.is_contiguous()
    num_tokens, hidden = x.shape
    assert hidden in (4096, 7168)
    assert weight.shape == (hidden,) and weight.dtype == torch.bfloat16 and weight.is_contiguous()

    do_fp8_cast = sf_layout != "bf16"
    y_bf16 = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    if do_fp8_cast:
        y = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
        num_scale_packs = hidden // 128
        if sf_layout == "col":
            aligned_tokens = (num_tokens + 3) // 4 * 4
            sf_storage = torch.empty(
                num_scale_packs, aligned_tokens, dtype=torch.int32, device=x.device)
            sf = sf_storage.mT[:num_tokens]
            sf_arg = sf_storage.view(torch.uint8).as_strided(
                (num_tokens, num_scale_packs, 4), (4, aligned_tokens * 4, 1))
        else:
            sf = torch.empty(
                num_tokens, num_scale_packs, dtype=torch.int32, device=x.device)
            sf_arg = sf.view(torch.uint8)
    else:
        y = y_bf16
        sf = None
        sf_arg = x

    if sf_layout == "extra":
        assert shared_sf_out is not None and shared_sf_block_m > 0
        num_rows = shared_sf_out.stride(1)
        assert shared_sf_out.shape == (num_tokens, hidden // 128)
        assert shared_sf_out.dtype == torch.int32 and shared_sf_out.stride(0) == 1
        aligned_block_m = (shared_sf_block_m + 127) // 128 * 128
        assert (num_tokens + shared_sf_block_m - 1) // shared_sf_block_m * aligned_block_m <= num_rows
        shared_sf_arg = shared_sf_out.mT.view(torch.uint8).as_strided(
            (num_rows, hidden // 128, 4), (4, num_rows * 4, 1))
    else:
        assert shared_sf_out is None and shared_sf_block_m == 0
        shared_sf_arg = x

    kernel = norm_tl(hidden, eps, out_scale, sf_layout, shared_sf_block_m)
    kernel(x, y, weight, sf_arg, shared_sf_arg, y_bf16)
    return (y_bf16, y, sf) if do_fp8_cast else y
