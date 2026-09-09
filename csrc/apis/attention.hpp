#pragma once

#include "../utils/compatibility.hpp"

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm120_fp8_fp4_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm100_mqa_logits.hpp"
#include "../jit_kernels/impls/sm100_mqa_logits_f16_weights.hpp"
#include "../jit_kernels/impls/sm90_fp8_mqa_logits.hpp"
#include "../jit_kernels/impls/sm120_mqa_logits.hpp"
#include "../jit_kernels/impls/smxx_clean_logits.hpp"
#endif

#include "layout.hpp"
#include <torch/library.h>
#include "../torch_library_utils.hpp"

namespace deep_gemm::attention {

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
static void fp8_gemm_nt_skip_head_mid(const std::pair<torch::Tensor, torch::Tensor>& a,
                                      const std::pair<torch::Tensor, torch::Tensor>& b,
                                      const torch::Tensor& d,
                                      const std::tuple<int, int, int>& head_splits,
                                      std::optional<std::tuple<int, int, int>> recipe,
                                      const std::string& compiled_dims,
                                      const bool& disable_ue8m0_cast) {
    // Shape must be `[M, K] @ [N, K].T`
    const auto major_a = get_major_type_ab(a.first);
    const auto major_b = get_major_type_ab(b.first);
    if (fp8_requires_k_major()) {
        DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);
        DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    }

    // D must be N-major
    check_major_type_cd(d);

    // Type and shape checks
    const auto [m , k ] = get_shape<2>(a.first);
    const auto [n , k_] = get_shape<2>(b.first);
    const auto [m_, n_] = get_shape<2>(d);
    DG_HOST_ASSERT(m == m_ and k == k_);
    DG_HOST_ASSERT(n > 0 and k > 0);
    DG_HOST_ASSERT(a.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.first.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);

    // Check head splits and N
    const auto [left, mid, right] = head_splits;
    DG_HOST_ASSERT(n % (left + right) == 0 and n_ == n + n / (left + right) * mid);

    // Do nothing if the problem is empty
    if (m == 0)
        return;

    // Transform SFA and SFB into compute-required layout
    const auto [sfa, sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        a.second, b.second, m, n, k, recipe, std::nullopt, std::nullopt,
        std::nullopt, std::nullopt, disable_ue8m0_cast);
    DG_HOST_ASSERT(gran_k_a == 128 and gran_k_b == 128);

    // Dispatch into different implements
    const auto arch_major = device_runtime->get_arch_major();
    const auto epilogue_type = fmt::format("epilogue::transform::EpilogueHeadSplits<{}, {}, {}>", left, mid, right);
    if (arch_major == 9 and sfa.scalar_type() == torch::kFloat and std::get<1>(recipe.value()) != 1) {
        const auto major_sfb = get_major_type_ab(sfb);
        sm90_fp8_gemm_1d2d(a.first, sfa, b.first, sfb, std::nullopt, d, m, n, k, major_a, major_b, major_sfb, compiled_dims, epilogue_type);
    } else if (arch_major == 10 and sfa.scalar_type() == torch::kInt) {
        // NOTES: Only granularity 128 and FP8 are exposed in the API
        sm100_fp8_fp4_gemm_1d1d(a.first, sfa, b.first, sfb, std::nullopt, d, m, n, k,
                                128, 128, major_a, major_b, compiled_dims, epilogue_type);
    } else if (arch_major == 12 and sfa.scalar_type() == torch::kInt) {
        sm120_fp8_fp4_gemm_1d1d(a.first, sfa, b.first, sfb, std::nullopt, d, m, n, k,
                                128, 128, major_a, major_b, compiled_dims, epilogue_type);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture or scaling factor types");
    }
}

static torch::Tensor fp8_fp4_mqa_logits(const std::tuple<torch::Tensor, std::optional<torch::Tensor>>& q,
                                        const std::tuple<torch::Tensor, torch::Tensor>& kv,
                                        const torch::Tensor& weights,
                                        const torch::Tensor& cu_seq_len_k_start,
                                        const torch::Tensor& cu_seq_len_k_end,
                                        const bool& clean_logits,
                                        const int& max_seqlen_k,
                                        const at::ScalarType& logits_dtype) {
    const auto [q_fp, q_sf] = q;
    const auto [kv_fp, kv_sf] = kv;
    const auto qk_dtype = q_fp.scalar_type();
    const bool is_fp4 = qk_dtype == kPackedFP4;
    const bool is_mx_sf = q_sf.has_value();
    DG_HOST_ASSERT(not is_fp4 or is_mx_sf);
    int seq_len, seq_len_kv, num_heads, head_dim;

    const auto arch_major = device_runtime->get_arch_major();
    // Check Q
    std::tie(seq_len, num_heads, head_dim) = get_logical_shape<3>(q_fp);
    DG_HOST_ASSERT((not is_fp4 and head_dim == 32) or head_dim == 64 or head_dim == 128);
    DG_HOST_ASSERT(q_fp.is_contiguous());
    DG_HOST_ASSERT(q_fp.scalar_type() == (is_fp4 ? kPackedFP4 : torch::kFloat8_e4m3fn));

    // Check SF Q
    if (is_mx_sf) {
        DG_HOST_ASSERT(arch_major == 10 or (arch_major == 12 and is_fp4));
        DG_HOST_ASSERT(q_sf.has_value());
        auto [_seq_len, _num_heads] = get_shape<2>(q_sf.value());
        DG_HOST_ASSERT(seq_len == _seq_len and num_heads == _num_heads);
        DG_HOST_ASSERT(q_sf.value().is_contiguous());
        DG_HOST_ASSERT(q_sf.value().scalar_type() == torch::kInt32);
    }

    // Check KV
    int _head_dim;
    std::tie(seq_len_kv, _head_dim) = get_logical_shape<2>(kv_fp);
    DG_HOST_ASSERT(head_dim == _head_dim);
    DG_HOST_ASSERT(kv_fp.is_contiguous());
    DG_HOST_ASSERT(kv_fp.scalar_type() == (is_fp4 ? kPackedFP4 : torch::kFloat8_e4m3fn));

    // Check SF KV
    auto [_seq_len_kv] = get_shape<1>(kv_sf);
    DG_HOST_ASSERT(seq_len_kv == _seq_len_kv);
    DG_HOST_ASSERT(kv_sf.is_contiguous());
    DG_HOST_ASSERT(kv_sf.scalar_type() == (is_mx_sf ? torch::kInt32 : torch::kFloat));

    // Check weights
    auto [_seq_len, _num_heads] = get_shape<2>(weights);
    DG_HOST_ASSERT(seq_len == _seq_len and num_heads == _num_heads);
    DG_HOST_ASSERT(weights.stride(1) == 1);
    // The `weights` dtype explicitly selects the MQA-logits accumulation precision:
    //   - FP16 selects nv_dev's SM100 two-CTA FP16-accumulator kernel.
    //   - BF16 selects #377's SM100 unified BF16 reduction.
    //   - FP32 is supported by every architecture.
    const auto weights_dtype = weights.scalar_type();
    const bool weights_is_f16 = weights_dtype == torch::kFloat16;
    const bool weights_is_bf16 = weights_dtype == torch::kBFloat16;
    DG_HOST_ASSERT(weights_dtype == torch::kFloat or
                   (arch_major == 10 and (weights_is_f16 or weights_is_bf16)));
    DG_HOST_ASSERT(not weights_is_bf16 or logits_dtype == torch::kBFloat16);
    DG_HOST_ASSERT(not weights_is_f16 or
                   (not is_mx_sf and qk_dtype == torch::kFloat8_e4m3fn));

    // Check cu_seq_len_k_start
    DG_HOST_ASSERT(cu_seq_len_k_start.size(0) == seq_len);
    DG_HOST_ASSERT(cu_seq_len_k_start.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_start.scalar_type() == torch::kInt);

    // Check cu_seq_len_k_end
    DG_HOST_ASSERT(cu_seq_len_k_end.size(0) == seq_len);
    DG_HOST_ASSERT(cu_seq_len_k_end.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_end.scalar_type() == torch::kInt);

    // Allocate output
    constexpr int block_qh = 128;
    const int block_kv = (device_runtime->get_arch_major() == 12) ? 128 : 256;
    const int block_q = block_qh / num_heads;
    DG_HOST_ASSERT(block_qh % num_heads == 0);

    torch::Tensor logits;
    int aligned_seq_len = align(seq_len, block_q), stride_logits;
    if (weights_is_f16) {
        // FP16 weights select the SM100 2-CTA FP16 kernel, which tiles the query dimension
        // in `block_q * 2` chunks (no per-row bound check), so the output is padded accordingly
        DG_HOST_ASSERT(arch_major == 10 and "FP16 weights MQA logits requires SM100 (arch 10)");
        DG_HOST_ASSERT(seq_len % 4 == 0 and "FP16 weights MQA logits requires seq_len % 4 == 0");
        aligned_seq_len = align(seq_len, block_q * 2);
    }
    // Logits row stride must be 1024-byte aligned
    const int stride_logits_alignment = 1024 / static_cast<int>(c10::elementSize(logits_dtype));
    if (max_seqlen_k == 0) {
        stride_logits = align(seq_len_kv + block_kv, stride_logits_alignment);
        logits = torch::empty({aligned_seq_len, stride_logits}, q_fp.options().dtype(logits_dtype));
        logits = logits.index({torch::indexing::Slice(0, seq_len), torch::indexing::Slice(0, seq_len_kv)});
    } else {
        stride_logits = align(align(max_seqlen_k, block_kv), stride_logits_alignment);
        logits = torch::empty({aligned_seq_len, stride_logits}, q_fp.options().dtype(logits_dtype));
        logits = logits.index({torch::indexing::Slice(0, seq_len), torch::indexing::Slice(0, max_seqlen_k)});
        DG_HOST_ASSERT(not clean_logits);
    }

    // Dispatch implementation
    if (weights_is_f16) {
        // FP16 weights -> FP16 MMA accumulator (Q*K score + per-head reduction in FP16); see note above
        sm100_mqa_logits_f16_weights(
            q_fp, kv_fp, kv_sf, weights, cu_seq_len_k_start, cu_seq_len_k_end,
            logits, logits_dtype, seq_len, seq_len_kv, max_seqlen_k,
            stride_logits, num_heads, head_dim, block_q, block_kv);
    } else if (arch_major == 10) {
        DG_HOST_ASSERT(qk_dtype == torch::kFloat8_e4m3fn or qk_dtype == kPackedFP4);
        DG_HOST_ASSERT(num_heads == 8 or num_heads == 16 or num_heads == 32 or num_heads == 64);
        sm100_mqa_logits(q_fp, q_sf, kv_fp, kv_sf, weights, cu_seq_len_k_start, cu_seq_len_k_end, logits, logits_dtype,
                         seq_len, seq_len_kv, max_seqlen_k, stride_logits, num_heads, head_dim, block_q, block_kv,
                         is_mx_sf, qk_dtype);
    } else if (arch_major == 9) {
        DG_HOST_ASSERT(not is_mx_sf);
        DG_HOST_ASSERT(qk_dtype == torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(num_heads == 32 or num_heads == 64);
        DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat);
        sm90_fp8_mqa_logits(q_fp, kv_fp, kv_sf, weights, cu_seq_len_k_start, cu_seq_len_k_end, logits, logits_dtype,
                            seq_len, seq_len_kv, max_seqlen_k, stride_logits, num_heads, head_dim, block_q, block_kv);
    } else if (arch_major == 12) {
        DG_HOST_ASSERT(qk_dtype == torch::kFloat8_e4m3fn or qk_dtype == kPackedFP4);
        DG_HOST_ASSERT(num_heads == 16 or num_heads == 32 or num_heads == 64);
        DG_HOST_ASSERT(weights_dtype == torch::kFloat);
        sm120_mqa_logits(
            q_fp, q_sf, kv_fp, kv_sf, weights,
            cu_seq_len_k_start, cu_seq_len_k_end, logits, logits_dtype,
            seq_len, seq_len_kv, max_seqlen_k, stride_logits,
            num_heads, head_dim, block_q, block_kv, is_mx_sf, qk_dtype);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Clean unfilled logits
    if (clean_logits)
        smxx_clean_logits(logits, cu_seq_len_k_start, cu_seq_len_k_end, 1, seq_len, seq_len_kv, stride_logits);
    return logits;
}

static torch::Tensor get_paged_mqa_logits_metadata(const torch::Tensor& context_lens, int block_kv, int num_sms, const std::optional<torch::Tensor>& indices) {
    // NOTES: Only 2D context lens is supported for now
    DG_HOST_ASSERT(context_lens.dim() == 2);
    const bool is_context_lens_2d = true;
    const int batch_size = context_lens.size(0);
    const int next_n = context_lens.size(1);
    const bool is_varlen = indices.has_value();
    DG_HOST_ASSERT(context_lens.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(context_lens.is_contiguous());

    // Create metadata tensor. `num_sms` here is actually the scheduler slot count
    // (= num_clusters on SM90 next_n=4 multicast, = num_sms elsewhere); callers
    // pre-divide.
    auto schedule_metadata = torch::empty({num_sms + 1, 2}, context_lens.options());

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    const int* indices_ptr = nullptr;
    if (is_varlen) {
        const auto& indices_tensor = indices.value();
        DG_HOST_ASSERT((arch_major == 10 or arch_major == 12) and next_n == 1);
        DG_HOST_ASSERT(indices_tensor.dim() == 1 and indices_tensor.size(0) == batch_size);
        DG_HOST_ASSERT(indices_tensor.is_contiguous());
        DG_HOST_ASSERT(indices_tensor.scalar_type() == torch::kInt);
        indices_ptr = indices_tensor.data_ptr<int>();
    }

    if (arch_major == 10) {
        DG_HOST_ASSERT(block_kv == 32 or block_kv == 64 or block_kv == 128);
        sm100_paged_mqa_logits_metadata(
            context_lens, schedule_metadata, batch_size, batch_size * next_n,
            next_n, num_sms, is_context_lens_2d, is_varlen, indices_ptr);
    } else if (arch_major == 9) {
        DG_HOST_ASSERT(not is_varlen and (block_kv == 32 or block_kv == 64));
        // Metadata schedules 64-row compute tiles even when pages contain 32 rows.
        sm90_paged_mqa_logits_metadata(
            context_lens, schedule_metadata, batch_size, next_n,
            /*block_kv=*/64, num_sms, is_context_lens_2d,
            /*num_next_n_atoms=*/1, /*is_varlen=*/false, /*indices=*/nullptr);
    } else if (arch_major == 12) {
        DG_HOST_ASSERT(block_kv == 32 or block_kv == 64);
        const int next_n_atom = (is_varlen or next_n >= 2) ? 2 : 1;
        const int num_next_n_atoms = (next_n + next_n_atom - 1) / next_n_atom;
        sm120_paged_mqa_logits_metadata(
            context_lens, schedule_metadata, batch_size, next_n, block_kv,
            num_sms, is_context_lens_2d, num_next_n_atoms,
            is_varlen, indices_ptr);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    return schedule_metadata;
}

static torch::Tensor fp8_fp4_paged_mqa_logits(const std::tuple<torch::Tensor, std::optional<torch::Tensor>>& q,
                                              const torch::Tensor& fused_kv_cache,
                                              const torch::Tensor& weights,
                                              const torch::Tensor& context_lens,
                                              const torch::Tensor& block_table,
                                              const torch::Tensor& schedule_meta,
                                              const int& max_context_len,
                                              const bool& clean_logits,
                                              const at::ScalarType& logits_dtype,
                                              const std::optional<torch::Tensor>& indices) {
    const auto [q_fp, q_sf] = q;
    const auto qk_dtype = q_fp.scalar_type();
    const bool is_fp4 = qk_dtype == kPackedFP4;
    const bool is_mx_sf = q_sf.has_value();
    DG_HOST_ASSERT(not is_fp4 or is_mx_sf);

    torch::Tensor kv_cache, kv_cache_sf;
    int batch_size, next_n, num_heads, head_dim;
    int num_kv_blocks, block_kv;
    int kv_cache_stride_bytes;
    int block_table_stride = block_table.stride(0);
    int num_sms = device_runtime->get_num_sms();
    const auto arch_major = device_runtime->get_arch_major();

    // Check Q
    std::tie(batch_size, next_n, num_heads, head_dim) = get_logical_shape<4>(q_fp);
    DG_HOST_ASSERT((not is_fp4 and head_dim == 32) or head_dim == 64 or head_dim == 128);
    DG_HOST_ASSERT(q_fp.is_contiguous());
    DG_HOST_ASSERT(q_fp.scalar_type() == (is_fp4 ? kPackedFP4 : torch::kFloat8_e4m3fn));
    DG_HOST_ASSERT(next_n >= 1);

    // Check SF Q
    if (is_mx_sf) {
        DG_HOST_ASSERT(arch_major == 10 or (arch_major == 12 and is_fp4));
        DG_HOST_ASSERT(q_sf.has_value());
        auto [_batch_size, _next_n, _num_heads] = get_shape<3>(q_sf.value());
        DG_HOST_ASSERT(batch_size == _batch_size and next_n == _next_n and num_heads == _num_heads);
        DG_HOST_ASSERT(q_sf.value().is_contiguous());
        DG_HOST_ASSERT(q_sf.value().scalar_type() == torch::kInt32);
    }

    // Check fused KV cache
    int num_heads_kv, head_dim_with_sf;
    std::tie(num_kv_blocks, block_kv, num_heads_kv, head_dim_with_sf) = get_shape<4>(fused_kv_cache);
    DG_HOST_ASSERT(
        (arch_major == 10 and (block_kv == 32 or block_kv == 64 or block_kv == 128)) or
        (arch_major == 9 and (block_kv == 32 or block_kv == 64)) or
        (arch_major == 12 and ((is_fp4 and (block_kv == 32 or block_kv == 64)) or
                               (not is_fp4 and block_kv == 64))));
    const int kv_head_dim = is_fp4 ? head_dim / 2 : head_dim;
    const int sf_bytes = static_cast<int>(is_mx_sf ? sizeof(int) : sizeof(float));
    DG_HOST_ASSERT(num_heads_kv == 1 and head_dim_with_sf == kv_head_dim + sf_bytes);
    DG_HOST_ASSERT(fused_kv_cache.stride(1) == head_dim_with_sf and fused_kv_cache.stride(3) == 1);
    DG_HOST_ASSERT(fused_kv_cache.scalar_type() == torch::kByte);

    // Derive KV values and SF tensor
    kv_cache_stride_bytes = fused_kv_cache.stride(0);
    DG_HOST_ASSERT(kv_cache_stride_bytes % sf_bytes == 0);
    kv_cache = torch::from_blob(
        fused_kv_cache.data_ptr(),
        {num_kv_blocks, block_kv, kv_head_dim},
        {kv_cache_stride_bytes, kv_head_dim, 1},
        torch::TensorOptions().dtype(is_fp4 ? kPackedFP4 : torch::kFloat8_e4m3fn)
    );
    kv_cache_sf = torch::from_blob(
        fused_kv_cache.data_ptr<uint8_t>() + block_kv * kv_head_dim,
        {num_kv_blocks, block_kv},
        {kv_cache_stride_bytes / sf_bytes, 1},
        torch::TensorOptions().dtype(is_mx_sf ? torch::kInt32 : torch::kFloat32)
    );

    // Check weights
    auto [_batch_size_next_n, _num_heads] = get_shape<2>(weights);
    DG_HOST_ASSERT(_batch_size_next_n == batch_size * next_n and _num_heads == num_heads);
    DG_HOST_ASSERT(weights.stride(1) == 1);
    DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat or (arch_major == 10 and weights.scalar_type() == torch::kBFloat16));
    DG_HOST_ASSERT(weights.scalar_type() != torch::kBFloat16 or logits_dtype == torch::kBFloat16);

    // Check block table
    auto [_batch_size, _max_block_len] = get_shape<2>(block_table);
    DG_HOST_ASSERT(_batch_size == batch_size);
    DG_HOST_ASSERT(block_table.stride(1) == 1);
    DG_HOST_ASSERT(block_table.scalar_type() == torch::kInt);

    // Check indices
    const bool is_varlen = indices.has_value();
    const auto indices_tensor = indices.value_or(torch::Tensor());
    if (is_varlen) {
        DG_HOST_ASSERT((arch_major == 10 or arch_major == 12) and next_n == 1);
        DG_HOST_ASSERT(indices_tensor.dim() == 1 and indices_tensor.size(0) == batch_size);
        DG_HOST_ASSERT(indices_tensor.is_contiguous());
        DG_HOST_ASSERT(indices_tensor.scalar_type() == torch::kInt);
    }

    // Check schedule metadata. SM90 next_n=4 uses a 2-CTA cluster per task, so
    // schedule_meta carries one entry per cluster instead of per SM.
    auto [_schedule_meta_size, _meta_info_size] = get_shape<2>(schedule_meta);
    const int num_kv_multicast = (arch_major == 9 and next_n == 4) ? 2 : 1;
    DG_HOST_ASSERT(_schedule_meta_size == num_sms / num_kv_multicast + 1 and _meta_info_size == 2);
    DG_HOST_ASSERT(schedule_meta.is_contiguous());
    DG_HOST_ASSERT(schedule_meta.scalar_type() == torch::kInt);

    // Check context lengths
    // NOTES: Only 2D context lens is supported for now
    DG_HOST_ASSERT(context_lens.dim() == 2);
    const bool is_context_lens_2d = true;
    const auto [__batch_size, _next_n] = get_shape<2>(context_lens);
    DG_HOST_ASSERT(batch_size == __batch_size and next_n == _next_n);
    DG_HOST_ASSERT(context_lens.is_contiguous());
    DG_HOST_ASSERT(context_lens.scalar_type() == torch::kInt);

    // Allocate output
    // SM120a: 2 groups × 64 KV rows = 128; SM90/100: 256
    const int split_kv = (arch_major == 12) ? 128 : 256;
    DG_HOST_ASSERT(logits_dtype == torch::kFloat32 or logits_dtype == torch::kBFloat16);
    // Logits row stride must be 1024-byte aligned
    const int stride_logits_alignment = 1024 / static_cast<int>(c10::elementSize(logits_dtype));
    const auto aligned_max_context_len = align(align(max_context_len, split_kv), stride_logits_alignment);
    auto logits = torch::empty({batch_size * next_n, aligned_max_context_len}, q_fp.options().dtype(logits_dtype));
    logits = logits.slice(-1, 0, max_context_len);

    // Dispatch implementation
    if (arch_major == 10) {
        DG_HOST_ASSERT(qk_dtype == torch::kFloat8_e4m3fn or qk_dtype == kPackedFP4);
        DG_HOST_ASSERT(num_heads == 8 or num_heads == 16 or num_heads == 32 or num_heads == 64);
        constexpr int splits_per_chunk = 16;
        sm100_paged_mqa_logits(q_fp, q_sf, kv_cache, kv_cache_sf, weights, context_lens, logits, block_table, indices_tensor, schedule_meta,
                               logits_dtype, batch_size, batch_size * next_n, next_n, num_heads, head_dim, num_kv_blocks, block_kv, is_context_lens_2d,
                               is_varlen, aligned_max_context_len, block_table_stride, num_sms, split_kv, splits_per_chunk,
                               is_mx_sf, qk_dtype);
    } else if (arch_major == 9) {
        DG_HOST_ASSERT(not is_mx_sf);
        DG_HOST_ASSERT(qk_dtype == torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(num_heads == 32 or num_heads == 64);
        DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat);
        sm90_fp8_paged_mqa_logits(q_fp, kv_cache, kv_cache_sf, weights, context_lens, logits, block_table, indices_tensor, schedule_meta,
                                  logits_dtype, batch_size, next_n, num_heads, head_dim, num_kv_blocks, block_kv, is_context_lens_2d,
                                  is_varlen, aligned_max_context_len, block_table_stride, num_sms, split_kv);
    } else if (arch_major == 12) {
        DG_HOST_ASSERT(qk_dtype == torch::kFloat8_e4m3fn or qk_dtype == kPackedFP4);
        DG_HOST_ASSERT(num_heads == 16 or num_heads == 32 or num_heads == 64);
        DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat);
        sm120_paged_mqa_logits(
            q_fp, q_sf, kv_cache, kv_cache_sf, weights, context_lens,
            logits, block_table, indices_tensor, schedule_meta, logits_dtype,
            batch_size, batch_size * next_n, next_n, num_heads, head_dim,
            num_kv_blocks, block_kv, is_context_lens_2d, is_varlen,
            aligned_max_context_len, block_table_stride, num_sms, split_kv,
            is_mx_sf, qk_dtype);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Clean unfilled logits
    if (clean_logits) {
        DG_HOST_ASSERT(not is_context_lens_2d);
        smxx_clean_logits(logits, std::nullopt, context_lens, next_n, batch_size * next_n, max_context_len, aligned_max_context_len);
    }
    return logits;
}


// Legacy API wrappers
static torch::Tensor fp8_mqa_logits(const torch::Tensor& q,
                                    const std::tuple<torch::Tensor, torch::Tensor>& kv,
                                    const torch::Tensor& weights,
                                    const torch::Tensor& cu_seq_len_k_start,
                                    const torch::Tensor& cu_seq_len_k_end,
                                    const bool& clean_logits,
                                    const int& max_seqlen_k) {
    return fp8_fp4_mqa_logits(std::make_tuple(q, std::nullopt), kv, weights,
                              cu_seq_len_k_start, cu_seq_len_k_end,
                              clean_logits, max_seqlen_k, torch::kFloat);
}

static torch::Tensor fp8_paged_mqa_logits(const torch::Tensor& q,
                                          const torch::Tensor& fused_kv_cache,
                                          const torch::Tensor& weights,
                                          const torch::Tensor& context_lens,
                                          const torch::Tensor& block_table,
                                          const torch::Tensor& schedule_meta,
                                          const int& max_context_len,
                                          const bool& clean_logits,
                                          const std::optional<torch::Tensor>& indices) {
    return fp8_fp4_paged_mqa_logits(std::make_tuple(q, std::nullopt), fused_kv_cache, weights,
                                    context_lens, block_table, schedule_meta,
                                    max_context_len, clean_logits, torch::kFloat, indices);
}
#endif

}  // namespace deep_gemm::attention

namespace deep_gemm::torch_registration {

using namespace deep_gemm::torch_utils;

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
static void fp8_gemm_nt_skip_head_mid(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d,
    const std::vector<int64_t>& head_splits,
    const c10::optional<std::vector<int64_t>>& recipe,
    const std::string& compiled_dims,
    const bool& disable_ue8m0_cast) {
    attention::fp8_gemm_nt_skip_head_mid(
        {a, sfa}, {b, sfb}, d,
        list_to_tuple3(head_splits),
        list_to_recipe3(recipe),
        compiled_dims, disable_ue8m0_cast);
}

static torch::Tensor fp8_fp4_mqa_logits(
    const torch::Tensor& q, const c10::optional<torch::Tensor>& q_sf,
    const torch::Tensor& kv, const torch::Tensor& kv_sf,
    const torch::Tensor& weights,
    const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end,
    const bool& clean_logits,
    const int64_t& max_seqlen_k,
    at::ScalarType logits_dtype) {
    return attention::fp8_fp4_mqa_logits(
        std::make_tuple(q, q_sf),
        std::make_tuple(kv, kv_sf),
        weights, cu_seq_len_k_start, cu_seq_len_k_end,
        clean_logits, static_cast<int>(max_seqlen_k),
        logits_dtype);
}

static torch::Tensor get_paged_mqa_logits_metadata(
    const torch::Tensor& context_lens, const int64_t& block_kv,
    const int64_t& num_sms, const c10::optional<torch::Tensor>& indices) {
    return attention::get_paged_mqa_logits_metadata(
        context_lens, static_cast<int>(block_kv),
        static_cast<int>(num_sms), indices);
}

static torch::Tensor fp8_fp4_paged_mqa_logits(
    const torch::Tensor& q, const c10::optional<torch::Tensor>& q_sf,
    const torch::Tensor& kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    const torch::Tensor& schedule_meta,
    const int64_t& max_context_len,
    const bool& clean_logits,
    at::ScalarType logits_dtype,
    const c10::optional<torch::Tensor>& indices) {
    return attention::fp8_fp4_paged_mqa_logits(
        std::make_tuple(q, q_sf),
        kv_cache, weights, context_lens, block_table, schedule_meta,
        static_cast<int>(max_context_len), clean_logits,
        logits_dtype, indices);
}

static torch::Tensor fp8_mqa_logits(
    const torch::Tensor& q,
    const torch::Tensor& kv, const torch::Tensor& kv_sf,
    const torch::Tensor& weights,
    const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end,
    const bool& clean_logits,
    const int64_t& max_seqlen_k) {
    return attention::fp8_mqa_logits(
        q, std::make_tuple(kv, kv_sf), weights,
        cu_seq_len_k_start, cu_seq_len_k_end,
        clean_logits, static_cast<int>(max_seqlen_k));
}

static torch::Tensor fp8_paged_mqa_logits(
    const torch::Tensor& q,
    const torch::Tensor& kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& context_lens,
    const torch::Tensor& block_table,
    const torch::Tensor& schedule_meta,
    const int64_t& max_context_len,
    const bool& clean_logits,
    const c10::optional<torch::Tensor>& indices) {
    return attention::fp8_paged_mqa_logits(
        q, kv_cache, weights,
        context_lens, block_table, schedule_meta,
        static_cast<int>(max_context_len), clean_logits, indices);
}
#endif

}  // namespace deep_gemm::torch_registration

TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
    m.def(
        "fp8_gemm_nt_skip_head_mid(Tensor a, Tensor sfa, Tensor b, Tensor sfb, Tensor(d!) d, int[3] head_splits, int[3]? recipe=None, str compiled_dims='nk', bool disable_ue8m0_cast=False) -> ()");
    m.def(
        "fp8_fp4_mqa_logits(Tensor q, Tensor? q_sf, Tensor kv, Tensor kv_sf, Tensor weights, Tensor cu_seq_len_k_start, Tensor cu_seq_len_k_end, bool clean_logits=True, int max_seqlen_k=0, ScalarType logits_dtype=float) -> Tensor");
    m.def(
        "get_paged_mqa_logits_metadata(Tensor context_lens, int block_kv, int num_sms, Tensor? indices=None) -> Tensor");
    m.def(
        "fp8_fp4_paged_mqa_logits(Tensor q, Tensor? q_sf, Tensor kv_cache, Tensor weights, Tensor context_lens, Tensor block_table, Tensor schedule_meta, int max_context_len, bool clean_logits=False, ScalarType logits_dtype=float, Tensor? indices=None) -> Tensor");
    m.def(
        "fp8_mqa_logits(Tensor q, Tensor kv, Tensor kv_sf, Tensor weights, Tensor cu_seq_len_k_start, Tensor cu_seq_len_k_end, bool clean_logits=True, int max_seqlen_k=0) -> Tensor");
    m.def(
        "fp8_paged_mqa_logits(Tensor q, Tensor kv_cache, Tensor weights, Tensor context_lens, Tensor block_table, Tensor schedule_meta, int max_context_len, bool clean_logits=False, Tensor? indices=None) -> Tensor");
#endif
}

TORCH_LIBRARY_IMPL(deep_gemm, CUDA, m) {
    using namespace deep_gemm::torch_registration;

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
    m.impl("fp8_gemm_nt_skip_head_mid", TORCH_FN(fp8_gemm_nt_skip_head_mid));
    m.impl("fp8_fp4_mqa_logits", TORCH_FN(fp8_fp4_mqa_logits));
    m.impl("get_paged_mqa_logits_metadata", TORCH_FN(get_paged_mqa_logits_metadata));
    m.impl("fp8_fp4_paged_mqa_logits", TORCH_FN(fp8_fp4_paged_mqa_logits));
    m.impl("fp8_mqa_logits", TORCH_FN(fp8_mqa_logits));
    m.impl("fp8_paged_mqa_logits", TORCH_FN(fp8_paged_mqa_logits));
#endif
}
