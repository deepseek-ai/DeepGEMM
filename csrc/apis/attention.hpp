#pragma once

#include "../utils/compatibility.hpp"

#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp"
#include "../jit_kernels/impls/smxx_fp8_fp4_mqa_logits.hpp"
#include "../jit_kernels/impls/smxx_fp8_fp4_paged_mqa_logits.hpp"
#include "../jit_kernels/impls/smxx_clean_logits.hpp"
#endif

#include "layout.hpp"

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
    const bool is_fp4 = q_sf.has_value();
    int seq_len, seq_len_kv, num_heads, head_dim;

    if (is_fp4) {
        // Check FP4 Q
        std::tie(seq_len, num_heads, head_dim) = get_shape<3>(q_fp);
        head_dim *= 2;
        DG_HOST_ASSERT(num_heads == 32 or num_heads == 64);
        DG_HOST_ASSERT(head_dim == 128);
        DG_HOST_ASSERT(q_fp.is_contiguous());
        DG_HOST_ASSERT(q_fp.scalar_type() == kPackedFP4);

        // Check SF Q
        auto [_seq_len, _num_heads] = get_shape<2>(q_sf.value());
        DG_HOST_ASSERT(seq_len == _seq_len and num_heads == _num_heads);
        DG_HOST_ASSERT(q_sf.value().is_contiguous());
        DG_HOST_ASSERT(q_sf.value().scalar_type() == torch::kInt32);

        // Check FP4 KV
        int _head_dim;
        std::tie(seq_len_kv, _head_dim) = get_shape<2>(kv_fp);
        _head_dim *= 2;
        DG_HOST_ASSERT(head_dim == _head_dim);
        DG_HOST_ASSERT(kv_fp.is_contiguous());
        DG_HOST_ASSERT(kv_fp.scalar_type() == kPackedFP4);

        // Check SF KV
        auto [_seq_len_kv] = get_shape<1>(kv_sf);
        DG_HOST_ASSERT(seq_len_kv == _seq_len_kv);
        DG_HOST_ASSERT(kv_sf.is_contiguous());
        DG_HOST_ASSERT(kv_sf.scalar_type() == torch::kInt32);
    } else {
        // Check FP8 Q
        std::tie(seq_len, num_heads, head_dim) = get_shape<3>(q_fp);
        DG_HOST_ASSERT(num_heads == 32 or num_heads == 64);
        DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);
        DG_HOST_ASSERT(q_fp.is_contiguous());
        DG_HOST_ASSERT(q_fp.scalar_type() == torch::kFloat8_e4m3fn);

        // Check FP4 KV
        int _head_dim;
        std::tie(seq_len_kv, _head_dim) = get_shape<2>(kv_fp);
        DG_HOST_ASSERT(head_dim == _head_dim);
        DG_HOST_ASSERT(kv_fp.is_contiguous());
        DG_HOST_ASSERT(kv_fp.scalar_type() == torch::kFloat8_e4m3fn);

        // Check SF KV
        auto [_seq_len_kv] = get_shape<1>(kv_sf);
        DG_HOST_ASSERT(seq_len_kv == _seq_len_kv);
        DG_HOST_ASSERT(kv_sf.is_contiguous());
        DG_HOST_ASSERT(kv_sf.scalar_type() == torch::kFloat);
    }
    
    // Check weights
    auto [_seq_len, _num_heads] = get_shape<2>(weights);
    DG_HOST_ASSERT(seq_len == _seq_len and num_heads == _num_heads);
    DG_HOST_ASSERT(weights.stride(1) == 1);
    DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat);

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
    constexpr int block_kv = 256;
    const int block_q = block_qh / num_heads;
    DG_HOST_ASSERT(block_qh % num_heads == 0);

    torch::Tensor logits;
    int aligned_seq_len = align(seq_len, block_q), stride_logits;
    if (max_seqlen_k == 0) {
        // Logits stride must be 16-byte aligned
        stride_logits = align(seq_len_kv + block_kv, 8);
        logits = torch::empty({aligned_seq_len, stride_logits}, q_fp.options().dtype(logits_dtype));
        logits = logits.index({torch::indexing::Slice(0, seq_len), torch::indexing::Slice(0, seq_len_kv)});
    } else {
        stride_logits = align(max_seqlen_k, block_kv);
        logits = torch::empty({aligned_seq_len, stride_logits}, q_fp.options().dtype(logits_dtype));
        logits = logits.index({torch::indexing::Slice(0, seq_len), torch::indexing::Slice(0, max_seqlen_k)});
        DG_HOST_ASSERT(not clean_logits);
    }

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    if (is_fp4 and arch_major == 10) {
        sm100_fp4_mqa_logits(q_fp, q_sf.value(), kv_fp, kv_sf, weights, cu_seq_len_k_start, cu_seq_len_k_end, logits, logits_dtype,
                             seq_len, seq_len_kv, max_seqlen_k, stride_logits, num_heads, head_dim, block_q, block_kv);
    } else if (not is_fp4 and (arch_major == 9 or arch_major == 10)) {
        smxx_fp8_mqa_logits(q_fp, kv_fp, kv_sf, weights, cu_seq_len_k_start, cu_seq_len_k_end, logits, logits_dtype,
                            seq_len, seq_len_kv, max_seqlen_k, stride_logits, num_heads, head_dim, block_q, block_kv);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }

    // Clean unfilled logits
    if (clean_logits)
        smxx_clean_logits(logits, cu_seq_len_k_start, cu_seq_len_k_end, 1, seq_len, seq_len_kv, stride_logits);
    return logits;
}

static torch::Tensor get_paged_mqa_logits_metadata(const torch::Tensor& context_lens, int block_kv, int num_sms) {
    // NOTES: Only 2D context lens is supported for now
    DG_HOST_ASSERT(context_lens.dim() == 2);
    const bool is_context_lens_2d = true;
    const int batch_size = context_lens.size(0);
    const int next_n = context_lens.size(1);
    DG_HOST_ASSERT(context_lens.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(context_lens.is_contiguous());

    // Create metadata tensor
    auto schedule_metadata = torch::empty({num_sms + 1, 2}, context_lens.options());

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    if (arch_major == 9 or arch_major == 10) {
        DG_HOST_ASSERT(block_kv == 64 or (arch_major == 10 and block_kv == 32));
        smxx_paged_mqa_logits_metadata(context_lens, schedule_metadata, batch_size, next_n, block_kv, num_sms, is_context_lens_2d);
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
                                              const at::ScalarType& logits_dtype) {
    const auto [q_fp, q_sf] = q;
    const bool is_fp4 = q_sf.has_value();

    torch::Tensor kv_cache, kv_cache_sf;
    int batch_size, next_n, num_heads, head_dim;
    int num_kv_blocks, block_kv;
    int kv_cache_stride_bytes;
    int block_table_stride = block_table.stride(0);
    int num_sms = device_runtime->get_num_sms();

    if (is_fp4) {
        // Check FP4 Q
        std::tie(batch_size, next_n, num_heads, head_dim) = get_shape<4>(q_fp);
        head_dim *= 2;
        DG_HOST_ASSERT(next_n >= 1);
        DG_HOST_ASSERT(num_heads == 32 or num_heads == 64);
        DG_HOST_ASSERT(head_dim == 128);
        DG_HOST_ASSERT(q_fp.is_contiguous());
        DG_HOST_ASSERT(q_fp.scalar_type() == kPackedFP4);

        // Check SF Q
        auto [_batch_size, _next_n, _num_heads] = get_shape<3>(q_sf.value());
        DG_HOST_ASSERT(batch_size == _batch_size and next_n == _next_n and num_heads == _num_heads);
        DG_HOST_ASSERT(q_sf.value().is_contiguous());
        DG_HOST_ASSERT(q_sf.value().scalar_type() == torch::kInt32);

        // Check fused KV cache
        int num_heads_kv, fp4_with_sf_bytes;
        std::tie(num_kv_blocks, block_kv, num_heads_kv, fp4_with_sf_bytes) = get_shape<4>(fused_kv_cache);
        DG_HOST_ASSERT(block_kv == 32 or block_kv == 64);
        DG_HOST_ASSERT(num_heads_kv == 1 and fp4_with_sf_bytes == head_dim / 2 + static_cast<int>(sizeof(int)));
        DG_HOST_ASSERT(fused_kv_cache.stride(1) == fp4_with_sf_bytes and fused_kv_cache.stride(3) == 1);
        DG_HOST_ASSERT(fused_kv_cache.scalar_type() == torch::kByte);

        // Derive FP4 values and SF tensor
        kv_cache_stride_bytes = fused_kv_cache.stride(0);
        DG_HOST_ASSERT(kv_cache_stride_bytes % sizeof(int) == 0);
        kv_cache = torch::from_blob(
            fused_kv_cache.data_ptr(),
            {num_kv_blocks, block_kv, head_dim / 2},
            {kv_cache_stride_bytes, head_dim / 2, 1},
            torch::TensorOptions().dtype(kPackedFP4)
        );
        kv_cache_sf = torch::from_blob(
            fused_kv_cache.data_ptr<uint8_t>() + block_kv * head_dim / 2,
            {num_kv_blocks, block_kv},
            {kv_cache_stride_bytes / static_cast<int>(sizeof(int)), 1},
            torch::TensorOptions().dtype(torch::kInt32)
        );
    } else {
        // Check FP8 Q
        std::tie(batch_size, next_n, num_heads, head_dim) = get_shape<4>(q_fp);
        DG_HOST_ASSERT(next_n >= 1);
        DG_HOST_ASSERT(num_heads == 32 or num_heads == 64);
        DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);
        DG_HOST_ASSERT(q_fp.is_contiguous());
        DG_HOST_ASSERT(q_fp.scalar_type() == torch::kFloat8_e4m3fn);

        // Check fused KV cache
        int num_heads_kv, head_dim_with_sf;
        std::tie(num_kv_blocks, block_kv, num_heads_kv, head_dim_with_sf) = get_shape<4>(fused_kv_cache);
        DG_HOST_ASSERT(block_kv == 32 or block_kv == 64);
        DG_HOST_ASSERT(num_heads_kv == 1 and head_dim_with_sf == head_dim + static_cast<int>(sizeof(float)));
        DG_HOST_ASSERT(fused_kv_cache.stride(1) == head_dim_with_sf and fused_kv_cache.stride(3) == 1);
        DG_HOST_ASSERT(fused_kv_cache.scalar_type() == torch::kByte);

        // Derive FP8 values and SF tensor
        kv_cache_stride_bytes = fused_kv_cache.stride(0);
        DG_HOST_ASSERT(kv_cache_stride_bytes % sizeof(float) == 0);
        kv_cache = torch::from_blob(
            fused_kv_cache.data_ptr(),
            {num_kv_blocks, block_kv, head_dim},
            {kv_cache_stride_bytes, head_dim, 1},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn)
        );
        kv_cache_sf = torch::from_blob(
            fused_kv_cache.data_ptr<uint8_t>() + block_kv * head_dim,
            {num_kv_blocks, block_kv},
            {kv_cache_stride_bytes / static_cast<int>(sizeof(float)), 1},
            torch::TensorOptions().dtype(torch::kFloat32)
        );

        // Weights must be contiguous for FP8
        DG_HOST_ASSERT(weights.is_contiguous());
    }

    // Check weights
    auto [_batch_size_next_n, _num_heads] = get_shape<2>(weights);
    DG_HOST_ASSERT(_batch_size_next_n == batch_size * next_n and _num_heads == num_heads);
    DG_HOST_ASSERT(weights.stride(1) == 1);
    DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat);

    // Check block table
    auto [_batch_size, _max_block_len] = get_shape<2>(block_table);
    DG_HOST_ASSERT(_batch_size == batch_size);
    DG_HOST_ASSERT(block_table.stride(1) == 1);
    DG_HOST_ASSERT(block_table.scalar_type() == torch::kInt);

    // Check schedule metadata
    auto [_schedule_meta_size, _meta_info_size] = get_shape<2>(schedule_meta);
    DG_HOST_ASSERT(_schedule_meta_size == num_sms + 1 and _meta_info_size == 2);
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
    constexpr int split_kv = 256;
    const auto aligned_max_context_len = align(max_context_len, split_kv);
    auto logits = torch::empty({batch_size * next_n, aligned_max_context_len}, q_fp.options().dtype(logits_dtype));
    logits = logits.slice(-1, 0, max_context_len);
    DG_HOST_ASSERT(logits_dtype == torch::kFloat32 or logits_dtype == torch::kBFloat16);

    // Dispatch implementation
    const auto arch_major = device_runtime->get_arch_major();
    if (is_fp4 and arch_major == 10) {
        sm100_fp4_paged_mqa_logits(q_fp, q_sf.value(), kv_cache, kv_cache_sf, weights, context_lens, logits, block_table, schedule_meta,
                                   logits_dtype, batch_size, next_n, num_heads, head_dim, num_kv_blocks, block_kv, is_context_lens_2d,
                                   aligned_max_context_len, block_table_stride, num_sms, split_kv);
    } else if (not is_fp4 and (arch_major == 9 or arch_major == 10)) {
        smxx_fp8_paged_mqa_logits(q_fp, kv_cache, kv_cache_sf, weights, context_lens, logits, block_table, schedule_meta,
                                  logits_dtype, batch_size, next_n, num_heads, head_dim, num_kv_blocks, block_kv, is_context_lens_2d,
                                  aligned_max_context_len, block_table_stride, num_sms, split_kv);
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
                                          const bool& clean_logits) {
    return fp8_fp4_paged_mqa_logits(std::make_tuple(q, std::nullopt), fused_kv_cache, weights,
                                    context_lens, block_table, schedule_meta,
                                    max_context_len, clean_logits, torch::kFloat);
}
#endif

static void register_apis(pybind11::module_& m) {
#if DG_FP8_COMPATIBLE and DG_TENSORMAP_COMPATIBLE
    m.def("fp8_gemm_nt_skip_head_mid", &fp8_gemm_nt_skip_head_mid,
          py::arg("a"), py::arg("b"), py::arg("d"), py::arg("head_splits"),
          py::arg("recipe") = std::nullopt,
          py::arg("compiled_dims") = "nk",
          py::arg("disable_ue8m0_cast") = false);
    m.def("fp8_fp4_mqa_logits", &fp8_fp4_mqa_logits,
          py::arg("q"), py::arg("kv"), py::arg("weights"),
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"),
          py::arg("clean_logits") = true,
          py::arg("max_seqlen_k") = 0,
          py::arg("logits_dtype") = torch::kFloat32);
    m.def("get_paged_mqa_logits_metadata", &get_paged_mqa_logits_metadata,
          py::arg("context_lens"), py::arg("block_kv"), py::arg("num_sms"));
    m.def("fp8_fp4_paged_mqa_logits", &fp8_fp4_paged_mqa_logits,
          py::arg("q"), py::arg("kv_cache"), py::arg("weights"),
          py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"),
          py::arg("max_context_len"),
          py::arg("clean_logits") = false,
          py::arg("logits_dtype") = torch::kFloat32);
    // Legacy API
    m.def("fp8_mqa_logits", &fp8_mqa_logits,
          py::arg("q"), py::arg("kv"), py::arg("weights"),
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"),
          py::arg("clean_logits") = true,
          py::arg("max_seqlen_k") = 0);
    m.def("fp8_paged_mqa_logits", &fp8_paged_mqa_logits,
          py::arg("q"), py::arg("kv_cache"), py::arg("weights"),
          py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"),
          py::arg("max_context_len"), py::arg("clean_logits") = false);
#endif
}

} // namespace deep_gemm::attention
