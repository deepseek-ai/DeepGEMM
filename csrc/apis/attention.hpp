#pragma once

#include <format>
#include <unordered_map>

#include <c10/cuda/CUDAGraphsC10Utils.h>

#include "../utils/compatibility.hpp"

#include "../jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp"
#include "../jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp"
#include "../jit_kernels/impls/sm100_mqa_logits.hpp"
#include "../jit_kernels/impls/sm100_sparse_mqa_logits.hpp"
#include "../jit_kernels/impls/sm90_fp8_mqa_logits.hpp"

#include "layout.hpp"

namespace deep_gemm::attention {

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
    if (fp8_fp4_requires_k_major(a.first, b.first)) {
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
    const auto arch_major = jit->device.get_arch_major();
    const auto epilogue_type = std::format("epilogue::transform::EpilogueHeadSplits<{}, {}, {}>", left, mid, right);
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
                                        const at::ScalarType& logits_dtype,
                                        const std::optional<torch::Tensor>& schedule_meta = std::nullopt) {
    const auto [q_fp, q_sf] = q;
    const auto [kv_fp, kv_sf] = kv;
    const auto qk_dtype = q_fp.scalar_type();
    const bool is_fp4 = qk_dtype == kPackedFP4;
    const bool is_mx_sf = q_sf.has_value();
    DG_HOST_ASSERT(not is_fp4 or is_mx_sf);

    const auto arch_major = jit->device.get_arch_major();
    // Check Q
    const auto [seq_len, num_heads, head_dim] = get_logical_shape<3>(q_fp);
    DG_HOST_ASSERT((not is_fp4 and head_dim == 32) or head_dim == 64 or head_dim == 128);
    DG_HOST_ASSERT(q_fp.is_contiguous());
    DG_HOST_ASSERT(q_fp.scalar_type() == (is_fp4 ? kPackedFP4 : torch::kFloat8_e4m3fn));

    // Check SF Q
    if (is_mx_sf) {
        DG_HOST_ASSERT(arch_major == 10);
        DG_HOST_ASSERT(q_sf.has_value());
        auto [_seq_len, _num_heads] = get_shape<2>(q_sf.value());
        DG_HOST_ASSERT(seq_len == _seq_len and num_heads == _num_heads);
        DG_HOST_ASSERT(q_sf.value().is_contiguous());
        DG_HOST_ASSERT(q_sf.value().scalar_type() == torch::kInt32);
    }

    // Check KV
    const auto [seq_len_kv, _head_dim] = get_logical_shape<2>(kv_fp);
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
    DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat or (arch_major == 10 and weights.scalar_type() == torch::kBFloat16));
    DG_HOST_ASSERT(weights.scalar_type() != torch::kBFloat16 or logits_dtype == torch::kBFloat16);

    // Check cu_seq_len_k_start
    DG_HOST_ASSERT(cu_seq_len_k_start.size(0) == seq_len);
    DG_HOST_ASSERT(cu_seq_len_k_start.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_start.scalar_type() == torch::kInt);

    // Check cu_seq_len_k_end
    DG_HOST_ASSERT(cu_seq_len_k_end.size(0) == seq_len);
    DG_HOST_ASSERT(cu_seq_len_k_end.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_end.scalar_type() == torch::kInt);

    // Allocate output
    DG_HOST_ASSERT(num_heads > 0 and num_heads <= 128 and num_heads % 4 == 0);
    constexpr int block_kv = 256;
    const int block_q = 128 / num_heads;

    torch::Tensor logits;
    int aligned_seq_len = align(seq_len, block_q), stride_logits;
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
    // NOTES: the logits cleaning is fused into the kernels
    if (arch_major == 10) {
        DG_HOST_ASSERT(qk_dtype == torch::kFloat8_e4m3fn or qk_dtype == kPackedFP4);
        sm100_mqa_logits(q_fp, q_sf, kv_fp, kv_sf, weights, cu_seq_len_k_start, cu_seq_len_k_end, logits, logits_dtype,
                         seq_len, seq_len_kv, max_seqlen_k, stride_logits, num_heads, head_dim, block_q, block_kv,
                         clean_logits, is_mx_sf, qk_dtype, schedule_meta);
    } else if (arch_major == 9) {
        DG_HOST_ASSERT(not schedule_meta.has_value());
        DG_HOST_ASSERT(not is_mx_sf);
        DG_HOST_ASSERT(qk_dtype == torch::kFloat8_e4m3fn);
        DG_HOST_ASSERT(num_heads == 32 or num_heads == 64);
        DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat);
        sm90_fp8_mqa_logits(q_fp, kv_fp, kv_sf, weights, cu_seq_len_k_start, cu_seq_len_k_end, logits, logits_dtype,
                            seq_len, seq_len_kv, max_seqlen_k, stride_logits, num_heads, head_dim, block_q, block_kv,
                            clean_logits);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
    return logits;
}

static torch::Tensor get_mqa_logits_metadata(const torch::Tensor& cu_seq_len_k_start,
                                             const torch::Tensor& cu_seq_len_k_end,
                                             const int& num_kv_tokens, const int& num_heads) {
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);
    const int num_q_tokens = static_cast<int>(cu_seq_len_k_start.size(0));
    DG_HOST_ASSERT(num_q_tokens > 0 and cu_seq_len_k_end.size(0) == num_q_tokens);
    DG_HOST_ASSERT(cu_seq_len_k_start.is_cuda() and cu_seq_len_k_end.is_cuda());
    DG_HOST_ASSERT(cu_seq_len_k_start.is_contiguous() and cu_seq_len_k_end.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_start.scalar_type() == torch::kInt and cu_seq_len_k_end.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(num_kv_tokens > 0);
    DG_HOST_ASSERT(num_heads > 0 and num_heads <= 128 and num_heads % 4 == 0);

    constexpr int split_kv = 256;
    const int block_q = 128 / num_heads;
    const int num_sms = runtime->get_num_sms();
    const int required_words = get_mqa_logits_metadata_num_words(num_q_tokens, block_q, num_sms);
    const auto schedule_meta = torch::empty({required_words}, cu_seq_len_k_start.options());

    sm100_mqa_logits_metadata(cu_seq_len_k_start, cu_seq_len_k_end, schedule_meta,
                              num_q_tokens, num_kv_tokens, block_q, split_kv, num_sms);
    return schedule_meta;
}

static const torch::Tensor& get_sparse_mqa_logits_workspace(const torch::TensorOptions& options,
                                                            const int num_q_tokens) {
    using namespace layout::sparse_mqa_logits;
    constexpr int kNumMaxQTokens = 1 << 20;
    constexpr int64_t kNumWorkspaceBytes =
        sizeof(WorkspaceState) + static_cast<int64_t>(kNumMaxQTokens) * sizeof(QBlockInfo);
    DG_HOST_ASSERT(num_q_tokens <= kNumMaxQTokens);
    const auto stream = at::cuda::getCurrentCUDAStream();
    DG_HOST_ASSERT(options.device() == stream.device());
    static std::unordered_map<c10::cuda::CUDAStream, torch::Tensor> workspaces;
    auto& workspace = workspaces[stream];
    if (not workspace.defined()) {
        // Warm up each stream before capture so one-time zeroing is not replayed with the graph.
        DG_HOST_ASSERT(c10::cuda::currentStreamCaptureStatusMayInitCtx() == c10::cuda::CaptureStatus::None);
        workspace = torch::zeros({kNumWorkspaceBytes}, options.dtype(torch::kByte));
    }
    return workspace;
}

// Each sparse-index row starts with the valid blocks inferred from its KV length. This prefix must
// contain unique, strictly increasing absolute block indices within the corresponding KV range.
// With unaligned ks, block i starts at i * sparse_block_kv + ks % sparse_block_kv.
static torch::Tensor get_sparse_mqa_logits_metadata(const torch::Tensor& cu_seq_len_k_start,
                                                    const torch::Tensor& cu_seq_len_k_end,
                                                    const int& num_kv_tokens,
                                                    const torch::Tensor& sparse_kv_block_indices,
                                                    const at::ScalarType& qk_dtype,
                                                    const int& sparse_block_kv,
                                                    const bool& use_unaligned_ks) {
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);
    const auto [num_q_tokens, num_max_sparse_blocks] = get_shape<2>(sparse_kv_block_indices);
    DG_HOST_ASSERT(num_q_tokens > 0 and num_kv_tokens >= 0);
    DG_HOST_ASSERT(cu_seq_len_k_start.dim() == 1 and cu_seq_len_k_start.size(0) == num_q_tokens);
    DG_HOST_ASSERT(cu_seq_len_k_end.dim() == 1 and cu_seq_len_k_end.size(0) == num_q_tokens);
    DG_HOST_ASSERT(sparse_kv_block_indices.scalar_type() == torch::kInt and sparse_kv_block_indices.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_start.scalar_type() == torch::kInt and cu_seq_len_k_start.is_contiguous());
    DG_HOST_ASSERT(cu_seq_len_k_end.scalar_type() == torch::kInt and cu_seq_len_k_end.is_contiguous());

    const int split_kv = get_sparse_mqa_split_kv(qk_dtype);
    auto metadata = torch::empty({get_num_metadata_bytes(num_q_tokens, num_max_sparse_blocks, sparse_block_kv,
                                                         split_kv, false, runtime->get_num_sms())},
                                 sparse_kv_block_indices.options().dtype(torch::kUInt8));
    const auto& workspace = get_sparse_mqa_logits_workspace(metadata.options(), num_q_tokens);
    launch_sm100_sparse_mqa_logits_metadata(false, use_unaligned_ks, 1, num_kv_tokens, 0,
                                            sparse_kv_block_indices, metadata, workspace, split_kv, sparse_block_kv,
                                            cu_seq_len_k_start.data_ptr<int>(), cu_seq_len_k_end.data_ptr<int>(),
                                            nullptr, nullptr, nullptr);
    return metadata;
}

// Queries belonging to one request must be consecutive. Each sparse-index row starts with unique,
// strictly increasing logical block indices within its context length. Paired queries must also
// have identical block-table rows.
static torch::Tensor get_paged_sparse_mqa_logits_metadata(const torch::Tensor& context_lens,
                                                          const torch::Tensor& block_table,
                                                          const torch::Tensor& indices,
                                                          const int& page_kv,
                                                          const torch::Tensor& sparse_kv_block_indices,
                                                          const at::ScalarType& qk_dtype,
                                                          const int& sparse_block_kv) {
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);
    const auto [num_q_tokens, num_max_sparse_blocks] = get_shape<2>(sparse_kv_block_indices);
    DG_HOST_ASSERT(num_q_tokens > 0);
    DG_HOST_ASSERT(context_lens.numel() == num_q_tokens and context_lens.scalar_type() == torch::kInt and context_lens.is_contiguous());
    DG_HOST_ASSERT(block_table.dim() == 2 and block_table.size(0) == num_q_tokens and block_table.size(1) > 0 and
                   block_table.scalar_type() == torch::kInt and block_table.stride(1) == 1);
    DG_HOST_ASSERT(block_table.stride(0) <= std::numeric_limits<uint32_t>::max());
    DG_HOST_ASSERT(indices.dim() == 1 and indices.size(0) == num_q_tokens and
                   indices.scalar_type() == torch::kInt and indices.is_contiguous());
    DG_HOST_ASSERT(sparse_block_kv == 8 or sparse_block_kv == 16);
    DG_HOST_ASSERT(page_kv > 0 and page_kv % sparse_block_kv == 0);
    DG_HOST_ASSERT(sparse_kv_block_indices.scalar_type() == torch::kInt and sparse_kv_block_indices.is_contiguous());

    const int split_kv = get_sparse_mqa_split_kv(qk_dtype);
    auto metadata = torch::empty({get_num_metadata_bytes(num_q_tokens, num_max_sparse_blocks, sparse_block_kv,
                                                         split_kv, true, runtime->get_num_sms())},
                                 sparse_kv_block_indices.options().dtype(torch::kUInt8));
    const auto& workspace = get_sparse_mqa_logits_workspace(metadata.options(), num_q_tokens);
    launch_sm100_sparse_mqa_logits_metadata(true, false, page_kv, 0,
                                            static_cast<int>(block_table.stride(0)), sparse_kv_block_indices,
                                            metadata, workspace, split_kv, sparse_block_kv, nullptr, nullptr,
                                            context_lens.data_ptr<int>(), block_table.data_ptr<int>(), indices.data_ptr<int>());
    return metadata;
}

// Skip metadata header validation to avoid synchronizing the stream
static torch::Tensor fp8_fp4_sparse_mqa_logits(const std::tuple<torch::Tensor, std::optional<torch::Tensor>>& q,
                                               const std::tuple<torch::Tensor, torch::Tensor>& kv,
                                               const torch::Tensor& weights,
                                               const torch::Tensor& metadata,
                                               const int& num_max_sparse_blocks,
                                               const int& sparse_block_kv,
                                               const bool& use_unaligned_ks) {
    using namespace layout::sparse_mqa_logits;
    const auto [q_fp, q_sf_optional] = q;
    const auto [kv_fp, kv_sf] = kv;
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10 and q_sf_optional.has_value());
    DG_HOST_ASSERT(num_max_sparse_blocks > 0 and num_max_sparse_blocks % 4 == 0 and num_max_sparse_blocks <= 4096);
    DG_HOST_ASSERT(sparse_block_kv == 8 or sparse_block_kv == 16);
    const auto& q_sf = q_sf_optional.value();
    const auto qk_dtype = q_fp.scalar_type();
    const bool is_fp4 = qk_dtype == kPackedFP4;

    const auto [num_q_tokens, num_heads, head_dim] = get_logical_shape<3>(q_fp);
    DG_HOST_ASSERT(num_q_tokens > 0 and num_heads == kNumHeads and head_dim == kHeadDim);
    DG_HOST_ASSERT((is_fp4 or qk_dtype == torch::kFloat8_e4m3fn) and q_fp.is_contiguous());
    const auto [_num_q_tokens_sf, _num_heads_sf] = get_shape<2>(q_sf);
    DG_HOST_ASSERT(_num_q_tokens_sf == num_q_tokens and _num_heads_sf == num_heads);
    DG_HOST_ASSERT(q_sf.scalar_type() == torch::kInt32 and q_sf.is_contiguous());

    const auto [num_kv_tokens, kv_head_dim] = get_logical_shape<2>(kv_fp);
    DG_HOST_ASSERT(num_kv_tokens > 0 and kv_head_dim == head_dim and kv_fp.scalar_type() == qk_dtype and kv_fp.is_contiguous());
    const auto [_num_kv_tokens_sf] = get_shape<1>(kv_sf);
    DG_HOST_ASSERT(_num_kv_tokens_sf == num_kv_tokens and kv_sf.scalar_type() == torch::kInt32 and kv_sf.is_contiguous());

    const auto [_num_q_tokens_weights, _num_heads_weights] = get_shape<2>(weights);
    DG_HOST_ASSERT(_num_q_tokens_weights == num_q_tokens and _num_heads_weights == num_heads);
    DG_HOST_ASSERT(weights.scalar_type() == torch::kBFloat16 and weights.stride(1) == 1);

    const int num_output_tokens = num_max_sparse_blocks * sparse_block_kv;
    const int logits_stride = align(num_output_tokens, 1024 / static_cast<int>(sizeof(nv_bfloat16)));
    auto logits = torch::empty({align<int>(num_q_tokens, kBlockQ), logits_stride}, q_fp.options().dtype(torch::kBFloat16));
    logits = logits.index({torch::indexing::Slice(0, num_q_tokens), torch::indexing::Slice(0, num_output_tokens)});
    launch_sm100_sparse_mqa_logits(false, use_unaligned_ks, sparse_block_kv,
                                   q_fp, q_sf, kv_fp, kv_sf, weights, metadata, logits);
    return logits;
}

// Skip metadata header validation to avoid synchronizing the stream
static torch::Tensor fp8_fp4_paged_sparse_mqa_logits(const std::tuple<torch::Tensor, std::optional<torch::Tensor>>& q,
                                                     const torch::Tensor& fused_kv_cache,
                                                     const torch::Tensor& weights,
                                                     const torch::Tensor& metadata,
                                                     const int& num_max_sparse_blocks,
                                                     const int& sparse_block_kv) {
    using namespace layout::sparse_mqa_logits;
    const auto [q_fp, q_sf_optional] = q;
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10 and q_sf_optional.has_value());
    DG_HOST_ASSERT(num_max_sparse_blocks > 0 and num_max_sparse_blocks % 4 == 0 and num_max_sparse_blocks <= 4096);
    DG_HOST_ASSERT(sparse_block_kv == 8 or sparse_block_kv == 16);
    const auto& q_sf = q_sf_optional.value();
    const auto qk_dtype = q_fp.scalar_type();
    const bool is_fp4 = qk_dtype == kPackedFP4;

    const auto [num_q_tokens, next_n, num_heads, head_dim] = get_logical_shape<4>(q_fp);
    DG_HOST_ASSERT(num_q_tokens > 0 and next_n == 1 and num_heads == kNumHeads and head_dim == kHeadDim);
    DG_HOST_ASSERT((is_fp4 or qk_dtype == torch::kFloat8_e4m3fn) and q_fp.is_contiguous());
    const auto [_num_q_tokens_sf, _next_n_sf, _num_heads_sf] = get_shape<3>(q_sf);
    DG_HOST_ASSERT(_num_q_tokens_sf == num_q_tokens and _next_n_sf == 1 and _num_heads_sf == num_heads);
    DG_HOST_ASSERT(q_sf.scalar_type() == torch::kInt32 and q_sf.is_contiguous());

    const auto [num_kv_pages, page_kv, num_kv_heads, head_dim_with_sf] = get_shape<4>(fused_kv_cache);
    DG_HOST_ASSERT(num_kv_pages > 0 and page_kv > 0 and page_kv % sparse_block_kv == 0 and num_kv_heads == 1 and
                   head_dim_with_sf == (is_fp4 ? head_dim / 2 : head_dim) + static_cast<int>(sizeof(int)));
    DG_HOST_ASSERT(fused_kv_cache.scalar_type() == torch::kUInt8 and fused_kv_cache.stride(1) == head_dim_with_sf and
                   fused_kv_cache.stride(3) == 1 and fused_kv_cache.stride(0) <= std::numeric_limits<int>::max() and
                   fused_kv_cache.stride(0) % 512 == 0);

    const auto [_num_q_tokens_weights, _num_heads_weights] = get_shape<2>(weights);
    DG_HOST_ASSERT(_num_q_tokens_weights == num_q_tokens and _num_heads_weights == num_heads);
    DG_HOST_ASSERT(weights.scalar_type() == torch::kBFloat16 and weights.stride(1) == 1);

    const int num_output_tokens = num_max_sparse_blocks * sparse_block_kv;
    const int logits_stride = align(num_output_tokens, 1024 / static_cast<int>(sizeof(nv_bfloat16)));
    auto logits = torch::empty({align<int>(num_q_tokens, kBlockQ), logits_stride}, q_fp.options().dtype(torch::kBFloat16));
    logits = logits.index({torch::indexing::Slice(0, num_q_tokens), torch::indexing::Slice(0, num_output_tokens)});
    launch_sm100_sparse_mqa_logits(true, false, sparse_block_kv, q_fp, q_sf, fused_kv_cache, torch::Tensor(),
                                   weights, metadata, logits);
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

    // Create metadata tensor
    auto schedule_metadata = torch::empty({num_sms + 1, 2}, context_lens.options());

    // Dispatch implementation
    const auto arch_major = jit->device.get_arch_major();
    if (is_varlen) {
        const auto& indices_tensor = indices.value();
        DG_HOST_ASSERT(arch_major == 10 and next_n == 1 and (block_kv == 32 or block_kv == 64 or block_kv == 128));
        DG_HOST_ASSERT(indices_tensor.dim() == 1 and indices_tensor.size(0) == batch_size);
        DG_HOST_ASSERT(indices_tensor.is_contiguous());
        DG_HOST_ASSERT(indices_tensor.scalar_type() == torch::kInt);
        sm100_paged_mqa_logits_metadata(context_lens, schedule_metadata, batch_size, batch_size * next_n, next_n, num_sms, is_context_lens_2d, true, indices_tensor.data_ptr<int>());
    } else if (arch_major == 10) {
        DG_HOST_ASSERT(block_kv == 32 or block_kv == 64 or block_kv == 128);
        sm100_paged_mqa_logits_metadata(context_lens, schedule_metadata, batch_size, batch_size * next_n, next_n, num_sms, is_context_lens_2d, false, nullptr);
    } else if (arch_major == 9) {
        DG_HOST_ASSERT(block_kv == 64);
        sm90_paged_mqa_logits_metadata(context_lens, schedule_metadata, batch_size, next_n, block_kv, num_sms, is_context_lens_2d, false, nullptr);
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
    // NOTES: cleaning is not supported for paged logits
    DG_HOST_ASSERT(not clean_logits);

    torch::Tensor kv_cache, kv_cache_sf;
    int kv_cache_stride_bytes;
    int block_table_stride = block_table.stride(0);
    int num_sms = runtime->get_num_sms();
    const auto arch_major = jit->device.get_arch_major();

    // Check Q
    const auto [batch_size, next_n, num_heads, head_dim] = get_logical_shape<4>(q_fp);
    DG_HOST_ASSERT((not is_fp4 and head_dim == 32) or head_dim == 64 or head_dim == 128);
    DG_HOST_ASSERT(q_fp.is_contiguous());
    DG_HOST_ASSERT(q_fp.scalar_type() == (is_fp4 ? kPackedFP4 : torch::kFloat8_e4m3fn));
    DG_HOST_ASSERT(next_n >= 1);

    // Check SF Q
    if (is_mx_sf) {
        DG_HOST_ASSERT(arch_major == 10);
        DG_HOST_ASSERT(q_sf.has_value());
        auto [_batch_size, _next_n, _num_heads] = get_shape<3>(q_sf.value());
        DG_HOST_ASSERT(batch_size == _batch_size and next_n == _next_n and num_heads == _num_heads);
        DG_HOST_ASSERT(q_sf.value().is_contiguous());
        DG_HOST_ASSERT(q_sf.value().scalar_type() == torch::kInt32);
    }

    // Check fused KV cache
    const auto [num_kv_blocks, block_kv, num_heads_kv, head_dim_with_sf] = get_shape<4>(fused_kv_cache);
    DG_HOST_ASSERT((arch_major == 10 and (block_kv == 32 or block_kv == 64 or block_kv == 128)) or
                   (arch_major == 9 and block_kv == 64));
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
        DG_HOST_ASSERT(arch_major == 10 and next_n == 1);
        DG_HOST_ASSERT(indices_tensor.dim() == 1 and indices_tensor.size(0) == batch_size);
        DG_HOST_ASSERT(indices_tensor.is_contiguous());
        DG_HOST_ASSERT(indices_tensor.scalar_type() == torch::kInt);
    }

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
    DG_HOST_ASSERT(logits_dtype == torch::kFloat32 or logits_dtype == torch::kBFloat16);
    constexpr int split_kv = 256;
    // Logits row stride must be 1024-byte aligned
    const int stride_logits_alignment = 1024 / static_cast<int>(c10::elementSize(logits_dtype));
    const auto aligned_max_context_len = align(align(max_context_len, split_kv), stride_logits_alignment);
    auto logits = torch::empty({batch_size * next_n, aligned_max_context_len}, q_fp.options().dtype(logits_dtype));
    logits = logits.slice(-1, 0, max_context_len);

    // Dispatch implementation
    if (arch_major == 10) {
        DG_HOST_ASSERT(qk_dtype == torch::kFloat8_e4m3fn or qk_dtype == kPackedFP4);
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
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
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
static void register_apis(pybind11::module_& m) {
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
          py::arg("logits_dtype") = torch::kFloat32,
          py::arg("schedule_meta") = std::nullopt);
    m.def("get_mqa_logits_metadata", &get_mqa_logits_metadata,
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"),
          py::arg("num_kv_tokens"), py::arg("num_heads"));
    m.def("get_sparse_mqa_logits_metadata", &get_sparse_mqa_logits_metadata,
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"), py::arg("num_kv_tokens"),
          py::arg("sparse_kv_block_indices"), py::arg("qk_dtype"), py::arg("sparse_block_kv"),
          py::arg("use_unaligned_ks") = false);
    m.def("get_paged_sparse_mqa_logits_metadata", &get_paged_sparse_mqa_logits_metadata,
          py::arg("context_lens"), py::arg("block_table"), py::arg("indices"), py::arg("page_kv"),
          py::arg("sparse_kv_block_indices"), py::arg("qk_dtype"), py::arg("sparse_block_kv"));
    m.def("fp8_fp4_sparse_mqa_logits", &fp8_fp4_sparse_mqa_logits,
          py::arg("q"), py::arg("kv"), py::arg("weights"), py::arg("metadata"),
          py::arg("num_max_sparse_blocks"), py::arg("sparse_block_kv"),
          py::arg("use_unaligned_ks") = false);
    m.def("fp8_fp4_paged_sparse_mqa_logits", &fp8_fp4_paged_sparse_mqa_logits,
          py::arg("q"), py::arg("kv_cache"), py::arg("weights"), py::arg("metadata"),
          py::arg("num_max_sparse_blocks"), py::arg("sparse_block_kv"));
    m.def("get_paged_mqa_logits_metadata", &get_paged_mqa_logits_metadata,
          py::arg("context_lens"), py::arg("block_kv"), py::arg("num_sms"),
          py::arg("indices") = std::nullopt);
    m.def("fp8_fp4_paged_mqa_logits", &fp8_fp4_paged_mqa_logits,
          py::arg("q"), py::arg("kv_cache"), py::arg("weights"),
          py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"),
          py::arg("max_context_len"),
          py::arg("clean_logits") = false,
          py::arg("logits_dtype") = torch::kFloat32,
          py::arg("indices") = std::nullopt);
    // Legacy API
    m.def("fp8_mqa_logits", &fp8_mqa_logits,
          py::arg("q"), py::arg("kv"), py::arg("weights"),
          py::arg("cu_seq_len_k_start"), py::arg("cu_seq_len_k_end"),
          py::arg("clean_logits") = true,
          py::arg("max_seqlen_k") = 0);
    m.def("fp8_paged_mqa_logits", &fp8_paged_mqa_logits,
          py::arg("q"), py::arg("kv_cache"), py::arg("weights"),
          py::arg("context_lens"), py::arg("block_table"), py::arg("schedule_meta"),
          py::arg("max_context_len"), py::arg("clean_logits") = false,
          py::arg("indices") = std::nullopt);
}

} // namespace deep_gemm::attention
