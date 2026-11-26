#pragma once

#include <format>
#include <torch/python.h>

#include "../runtime/jit.hpp"
#include "../runtime/runtime.hpp"
#include "../utils/layout.hpp"
#include "../jit_kernels/heuristics/sm100.hpp"
#include "../jit_kernels/impls/runtime_utils.hpp"
#include "../jit_kernels/impls/smxx_cublaslt.hpp"

namespace deep_gemm::experimental {

static PipelineConfig get_newton_schulz_pipeline_config(const int block_m, const int block_n, const int block_k) {
    constexpr int kSharedMemoryAlignment = 1024;
    constexpr int kNumTMAStoreStages = 2;
    const auto num_operand_bytes = (block_m + block_n) / 2 * block_k * sizeof(uint16_t);
    const auto num_store_bytes = kNumTMAStoreStages * block_m / 2 * block_k * sizeof(uint16_t);

    // The final alignment page holds the barriers, task queue and TMEM allocation pointer.
    const auto num_stages = (SM100ArchSpec::smem_capacity - num_store_bytes - kSharedMemoryAlignment) / num_operand_bytes;
    const auto num_smem_bytes = num_stages * num_operand_bytes + num_store_bytes + kSharedMemoryAlignment;
    DG_HOST_ASSERT(num_stages >= block_m / block_k);
    return PipelineConfig(static_cast<int>(num_smem_bytes), static_cast<int>(num_stages), kNumTMAStoreStages);
}

static void sm100_bf16_batched_newton_schulz(
    const torch::Tensor& input, const torch::Tensor& gram,
    const torch::Tensor& polynomial, const torch::Tensor& output,
    const float identity, const float linear, const float quadratic) {
    constexpr uint32_t BLOCK_M = 256, BLOCK_N = 256, BLOCK_K = 64;
    constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2, STORE_BLOCK_M = 64;
    constexpr uint32_t kNumCTAs = 2;
    constexpr uint32_t kNumReadyWordsPerWarp = 32 * 4;
    const auto batches = static_cast<uint32_t>(input.size(0));
    const auto rows = static_cast<uint32_t>(input.size(1));
    const auto columns = static_cast<uint32_t>(input.size(2));
    const auto num_clusters = runtime->get_num_sms() / kNumCTAs;
    const auto tile_rows = ceil_div(rows, BLOCK_M);
    const auto num_tiles = batches * tile_rows * (tile_rows + 1) / 2;
    const auto num_ready_words = ceil_div(ceil_div(num_tiles, 32u), kNumReadyWordsPerWarp) * kNumReadyWordsPerWarp;
    const auto config = get_newton_schulz_pipeline_config(BLOCK_M, BLOCK_N, BLOCK_K);
    const auto stream = at::cuda::getCurrentCUDAStream();

    // Two completion bits per tile distinguish the two CTA writebacks.
    // The final word contains the monotonically increasing task counter.
    const auto ready = torch::zeros({static_cast<int64_t>(2 * num_ready_words + 1)}, input.options().dtype(torch::kInt64));

    // Each descriptor maps one contiguous operand panel or output strip.
    const auto make_descriptor = [](const torch::Tensor& tensor, const uint32_t inner, const uint32_t outer) {
        return make_tma_3d_desc(tensor, tensor.size(2), tensor.size(1), tensor.size(0),
            inner, outer, 1, tensor.stride(1), tensor.stride(0), 128);
    };
    const auto map_x = make_descriptor(input, BLOCK_K, LOAD_BLOCK_M);
    const auto map_a = make_descriptor(gram, BLOCK_K, LOAD_BLOCK_M);
    const auto map_at = make_descriptor(gram, LOAD_BLOCK_M, BLOCK_K);
    const auto map_b = make_descriptor(polynomial, BLOCK_K, LOAD_BLOCK_M);
    const auto map_bt = make_descriptor(polynomial, LOAD_BLOCK_M, BLOCK_K);
    const auto map_xt = make_descriptor(input, LOAD_BLOCK_M, BLOCK_K);
    const auto map_out = make_descriptor(output, LOAD_BLOCK_M, STORE_BLOCK_M);

    const auto kernel = jit->compile("sm100_bf16_newton_schulz_impl", std::format(R"(
#include <deep_gemm/experimental/sm100_bf16_newton_schulz.cuh>

using namespace deep_gemm::experimental;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_newton_schulz_impl<{}, {}, {}, {}>);
}}
)", batches, rows, columns, config.num_stages));

    const auto options = deep_jit::cuda::LaunchOptions {
        .stream = stream,
        .num_smem_bytes = static_cast<uint32_t>(config.smem_size),
        .grid_dim = dim3(num_clusters * kNumCTAs, 1, 1),
        .block_dim = dim3(256, 1, 1),
        .cluster_dim = dim3(kNumCTAs, 1, 1),
        .cooperative = true,
        .enable_pdl = false,
        .nonportable_cluster_size_allowed = false,
    };

    jit->launch(kernel, options, reinterpret_cast<uint64_t*>(ready.data_ptr()),
        identity, linear, quadratic, map_x, map_a, map_at, map_b, map_bt, map_xt, map_out);
}

static void bf16_newton_schulz_polynomial(
    const torch::Tensor& gram, const torch::Tensor& polynomial,
    const float identity, const float linear, const float quadratic) {
    DG_HOST_ASSERT(gram.dim() == 3 and polynomial.dim() == 3);
    DG_HOST_ASSERT(gram.sizes() == polynomial.sizes() and gram.size(1) == gram.size(2));
    DG_HOST_ASSERT(gram.scalar_type() == torch::kBFloat16 and polynomial.scalar_type() == gram.scalar_type());
    DG_HOST_ASSERT(gram.is_contiguous() and polynomial.is_contiguous());
    const auto rows = static_cast<uint32_t>(gram.size(2));
    const auto num_elements = static_cast<uint64_t>(gram.numel());
    if (num_elements == 0)
        return;
    constexpr uint32_t kNumThreads = 128;
    constexpr uint32_t kNumElementsPerBlock = 1024;
    const auto kernel = jit->compile("bf16_newton_schulz_polynomial_impl", std::format(R"(
#include <deep_gemm/experimental/sm100_bf16_newton_schulz.cuh>

using namespace deep_gemm::experimental;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&bf16_newton_schulz_polynomial_impl<{}, {}, {}>);
}}
)", rows, kNumThreads, kNumElementsPerBlock));
    const auto options = deep_jit::cuda::LaunchOptions {
        .stream = at::cuda::getCurrentCUDAStream(),
        .num_smem_bytes = 0,
        .grid_dim = dim3(ceil_div(num_elements, uint64_t(kNumElementsPerBlock)), 1, 1),
        .block_dim = dim3(kNumThreads, 1, 1),
        .cluster_dim = dim3(1, 1, 1),
        .enable_pdl = false,
    };
    jit->launch(kernel, options, gram.data_ptr(), polynomial.data_ptr(), num_elements,
        identity, linear, quadratic);
}

static void batched_bf16_newton_schulz(
    const torch::Tensor& input, const torch::Tensor& output,
    const float identity, const float linear, const float quadratic) {
    const auto input_3d = input.dim() == 2 ? input.unsqueeze(0) : input;
    const auto output_3d = output.dim() == 2 ? output.unsqueeze(0) : output;
    const auto [num_batches, rows, columns] = get_shape<3>(input_3d);
    const auto [num_batches_out, rows_out, columns_out] = get_shape<3>(output_3d);
    const auto major_input = get_major_type_ab<false>(input_3d);
    check_major_type_cd<false>(output_3d);
    DG_HOST_ASSERT(num_batches == num_batches_out and rows == rows_out and columns == columns_out);
    DG_HOST_ASSERT(input_3d.scalar_type() == torch::kBFloat16 and output_3d.scalar_type() == input_3d.scalar_type());

    if (num_batches == 0 or rows == 0 or columns == 0)
        return;
    DG_HOST_ASSERT(input_3d.data_ptr() != output_3d.data_ptr());

    const auto gram = torch::empty({num_batches, rows, rows}, input.options());
    const auto polynomial = torch::empty_like(gram);
    constexpr uint32_t BLOCK_M = 256;
    const auto num_clusters = runtime->get_num_sms() / 2;
    // Complete tiles provide the work needed to amortize the persistent launch.
    const auto full_tile_rows = rows / BLOCK_M;
    const auto num_full_tiles = static_cast<uint32_t>(num_batches) * full_tile_rows * (full_tile_rows + 1) / 2;
    // Two tile rows and a full triangular wave amortize persistent scheduling across the SMs.
    // Contiguous panels allow the three products to share the same TMA descriptors.
    const bool use_fused = jit->device.get_arch_major() == 10 and rows >= 2 * BLOCK_M and
        input_3d.is_contiguous() and output_3d.is_contiguous() and
        num_clusters > 0 and num_full_tiles >= num_clusters;

    if (use_fused) {
        sm100_bf16_batched_newton_schulz(input_3d, gram, polynomial, output_3d,
            identity, linear, quadratic);
    } else {
        constexpr auto major_gram = cute::UMMA::Major::K;
        cublaslt_batched_gemm(input_3d, input_3d, gram, rows, rows, columns,
            num_batches, major_input, major_input);
        cublaslt_batched_gemm(gram, gram, polynomial, rows, rows, rows,
            num_batches, major_gram, major_gram);
        bf16_newton_schulz_polynomial(gram, polynomial, identity, linear, quadratic);
        const auto input_transposed = input_3d.transpose(1, 2);
        const auto major_transposed = input_3d.stride(1) == 1 ? cute::UMMA::Major::K : cute::UMMA::Major::MN;
        cublaslt_batched_gemm(polynomial, input_transposed, output_3d, rows, columns, rows,
            num_batches, major_gram, major_transposed);
    }
}

} // namespace deep_gemm::experimental
