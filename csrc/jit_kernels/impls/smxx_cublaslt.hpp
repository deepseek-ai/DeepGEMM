#pragma once

#include <cublasLt.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <cute/arch/mma_sm100_umma.hpp>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/compatibility.hpp"

namespace deep_gemm {

static auto get_cublaslt_layout(const cudaDataType& type, const int& rows, const int& cols, const int64_t& ld,
                                const std::optional<int>& batch_count = std::nullopt,
                                const std::optional<int64_t>& batch_offset = std::nullopt) {
    cublasLtMatrixLayout_t layout;
    DG_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layout, type, rows, cols, ld));
    if (batch_count.has_value()) {
        DG_HOST_ASSERT(batch_offset.has_value());

        DG_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count.value(), sizeof(batch_count.value())));
        DG_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_offset.value(), sizeof(batch_offset.value())));
    }
    return layout;
}

static void call_cublaslt_api(const cublasOperation_t& trans_a,
                              const cublasOperation_t& trans_b,
                              const cublasLtMatrixLayout_t& layout_a,
                              const cublasLtMatrixLayout_t& layout_b,
                              const cublasLtMatrixLayout_t& layout_d,
                              const torch::Tensor& a,
                              const torch::Tensor& b,
                              const torch::Tensor& d,
                              const bool& accumulate,
                              const float& alpha = 1.0f,
                              // NOTES: block-scaled UE4M3 scale pointers for the cuBLASLt A/B operands
                              // (i.e. the already-swapped column-major operands), `nullptr` means no block scaling
                              const void* a_block_sf = nullptr,
                              const void* b_block_sf = nullptr) {
    const bool with_block_sf = a_block_sf != nullptr;
    DG_HOST_ASSERT(with_block_sf == (b_block_sf != nullptr));

    // Block-scaled matmuls require the plain FP32 compute type
    cublasComputeType_t compute_type = with_block_sf ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_TF32;
    cudaDataType_t scale_type = CUDA_R_32F;

    // Operation description
    cublasLtMatmulDesc_t desc;
    DG_CUBLASLT_CHECK(cublasLtMatmulDescCreate(&desc, compute_type, scale_type));
    DG_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    DG_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));
    DG_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

    if (with_block_sf) {
        // NOTES: cuBLASLt only supports the NVFP4 recipe for FP4 operands
        // (UE4M3 scaling factors with 16-element blocks)
        const int32_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        DG_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
        DG_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)));
        DG_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_block_sf, sizeof(a_block_sf)));
        DG_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_block_sf, sizeof(b_block_sf)));
    }

    const int num_math_sms = runtime->get_num_sms();
    DG_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &num_math_sms, sizeof(num_math_sms)));

    bool fp8_fast_accumulate = false;
    if (a.scalar_type() == torch::kFloat8_e4m3fn)
        DG_CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fp8_fast_accumulate, sizeof(fp8_fast_accumulate)));

    // Get cuBLASLt handle, workspace, and stream
    const auto handle = runtime->get_cublaslt_handle();
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto workspace = runtime->get_cublaslt_workspace(stream);
    const auto workspace_bytes = workspace.nbytes();

    // Algorithm selection
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulHeuristicResult_t heuristic;
    int num_heuristic_results = 0;
    uint32_t reduction_scheme_mask = CUBLASLT_REDUCTION_SCHEME_NONE | CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
    DG_CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    DG_CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                           &workspace_bytes, sizeof(workspace_bytes)));
    DG_CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                                           &reduction_scheme_mask, sizeof(reduction_scheme_mask)));
    DG_CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(handle, desc, layout_a, layout_b, layout_d, layout_d,
                                                     pref, 1, &heuristic, &num_heuristic_results));
    DG_HOST_ASSERT(num_heuristic_results == 1 and "Unable to find any algorithm for the GEMM");

    // Call: D = alpha * (A @ B) + beta * C
    const float beta = accumulate ? 1.0f : 0.0f;
    DG_CUBLASLT_CHECK(cublasLtMatmul(handle,                                // Light handle
                                     desc,                                  // Operation description
                                     &alpha,                                // Alpha
                                     b.data_ptr(), layout_a,                // A
                                     a.data_ptr(), layout_b,                // B
                                     &beta,                                 // Beta
                                     d.data_ptr(), layout_d,                // C
                                     d.data_ptr(), layout_d,                // D
                                     &heuristic.algo,                       // Algorithm
                                     workspace.data_ptr(), workspace_bytes, // Workspace
                                     stream));                              // Stream

    // Free memory
    DG_CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(pref));
    DG_CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layout_a));
    DG_CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layout_b));
    DG_CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(layout_d));
    DG_CUBLASLT_CHECK(cublasLtMatmulDescDestroy(desc));
}

static void cublaslt_gemm(const torch::Tensor& lhs, const torch::Tensor& rhs,
                          const torch::Tensor& out,
                          const int& m, const int& n, const int& k,
                          const cute::UMMA::Major& a_major, const cute::UMMA::Major& b_major,
                          const bool& accumulate,
                          const float& alpha = 1.0f) {
    const auto trans_a = b_major == cute::UMMA::Major::K ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto trans_b = a_major == cute::UMMA::Major::K ? CUBLAS_OP_N : CUBLAS_OP_T;

    // Matrix layouts
    const auto cuda_type_a = at::cuda::ScalarTypeToCudaDataType(rhs.scalar_type());
    const auto cuda_type_b = at::cuda::ScalarTypeToCudaDataType(lhs.scalar_type());
    const auto cuda_type_d = at::cuda::ScalarTypeToCudaDataType(out.scalar_type());
    const auto layout_a = b_major == cute::UMMA::Major::K ? get_cublaslt_layout(cuda_type_a, k, n, rhs.stride(0))
                                                          : get_cublaslt_layout(cuda_type_a, n, k, rhs.stride(1));
    const auto layout_b = a_major == cute::UMMA::Major::K ? get_cublaslt_layout(cuda_type_b, k, m, lhs.stride(0))
                                                          : get_cublaslt_layout(cuda_type_b, m, k, lhs.stride(1));
    const auto layout_d = get_cublaslt_layout(cuda_type_d, n, m, out.stride(0));

    call_cublaslt_api(trans_a, trans_b, layout_a, layout_b, layout_d, lhs, rhs, out, accumulate, alpha);
}

static void cublaslt_batched_gemm(const torch::Tensor& lhs, const torch::Tensor& rhs,
                                  const torch::Tensor& out,
                                  const uint32_t& m, const uint32_t& n, const uint32_t& k,
                                  const uint32_t& num_batches,
                                  const cute::UMMA::Major& a_major, const cute::UMMA::Major& b_major) {
    const auto trans_a = b_major == cute::UMMA::Major::K ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto trans_b = a_major == cute::UMMA::Major::K ? CUBLAS_OP_N : CUBLAS_OP_T;
    const auto lhs_batch_offset = lhs.stride(0);
    const auto rhs_batch_offset = rhs.stride(0);
    const auto out_batch_offset = out.stride(0);

    // cuBLASLt uses column-major layouts. Swap the operands so that their column-major
    // interpretation computes the transpose of the requested row-major result.
    const auto cuda_type_a = at::cuda::ScalarTypeToCudaDataType(rhs.scalar_type());
    const auto cuda_type_b = at::cuda::ScalarTypeToCudaDataType(lhs.scalar_type());
    const auto cuda_type_d = at::cuda::ScalarTypeToCudaDataType(out.scalar_type());
    const auto layout_a = b_major == cute::UMMA::Major::K ?
        get_cublaslt_layout(cuda_type_a, k, n, rhs.stride(-2), num_batches, rhs_batch_offset) :
        get_cublaslt_layout(cuda_type_a, n, k, rhs.stride(-1), num_batches, rhs_batch_offset);
    const auto layout_b = a_major == cute::UMMA::Major::K ?
        get_cublaslt_layout(cuda_type_b, k, m, lhs.stride(-2), num_batches, lhs_batch_offset) :
        get_cublaslt_layout(cuda_type_b, m, k, lhs.stride(-1), num_batches, lhs_batch_offset);
    const auto layout_d = get_cublaslt_layout(cuda_type_d, n, m, out.stride(-2),
                                              num_batches, out_batch_offset);

    call_cublaslt_api(trans_a, trans_b, layout_a, layout_b, layout_d, lhs, rhs, out, false);
}

static void cublaslt_nvfp4_gemm(const torch::Tensor& lhs, const torch::Tensor& lhs_sf,
                                const torch::Tensor& rhs, const torch::Tensor& rhs_sf,
                                const torch::Tensor& out,
                                const int& m, const int& n, const int& k,
                                const bool& accumulate) {
    // NOTES: block-scaled FP4 only supports the NT layout (both operands K-major),
    // which maps to the column-major TN layout required by cuBLASLt
    const auto trans_a = CUBLAS_OP_T;
    const auto trans_b = CUBLAS_OP_N;

    // Matrix layouts
    // NOTES: dimensions and leading dims are in logical FP4 elements (2 elements per packed byte)
    const auto cuda_type_d = at::cuda::ScalarTypeToCudaDataType(out.scalar_type());
    const auto layout_a = get_cublaslt_layout(CUDA_R_4F_E2M1, k, n, static_cast<int>(rhs.stride(0)) * 2);
    const auto layout_b = get_cublaslt_layout(CUDA_R_4F_E2M1, k, m, static_cast<int>(lhs.stride(0)) * 2);
    const auto layout_d = get_cublaslt_layout(cuda_type_d, n, m, static_cast<int>(out.stride(0)));

    // NOTES: after the column-major operand swap, the cuBLASLt A operand is `rhs`
    call_cublaslt_api(trans_a, trans_b, layout_a, layout_b, layout_d, lhs, rhs, out, accumulate,
                      1.0f, rhs_sf.data_ptr(), lhs_sf.data_ptr());
}

static void cublaslt_bhr_hdr_bhd(const torch::Tensor& lhs, const torch::Tensor& rhs, const torch::Tensor& out,
                                 const int& b, const int& h, const int& r, const int& d) {
    const auto m = d, n = b, k = r;
    const auto trans_a = CUBLAS_OP_T;
    const auto trans_b = CUBLAS_OP_N;

    // Matrix layouts
    const auto layout_a = get_cublaslt_layout(CUDA_R_16BF, k, m, rhs.stride(1), h, rhs.stride(0));
    const auto layout_b = get_cublaslt_layout(CUDA_R_16BF, k, n, lhs.stride(0), h, lhs.stride(1));
    const auto layout_d = get_cublaslt_layout(CUDA_R_16BF, m, n, out.stride(0), h, out.stride(1));

    call_cublaslt_api(trans_a, trans_b, layout_a, layout_b, layout_d, lhs, rhs, out, false);
}


static void cublaslt_bhd_hdr_bhr(const torch::Tensor& lhs, const torch::Tensor& rhs, const torch::Tensor& out,
                                 const int& b, const int& h, const int& r, const int& d) {
    const auto m = r, n = b, k = d;
    const auto trans_a = CUBLAS_OP_N;
    const auto trans_b = CUBLAS_OP_N;

    // Matrix layouts
    const auto layout_a = get_cublaslt_layout(CUDA_R_16BF, m, k, rhs.stride(1), h, rhs.stride(0));
    const auto layout_b = get_cublaslt_layout(CUDA_R_16BF, k, n, lhs.stride(0), h, lhs.stride(1));
    const auto layout_d = get_cublaslt_layout(CUDA_R_16BF, m, n, out.stride(0), h, out.stride(1));

    call_cublaslt_api(trans_a, trans_b, layout_a, layout_b, layout_d, lhs, rhs, out, false);
}

static void cublaslt_bhd_bhr_hdr(const torch::Tensor& lhs, const torch::Tensor& rhs, const torch::Tensor& out,
                                 const uint32_t& b, const uint32_t& h, const uint32_t& r, const uint32_t& d,
                                 const bool& accumulate) {
    const auto m = r, n = d, k = b;
    const auto trans_a = CUBLAS_OP_N;
    const auto trans_b = CUBLAS_OP_T;

    // Matrix layouts
    const auto layout_a = get_cublaslt_layout(CUDA_R_16BF, m, k, rhs.stride(0), h, rhs.stride(1));
    const auto layout_b = get_cublaslt_layout(CUDA_R_16BF, n, k, lhs.stride(0), h, lhs.stride(1));
    const auto layout_d = get_cublaslt_layout(CUDA_R_32F, m, n, out.stride(1), h, out.stride(0));

    call_cublaslt_api(trans_a, trans_b, layout_a, layout_b, layout_d, lhs, rhs, out, accumulate);
}

} // namespace deep_gemm
