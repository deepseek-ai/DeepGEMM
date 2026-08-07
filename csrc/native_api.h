#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum deep_gemm_native_dtype {
    DEEP_GEMM_NATIVE_FLOAT16 = 0,
    DEEP_GEMM_NATIVE_BFLOAT16 = 1,
    DEEP_GEMM_NATIVE_FLOAT32 = 2,
} deep_gemm_native_dtype;

typedef struct deep_gemm_native_tensor_view {
    void* data;
    int64_t rows;
    int64_t cols;
    int64_t row_stride;
    deep_gemm_native_dtype dtype;
} deep_gemm_native_tensor_view;

// Computes D[M,N] = A[M,K] @ B[N,K]^T for row-major tensors.
// The caller owns all storage and must keep it alive until the CUDA work
// completes. `stream` is a cudaStream_t passed as an opaque pointer.
int deep_gemm_native_cublaslt_gemm_nn(
    const deep_gemm_native_tensor_view* a,
    const deep_gemm_native_tensor_view* b,
    const deep_gemm_native_tensor_view* d,
    void* stream,
    int accumulate);

// Returns a thread-local diagnostic for the most recent non-zero return.
const char* deep_gemm_native_last_error(void);

#ifdef __cplusplus
}
#endif
