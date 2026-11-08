// 07_cublas.cuh - NVIDIA's cuBLAS int8 GEMM, the reference to measure against.
//
// cuBLAS is column-major. A row-major M x N matrix is the same bytes as a
// column-major N x M matrix (its transpose), so we compute C^T = B^T * A^T:
//   - Bt (row-major N x K) seen column-major is K x N; with op T it is N x K = B^T
//   - A  (row-major M x K) seen column-major is K x M = A^T
//   - C  (row-major M x N) seen column-major is N x M = C^T
// This "TN" form is the layout cuBLAS int8 kernels support most widely.
#pragma once
#include <cublas_v2.h>
#include "../common.cuh"

namespace cublas_k {
inline cublasHandle_t handle() {
    static cublasHandle_t h = nullptr;
    if (!h) cublasCreate(&h);
    return h;
}
inline cublasStatus_t& last_status() {
    static cublasStatus_t s = CUBLAS_STATUS_SUCCESS;
    return s;
}
}

inline bool supports_cublas(int M, int N, int K) {
    // int8 cuBLAS needs dimensions and leading dimensions that are multiples of 4
    return M % 4 == 0 && N % 4 == 0 && K % 4 == 0;
}

inline void launch_cublas(const GemmArgs& g) {
    const int32_t alpha = 1, beta = 0;
    cublas_k::last_status() = cublasGemmEx(
        cublas_k::handle(), CUBLAS_OP_T, CUBLAS_OP_N,
        g.N, g.M, g.K,
        &alpha,
        g.Bt, CUDA_R_8I, g.K,
        g.A, CUDA_R_8I, g.K,
        &beta,
        g.C, CUDA_R_32I, g.N,
        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
}
