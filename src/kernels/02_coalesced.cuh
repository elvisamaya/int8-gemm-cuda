// 02_coalesced.cuh - same math as the naive kernel, threads re-mapped.
//
// Now threadIdx.x selects the COLUMN. In each iteration the 32 threads of a
// warp read 32 consecutive bytes of B (one coalesced transaction) and all read
// the SAME byte of A (a broadcast), and they write 32 consecutive int32s of C.
// Nothing else changed - only which thread computes which output.
#pragma once
#include "../common.cuh"

__global__ void gemm_coalesced(const int8_t* A, const int8_t* B, int32_t* C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        int32_t acc = 0;
        for (int k = 0; k < K; ++k)
            acc += int32_t(A[row * K + k]) * int32_t(B[k * N + col]);
        C[row * N + col] = acc;
    }
}

inline void launch_coalesced(const GemmArgs& g) {
    dim3 block(32, 32);
    dim3 grid(ceil_div(g.N, 32), ceil_div(g.M, 32));
    gemm_coalesced<<<grid, block>>>(g.A, g.B, g.C, g.M, g.N, g.K);
}
