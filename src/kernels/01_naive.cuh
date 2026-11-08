// 01_naive.cuh - one thread per output element.
//
// The thread -> output mapping is deliberately the "wrong" way round:
// threadIdx.x selects the ROW. The 32 threads of a warp therefore read A from
// 32 different rows (addresses K bytes apart) and write C 32 rows apart, so the
// memory system cannot combine their accesses into wide transactions
// (uncoalesced). This is the baseline every later kernel improves on.
#pragma once
#include "../common.cuh"

__global__ void gemm_naive(const int8_t* A, const int8_t* B, int32_t* C, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        int32_t acc = 0;
        for (int k = 0; k < K; ++k)
            acc += int32_t(A[row * K + k]) * int32_t(B[k * N + col]);
        C[row * N + col] = acc;
    }
}

inline bool supports_any(int, int, int) { return true; }

inline void launch_naive(const GemmArgs& g) {
    dim3 block(32, 32);
    dim3 grid(ceil_div(g.M, 32), ceil_div(g.N, 32));
    gemm_naive<<<grid, block>>>(g.A, g.B, g.C, g.M, g.N, g.K);
}
