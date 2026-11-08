// 03_smem_tiled.cuh - shared-memory tiling.
//
// Each 32x32 thread block computes a 32x32 tile of C. It walks along K in
// steps of 32: every thread loads ONE element of A and ONE of B into shared
// memory (fast on-chip SRAM), the block synchronizes, and then every thread
// does 32 multiply-adds reading only shared memory.
//
// Each byte loaded from global memory is now reused 32 times instead of once -
// the same data-reuse idea as a systolic array, where operands are passed
// between neighbouring PEs instead of re-read from memory.
#pragma once
#include "../common.cuh"

template <int T>
__global__ void gemm_smem(const int8_t* A, const int8_t* B, int32_t* C, int M, int N, int K) {
    __shared__ int8_t As[T][T];
    __shared__ int8_t Bs[T][T];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * T + ty;
    int col = blockIdx.x * T + tx;
    int32_t acc = 0;

    for (int k0 = 0; k0 < K; k0 += T) {
        As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : int8_t(0);
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : int8_t(0);
        __syncthreads();
#pragma unroll
        for (int k = 0; k < T; ++k)
            acc += int32_t(As[ty][k]) * int32_t(Bs[k][tx]);
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

inline void launch_smem(const GemmArgs& g) {
    constexpr int T = 32;
    dim3 block(T, T);
    dim3 grid(ceil_div(g.N, T), ceil_div(g.M, T));
    gemm_smem<T><<<grid, block>>>(g.A, g.B, g.C, g.M, g.N, g.K);
}
