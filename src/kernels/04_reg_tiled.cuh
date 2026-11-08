// 04_reg_tiled.cuh - shared memory + register tiling.
//
// In the shared-memory kernel each thread computes ONE output, so every
// multiply-add needs two shared-memory reads. Here each of the 256 threads
// computes a 4x4 block of outputs: per k it reads 4 values of A and 4 of B
// into registers and does 16 multiply-adds - 2 reads per 1 multiply-add
// becomes 8 reads per 16. The block tile grows to 64x64 for more reuse.
//
// Shared-memory tiles are stored k-major (As[k][m]) so the inner loop reads
// consecutive addresses. One column of padding avoids bank conflicts on the
// transposed stores.
#pragma once
#include "../common.cuh"

namespace regtile {
constexpr int BM = 64, BN = 64, BK = 32;   // block tile
constexpr int TM = 4, TN = 4;              // per-thread tile
constexpr int THREADS = (BM / TM) * (BN / TN);   // 256
}

__global__ void __launch_bounds__(regtile::THREADS)
gemm_regtile(const int8_t* A, const int8_t* B, int32_t* C, int M, int N, int K) {
    using namespace regtile;
    __shared__ int8_t As[BK][BM + 4];
    __shared__ int8_t Bs[BK][BN + 4];

    const int tx = threadIdx.x % (BN / TN);     // 0..15 -> columns
    const int ty = threadIdx.x / (BN / TN);     // 0..15 -> rows
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    int32_t acc[TM][TN] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // cooperative loads: consecutive threads read consecutive bytes (coalesced)
        for (int i = threadIdx.x; i < BM * BK; i += THREADS) {
            int m = i / BK, k = i % BK;
            int gr = rowBase + m, gk = k0 + k;
            As[k][m] = (gr < M && gk < K) ? A[gr * K + gk] : int8_t(0);
        }
        for (int i = threadIdx.x; i < BK * BN; i += THREADS) {
            int k = i / BN, n = i % BN;
            int gk = k0 + k, gc = colBase + n;
            Bs[k][n] = (gk < K && gc < N) ? B[gk * N + gc] : int8_t(0);
        }
        __syncthreads();

#pragma unroll 8
        for (int k = 0; k < BK; ++k) {
            int32_t a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[k][ty * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int r = rowBase + ty * TM + i;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            int c = colBase + tx * TN + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
    }
}

inline void launch_regtile(const GemmArgs& g) {
    using namespace regtile;
    dim3 grid(ceil_div(g.N, BN), ceil_div(g.M, BM));
    gemm_regtile<<<grid, THREADS>>>(g.A, g.B, g.C, g.M, g.N, g.K);
}
