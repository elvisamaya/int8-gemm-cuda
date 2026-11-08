// 05_dp4a.cuh - packed int8 math with the dp4a instruction.
//
// __dp4a(a, b, c) treats a and b as four packed int8 values each and returns
// c + a0*b0 + a1*b1 + a2*b2 + a3*b3: four multiply-accumulates in a single
// instruction. To use it, four consecutive K values must sit in one 32-bit
// word for BOTH operands, so this kernel reads B transposed (Bt, K contiguous)
// and loads A and Bt as 32-bit words - 4x fewer load instructions as well.
//
// Requires K % 4 == 0 (the 32-bit loads must be aligned).
#pragma once
#include "../common.cuh"

namespace dp4a_k {
constexpr int BM = 64, BN = 64;
constexpr int BKW = 16;                    // 16 words = 64 int8 per K step
constexpr int TM = 4, TN = 4;
constexpr int THREADS = (BM / TM) * (BN / TN);   // 256
}

__global__ void __launch_bounds__(dp4a_k::THREADS)
gemm_dp4a(const int8_t* A, const int8_t* Bt, int32_t* C, int M, int N, int K) {
    using namespace dp4a_k;
    __shared__ int As[BKW][BM + 1];   // As[kw][m] = A[m][4kw .. 4kw+3] packed
    __shared__ int Bs[BKW][BN + 1];   // Bs[kw][n] = B[4kw .. 4kw+3][n] packed

    const int tx = threadIdx.x % (BN / TN);
    const int ty = threadIdx.x / (BN / TN);
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    int acc[TM][TN] = {};

    for (int k0 = 0; k0 < K; k0 += 4 * BKW) {
        for (int i = threadIdx.x; i < BM * BKW; i += THREADS) {
            int m = i / BKW, kw = i % BKW;
            int gr = rowBase + m, gk = k0 + 4 * kw;
            As[kw][m] = (gr < M && gk < K) ? *reinterpret_cast<const int*>(A + gr * K + gk) : 0;
        }
        for (int i = threadIdx.x; i < BN * BKW; i += THREADS) {
            int n = i / BKW, kw = i % BKW;
            int gc = colBase + n, gk = k0 + 4 * kw;
            Bs[kw][n] = (gc < N && gk < K) ? *reinterpret_cast<const int*>(Bt + gc * K + gk) : 0;
        }
        __syncthreads();

#pragma unroll
        for (int kw = 0; kw < BKW; ++kw) {
            int a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[kw][ty * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[kw][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = __dp4a(a[i], b[j], acc[i][j]);
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

inline bool supports_dp4a(int, int, int K) { return K % 4 == 0; }

inline void launch_dp4a(const GemmArgs& g) {
    using namespace dp4a_k;
    dim3 grid(ceil_div(g.N, BN), ceil_div(g.M, BM));
    gemm_dp4a<<<grid, THREADS>>>(g.A, g.Bt, g.C, g.M, g.N, g.K);
}
