// 06_tensor_core.cuh - int8 tensor cores through the WMMA API.
//
// Tensor cores are dedicated matrix-multiply units: one warp-wide
// mma_sync computes a whole 16x16x16 int8 tile product (4096 multiply-
// accumulates) in hardware - a small systolic-style array inside every SM.
//
// Each block of 4 warps computes a 64x64 tile of C; each warp owns a 32x32
// sub-tile held as 2x2 accumulator fragments. Per K step of 32, the block
// stages A and Bt into shared memory with 16-byte loads, split into two
// 16-wide K halves so every fragment pointer is 256-bit aligned, as
// load_matrix_sync requires.
//
// Requires M % 64 == 0, N % 64 == 0, K % 32 == 0 and compute capability 7.2+.
#pragma once
#include <mma.h>
#include "../common.cuh"

namespace tc {
constexpr int BM = 64, BN = 64, BK = 32;
constexpr int THREADS = 128;   // 4 warps
}

__global__ void __launch_bounds__(tc::THREADS)
gemm_tensor_core(const int8_t* A, const int8_t* Bt, int32_t* C, int M, int N, int K) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 720
    using namespace nvcuda;
    using namespace tc;

    // [K half][row][16 bytes]: each row of a half is one 16-byte vector
    __shared__ __align__(32) int8_t As[2][BM][16];
    __shared__ __align__(32) int8_t Bs[2][BN][16];

    const int warp = threadIdx.x / 32;
    const int wm = warp / 2, wn = warp % 2;          // warp's 32x32 position in the block tile
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0);

    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b[2];

    // thread t loads row t/2, K half t%2 of both tiles (one 16-byte vector each)
    const int lr = threadIdx.x / 2, lh = threadIdx.x % 2;

    for (int k0 = 0; k0 < K; k0 += BK) {
        *reinterpret_cast<int4*>(&As[lh][lr][0]) =
            *reinterpret_cast<const int4*>(A + (rowBase + lr) * K + k0 + 16 * lh);
        *reinterpret_cast<int4*>(&Bs[lh][lr][0]) =
            *reinterpret_cast<const int4*>(Bt + (colBase + lr) * K + k0 + 16 * lh);
        __syncthreads();

#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], &As[h][wm * 32 + i * 16][0], 16);
#pragma unroll
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], &Bs[h][wn * 32 + j * 16][0], 16);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(C + (rowBase + wm * 32 + i * 16) * N + colBase + wn * 32 + j * 16,
                                    acc[i][j], N, wmma::mem_row_major);
#endif
}

inline bool supports_tensor_core(int M, int N, int K) {
    int dev = 0, major = 0, minor = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    bool arch_ok = major > 7 || (major == 7 && minor >= 2);
    return arch_ok && M % tc::BM == 0 && N % tc::BN == 0 && K % tc::BK == 0;
}

inline void launch_tensor_core(const GemmArgs& g) {
    dim3 grid(g.N / tc::BN, g.M / tc::BM);
    gemm_tensor_core<<<grid, tc::THREADS>>>(g.A, g.Bt, g.C, g.M, g.N, g.K);
}
