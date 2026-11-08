// common.cuh - shared helpers for the int8 GEMM kernels.
//
// Problem: C (M x N, int32) = A (M x K, int8) * B (K x N, int8), all row-major.
// Some kernels read B transposed (Bt, N x K row-major) so that the K dimension
// is contiguous for both operands - the layout dp4a and tensor cores want.
// Transposing B is a one-time preprocessing step (weights are fixed in
// inference), so it is not included in the timings.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,      \
                    __LINE__, cudaGetErrorString(err_));                          \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

struct GemmArgs {
    const int8_t* A;   // M x K, row-major
    const int8_t* B;   // K x N, row-major
    const int8_t* Bt;  // N x K, row-major (B transposed)
    int32_t* C;        // M x N, row-major
    int M, N, K;
};

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
