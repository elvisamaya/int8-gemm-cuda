// gemm_bench.cu - correctness tests and benchmarks for every int8 GEMM kernel.
//
//   ./build/gemm_bench --test                       correctness only, odd shapes
//   ./build/gemm_bench --sizes 512,1024,2048,4096 --out results
//
// For every kernel and size: run once, check the result against a CPU
// reference (exact - integer math has no rounding), then time it with CUDA
// events. Writes results/results.csv and results/device.json.
#include <algorithm>
#include <chrono>
#include <cstring>
#include <random>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "common.cuh"
#include "kernels/01_naive.cuh"

struct Kernel {
    const char* name;
    const char* description;
    bool (*supports)(int M, int N, int K);
    void (*launch)(const GemmArgs&);
};

static const Kernel KERNELS[] = {
    {"naive",       "1 thread/output, uncoalesced",        supports_any,         launch_naive},
};

// ------------------------------------------------------------------ CPU reference
static int32_t ref_entry(const std::vector<int8_t>& A, const std::vector<int8_t>& B, int N, int K, int r, int c) {
    int32_t acc = 0;
    for (int k = 0; k < K; ++k) acc += int32_t(A[size_t(r) * K + k]) * int32_t(B[size_t(k) * N + c]);
    return acc;
}

static void ref_full(const std::vector<int8_t>& A, const std::vector<int8_t>& B, std::vector<int32_t>& C,
                     int M, int N, int K) {
    std::fill(C.begin(), C.end(), 0);
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) {
            int32_t a = A[size_t(i) * K + k];
            const int8_t* brow = &B[size_t(k) * N];
            int32_t* crow = &C[size_t(i) * N];
            for (int j = 0; j < N; ++j) crow[j] += a * int32_t(brow[j]);
        }
}

// ------------------------------------------------------------------ one problem size
struct Problem {
    int M, N, K;
    std::vector<int8_t> A, B, Bt;
    std::vector<int32_t> ref;      // full reference (small sizes) or empty
    std::vector<int> sample_r, sample_c;
    std::vector<int32_t> sample_ref;
    int8_t *dA = nullptr, *dB = nullptr, *dBt = nullptr;
    int32_t* dC = nullptr;

    Problem(int M_, int N_, int K_, uint32_t seed) : M(M_), N(N_), K(K_) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist(-128, 127);
        A.resize(size_t(M) * K);
        B.resize(size_t(K) * N);
        Bt.resize(size_t(N) * K);
        for (auto& x : A) x = int8_t(dist(rng));
        for (auto& x : B) x = int8_t(dist(rng));
        A[0] = B[0] = -128;   // always include the extreme product
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n) Bt[size_t(n) * K + k] = B[size_t(k) * N + n];

        if (double(M) * N * K <= 1.1e9) {   // up to ~1024^3: check every element
            ref.resize(size_t(M) * N);
            ref_full(A, B, ref, M, N, K);
        } else {                            // larger: check 4096 random elements
            std::uniform_int_distribution<int> rr(0, M - 1), cc(0, N - 1);
            for (int s = 0; s < 4096; ++s) {
                int r = s == 0 ? M - 1 : rr(rng), c = s == 0 ? N - 1 : cc(rng);
                sample_r.push_back(r);
                sample_c.push_back(c);
                sample_ref.push_back(ref_entry(A, B, N, K, r, c));
            }
        }
        CUDA_CHECK(cudaMalloc(&dA, A.size()));
        CUDA_CHECK(cudaMalloc(&dB, B.size()));
        CUDA_CHECK(cudaMalloc(&dBt, Bt.size()));
        CUDA_CHECK(cudaMalloc(&dC, size_t(M) * N * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, B.data(), B.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBt, Bt.data(), Bt.size(), cudaMemcpyHostToDevice));
    }
    ~Problem() {
        cudaFree(dA); cudaFree(dB); cudaFree(dBt); cudaFree(dC);
    }
    GemmArgs args() const { return {dA, dB, dBt, dC, M, N, K}; }

    // returns the number of wrong elements checked
    long check() const {
        std::vector<int32_t> C(size_t(M) * N);
        CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
        long bad = 0;
        if (!ref.empty()) {
            for (size_t i = 0; i < C.size(); ++i) bad += C[i] != ref[i];
        } else {
            for (size_t s = 0; s < sample_ref.size(); ++s)
                bad += C[size_t(sample_r[s]) * N + sample_c[s]] != sample_ref[s];
        }
        return bad;
    }
};

// ------------------------------------------------------------------ helpers
static bool run_and_check(const Kernel& k, Problem& p, long& bad) {
    CUDA_CHECK(cudaMemset(p.dC, 0xFF, size_t(p.M) * p.N * sizeof(int32_t)));   // poison output
    k.launch(p.args());
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "  %s: launch failed: %s\n", k.name, cudaGetErrorString(err));
        return false;
    }
    bad = p.check();
    return true;
}

static double time_ms(const Kernel& k, Problem& p) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    k.launch(p.args());                       // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());

    // pick an iteration count that runs for roughly 200 ms (3..200 iterations)
    CUDA_CHECK(cudaEventRecord(start));
    k.launch(p.args());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float one = 0;
    CUDA_CHECK(cudaEventElapsedTime(&one, start, stop));
    int iters = std::max(3, std::min(200, int(200.0f / std::max(one, 0.001f))));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) k.launch(p.args());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return double(total) / iters;
}

static void write_device_json(const std::string& path) {
    int dev = 0;
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    int clock_khz = 0, mem_clock_khz = 0, bus_bits = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev);
    double bw_gbs = 2.0 * mem_clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;   // DDR: 2 transfers/clock
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return;
    fprintf(f,
            "{\n  \"name\": \"%s\",\n  \"compute_capability\": \"%d.%d\",\n  \"sm_count\": %d,\n"
            "  \"sm_clock_mhz\": %.0f,\n  \"memory_bandwidth_gbs\": %.1f,\n"
            "  \"shared_mem_per_block_kb\": %zu\n}\n",
            p.name, p.major, p.minor, p.multiProcessorCount, clock_khz / 1e3, bw_gbs,
            p.sharedMemPerBlock / 1024);
    fclose(f);
    printf("GPU: %s (sm_%d%d, %d SMs, %.0f MHz, %.0f GB/s)\n", p.name, p.major, p.minor,
           p.multiProcessorCount, clock_khz / 1e3, bw_gbs);
}

static std::vector<int> parse_sizes(const char* s) {
    std::vector<int> out;
    std::string str(s);
    size_t pos = 0;
    while (pos < str.size()) {
        size_t comma = str.find(',', pos);
        if (comma == std::string::npos) comma = str.size();
        out.push_back(std::stoi(str.substr(pos, comma - pos)));
        pos = comma + 1;
    }
    return out;
}

// ------------------------------------------------------------------ modes
static int run_tests() {
    // shapes chosen to hit edge cases: not multiples of the tile sizes, K = 1,
    // tall/skinny, and shapes the constrained kernels (dp4a, tensor cores) accept
    const int shapes[][3] = {{1, 1, 1},     {7, 5, 3},      {33, 65, 17},   {129, 67, 35},
                             {64, 64, 4},   {100, 36, 128}, {128, 192, 96}, {256, 128, 512},
                             {64, 64, 32},  {512, 512, 512}};
    int failures = 0, checks = 0;
    for (auto& s : shapes) {
        Problem p(s[0], s[1], s[2], 1234u + s[0] * 7 + s[1] * 13 + s[2]);
        for (const Kernel& k : KERNELS) {
            if (!k.supports(p.M, p.N, p.K)) continue;
            long bad = 0;
            bool ok = run_and_check(k, p, bad);
            ++checks;
            if (!ok || bad) {
                ++failures;
                printf("FAIL  %-12s M=%d N=%d K=%d  (%ld wrong elements)\n", k.name, p.M, p.N, p.K, bad);
            }
        }
    }
    printf("%d kernel/shape checks, %d failures\n", checks, failures);
    printf(failures ? "TEST FAILED\n" : "TEST PASSED\n");
    return failures ? 1 : 0;
}

static int run_bench(const std::vector<int>& sizes, const std::string& out) {
    mkdir(out.c_str(), 0755);
    write_device_json(out + "/device.json");
    FILE* csv = fopen((out + "/results.csv").c_str(), "w");
    fprintf(csv, "kernel,M,N,K,ms,gops,wrong\n");
    int failures = 0;
    printf("%-12s %6s %10s %10s  %s\n", "kernel", "size", "ms", "GOPS", "check");
    for (int S : sizes) {
        auto t0 = std::chrono::steady_clock::now();
        Problem p(S, S, S, 42u + S);
        double ref_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("-- %d x %d x %d (CPU reference %.1f s)\n", S, S, S, ref_s);
        for (const Kernel& k : KERNELS) {
            if (!k.supports(S, S, S)) continue;
            long bad = 0;
            if (!run_and_check(k, p, bad)) {
                fprintf(csv, "%s,%d,%d,%d,,,error\n", k.name, S, S, S);
                continue;
            }
            double ms = time_ms(k, p);
            double gops = 2.0 * S * S * double(S) / (ms * 1e6);
            failures += bad != 0;
            printf("%-12s %6d %10.3f %10.1f  %s\n", k.name, S, ms, gops, bad ? "WRONG" : "ok");
            fprintf(csv, "%s,%d,%d,%d,%.5f,%.2f,%ld\n", k.name, S, S, S, ms, gops, bad);
            fflush(csv);
        }
    }
    fclose(csv);
    printf("wrote %s/results.csv\n", out.c_str());
    return failures ? 1 : 0;
}

int main(int argc, char** argv) {
    std::vector<int> sizes = {512, 1024, 2048, 4096};
    std::string out = "results";
    bool test = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--test")) test = true;
        else if (!std::strcmp(argv[i], "--sizes") && i + 1 < argc) sizes = parse_sizes(argv[++i]);
        else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else {
            printf("usage: %s [--test] [--sizes 512,1024,...] [--out dir]\n", argv[0]);
            return 2;
        }
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        fprintf(stderr, "No CUDA GPU found. On Colab: Runtime -> Change runtime type -> GPU.\n");
        return 1;
    }
    return test ? run_tests() : run_bench(sizes, out);
}
