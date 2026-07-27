// sustained_load.cu
//
// Compute-bound sustained-load generator for power-limit / clock characterization.
// Deliberately NOT memory-bound (unlike SAXPY): each thread holds its working set
// in registers and performs many chained FMAs per element loaded, so throughput is
// gated by SM compute/power, not memory bandwidth. This is what actually stresses
// GPU Boost enough to produce a measurable clock response to nvmlDeviceSetPowerManagementLimit.

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

// Each thread performs itersPerLaunch chained FMAs entirely on register-resident
// values. No memory traffic inside the loop -> arithmetic intensity is effectively
// unbounded, so this is compute/power bound, not bandwidth bound.
__global__ void burnKernel(float* out, int n, int itersPerLaunch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float a = 1.0000001f + 0.0000001f * idx;
    float b = 0.9999999f - 0.0000001f * idx;
    float acc = a;

    #pragma unroll 8
    for (int i = 0; i < itersPerLaunch; ++i) {
        acc = fmaf(acc, a, b);
        acc = fmaf(acc, b, a);
        if (acc > 1e6f || acc < -1e6f) acc = a;   // keep bounded, avoid inf/nan
    }

    out[idx] = acc;   // single write per thread for the whole launch — negligible traffic
}

int main(int argc, char** argv) {
    int durationSec    = (argc > 1) ? atoi(argv[1]) : 30;
    int device         = (argc > 2) ? atoi(argv[2]) : 0;
    int itersPerLaunch = (argc > 3) ? atoi(argv[3]) : 200000;

    CHECK_CUDA(cudaSetDevice(device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("Device %d: %s\n", device, prop.name);

    const int n = 1 << 22;             // ~4M threads: comfortably saturates all SMs on a Titan X
    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;

    float* d_out;
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(float)));

    printf("grid=%d block=%d itersPerLaunch=%d duration=%ds\n",
           gridSize, blockSize, itersPerLaunch, durationSec);

    auto start = std::chrono::steady_clock::now();
    auto deadline = start + std::chrono::seconds(durationSec);

    long launchCount = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        burnKernel<<<gridSize, blockSize>>>(d_out, n, itersPerLaunch);
        launchCount++;

        // Sync periodically, not every launch: this checks the deadline and prevents
        // unbounded launch queueing, without introducing per-iteration host sync gaps
        // that would let the GPU go idle between launches.
        if (launchCount % 5 == 0) {
            CHECK_CUDA(cudaDeviceSynchronize());
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start).count();
            printf("  t=%lds launches=%ld\n", elapsed, launchCount);
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    printf("done: %ld launches in %.2fs\n", launchCount, totalMs / 1000.0);

    CHECK_CUDA(cudaFree(d_out));
    return 0;
}
