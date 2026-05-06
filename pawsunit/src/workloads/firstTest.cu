#include "../../include/gpuPhaseTypes.h"
#include "../../include/eventHandler.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <unistd.h>   // write()
#include <string>

// ------------------------------------------------------------
// Simple marker helper.
// write(2, ...) is useful because strace can observe it.
// ------------------------------------------------------------

static inline void marker(const char* msg) {
    write(STDERR_FILENO, msg, strlen(msg));
}

// ------------------------------------------------------------
// CUDA error checking
// ------------------------------------------------------------

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                  \
    } while (0)

// ------------------------------------------------------------
// Kernels
// ------------------------------------------------------------

__global__ void vector_add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void compute_heavy_kernel(
    float* data,
    int n,
    int iters
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float x = data[idx];

        for (int i = 0; i < iters; ++i) {
            x = x * 1.000001f + 0.000001f;
            x = __sinf(x);
            x = x * x + 0.1f;
        }

        data[idx] = x;
    }
}

// ------------------------------------------------------------
// Main benchmark
// ------------------------------------------------------------

int main(int argc, char** argv) {
    int n = 1 << 22;              // about 4 million floats
    int decode_steps = 32;
    int be_iters = 512;
    int lc_iters = 64;
    bool sync_each_decode_step = true;
    
    phaseManager newManager;
    newManager.initPhaseManager();

    if (argc > 1) {
        decode_steps = std::atoi(argv[1]);
    }

    if (argc > 2) {
        sync_each_decode_step = std::atoi(argv[2]) != 0;
    }

    printf("decode_steps=%d\n", decode_steps);
    printf("sync_each_decode_step=%d\n", sync_each_decode_step);

    size_t bytes = n * sizeof(float);

    float* h_a = nullptr;
    float* h_b = nullptr;
    float* h_c = nullptr;

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    float* d_be = nullptr;

    CUDA_CHECK(cudaMallocHost(&h_a, bytes));
    CUDA_CHECK(cudaMallocHost(&h_b, bytes));
    CUDA_CHECK(cudaMallocHost(&h_c, bytes));

    for (int i = 0; i < n; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
        h_c[i] = 0.0f;
    }

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMalloc(&d_be, bytes));

    cudaStream_t lc_stream;
    cudaStream_t be_stream;

    CUDA_CHECK(cudaStreamCreateWithFlags(&lc_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&be_stream, cudaStreamNonBlocking));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // ------------------------------------------------------------
    // Phase 0: setup / warmup
    // ------------------------------------------------------------


    CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, lc_stream));
    CUDA_CHECK(cudaMemcpyAsync(d_b, h_b, bytes, cudaMemcpyHostToDevice, lc_stream));
    CUDA_CHECK(cudaMemsetAsync(d_be, 0, bytes, be_stream));

    CUDA_CHECK(cudaStreamSynchronize(lc_stream));
    CUDA_CHECK(cudaStreamSynchronize(be_stream));


    // ------------------------------------------------------------
    // Phase 1: LC prefill-like work
    //
    // This mimics a larger latency-critical batch/prefill phase.
    // ------------------------------------------------------------

    newManager.phaseBegin("LC_PREFILL_SUBMISSION", "LC");
    vector_add_kernel<<<blocks, threads, 0, lc_stream>>>(d_a, d_b, d_c, n);
    compute_heavy_kernel<<<blocks, threads, 0, lc_stream>>>(d_c, n, lc_iters);

    newManager.phaseEnd();

    newManager.phaseBegin("LC_PREFILL_SYNC", "LC");
    CUDA_CHECK(cudaStreamSynchronize(lc_stream));

    newManager.phaseEnd();
    // ------------------------------------------------------------
    // Phase 2: BE background work
    //
    // This is best-effort work. It is intentionally heavier.
    // You can experiment with whether to launch it before, during,
    // or after LC decode work.
    // ------------------------------------------------------------

    newManager.phaseBegin("BE_BACKGROUND", "BE");
    compute_heavy_kernel<<<blocks, threads, 0, be_stream>>>(d_be, n, be_iters);

    newManager.phaseEnd();
    // Note: we do not synchronize BE yet.
    // This allows BE to overlap with LC decode work.


    // ------------------------------------------------------------
    // Phase 3: LC decode-like loop
    //
    // This mimics many smaller latency-critical decode steps.
    // The sync_each_decode_step flag lets you compare:
    //
    //   1 = per-step synchronization
    //   0 = submit all decode work, synchronize once at end
    // ------------------------------------------------------------

    newManager.phaseBegin("LC_DECODE_LOOP_START", "LC");
    for (int step = 0; step < decode_steps; ++step) {
        char buf[128];


        compute_heavy_kernel<<<blocks, threads, 0, lc_stream>>>(d_c, n, lc_iters);
    }

    if (!sync_each_decode_step) {
        newManager.phaseBegin("LC_DECODE_FINAL_SYNC_START", "LC");
        CUDA_CHECK(cudaStreamSynchronize(lc_stream));
        newManager.phaseEnd();
    }

    newManager.phaseEnd();
    // ------------------------------------------------------------
    // Phase 4: synchronize BE work
    // ------------------------------------------------------------
    
    newManager.phaseBegin("BE_FINAL_SYNC_START", "BE");
    CUDA_CHECK(cudaStreamSynchronize(be_stream));
    newManager.phaseEnd();
    // ------------------------------------------------------------
    // Phase 5: copy result back
    // ------------------------------------------------------------


    CUDA_CHECK(cudaMemcpyAsync(h_c, d_c, bytes, cudaMemcpyDeviceToHost, lc_stream));
    CUDA_CHECK(cudaStreamSynchronize(lc_stream));


    printf("h_c[0] = %f\n", h_c[0]);

    CUDA_CHECK(cudaStreamDestroy(lc_stream));
    CUDA_CHECK(cudaStreamDestroy(be_stream));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_be));

    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c));

    return 0;
}
