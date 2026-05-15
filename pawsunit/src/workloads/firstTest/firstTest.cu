#include "../../include/gpuPhaseTypes.h"
#include "../../include/eventHandler.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

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

static inline void marker(const char* msg) {
    write(STDERR_FILENO, msg, strlen(msg));
}

// ------------------------------------------------------------
// Kernels
// ------------------------------------------------------------

__global__ void init_kernel(float* x, int n, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] = value;
}

__global__ void compute_kernel(float* x, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float v = x[idx];

        for (int i = 0; i < iters; ++i) {
            v = v * 1.000001f + 0.000001f;
            v = __sinf(v);
            v = v * v + 0.1f;
        }

        x[idx] = v;
    }
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main(int argc, char** argv) {
    int n = 1 << 22;
    int decode_steps = 4;
    int lc_iters = 64;
    int be_iters = 512;

    if (argc > 1) decode_steps = std::atoi(argv[1]);

    phaseManager pm;
    pm.initPhaseManager();

    size_t bytes = n * sizeof(float);

    float* d_lc = nullptr;
    float* d_be = nullptr;

    CUDA_CHECK(cudaMalloc(&d_lc, bytes));
    CUDA_CHECK(cudaMalloc(&d_be, bytes));

    cudaStream_t lc_stream;
    cudaStream_t be_stream;

    CUDA_CHECK(cudaStreamCreateWithFlags(&lc_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&be_stream, cudaStreamNonBlocking));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // ------------------------------------------------------------
    // SETUP PHASE
    // Flat phase: useful for checking non-LC/BE setup behaviour.
    // ------------------------------------------------------------

    pm.phaseBegin("SETUP", "UNKNOWN");
    marker("MARKER_SETUP_BEGIN\n");

    init_kernel<<<blocks, threads, 0, lc_stream>>>(d_lc, n, 1.0f);
    init_kernel<<<blocks, threads, 0, be_stream>>>(d_be, n, 2.0f);

    CUDA_CHECK(cudaStreamSynchronize(lc_stream));
    CUDA_CHECK(cudaStreamSynchronize(be_stream));

    marker("MARKER_SETUP_END\n");
    pm.phaseEnd();

    // ------------------------------------------------------------
    // LC REQUEST PHASE
    //
    // Parent:
    //   LC_REQUEST
    //
    // Children:
    //   LC_PREFILL_SUBMISSION
    //   LC_PREFILL_SYNC
    //   LC_DECODE_LOOP
    //      LC_DECODE_STEP_0
    //      LC_DECODE_STEP_1
    //      ...
    //   LC_FINAL_SYNC
    // ------------------------------------------------------------

    pm.phaseBegin("LC_REQUEST", "LC");
    marker("MARKER_LC_REQUEST_BEGIN\n");

    // ------------------------------
    // LC prefill submission child
    // ------------------------------

    pm.phaseBegin("LC_PREFILL_SUBMISSION", "LC");
    marker("MARKER_LC_PREFILL_SUBMISSION_BEGIN\n");

    compute_kernel<<<blocks, threads, 0, lc_stream>>>(d_lc, n, lc_iters * 4);

    marker("MARKER_LC_PREFILL_SUBMISSION_END\n");
    pm.phaseEnd();

    // ------------------------------
    // LC prefill sync child
    // ------------------------------

    pm.phaseBegin("LC_PREFILL_SYNC", "LC");
    marker("MARKER_LC_PREFILL_SYNC_BEGIN\n");

    CUDA_CHECK(cudaStreamSynchronize(lc_stream));

    marker("MARKER_LC_PREFILL_SYNC_END\n");
    pm.phaseEnd();

    // ------------------------------------------------------------
    // BE BACKGROUND PHASE
    //
    // This is intentionally launched while LC_REQUEST is still open.
    // That tests whether your nesting and class logging are clear.
    //
    // Note: this is same process, same thread. Later you can move BE
    // into another process/thread.
    // ------------------------------------------------------------

    pm.phaseBegin("BE_BACKGROUND_PARENT", "BE");
    marker("MARKER_BE_BACKGROUND_PARENT_BEGIN\n");

    pm.phaseBegin("BE_BACKGROUND_SUBMISSION", "BE");
    marker("MARKER_BE_BACKGROUND_SUBMISSION_BEGIN\n");

    compute_kernel<<<blocks, threads, 0, be_stream>>>(d_be, n, be_iters);

    marker("MARKER_BE_BACKGROUND_SUBMISSION_END\n");
    pm.phaseEnd();

    // Do not sync BE yet. Let it overlap with later LC decode.

    marker("MARKER_BE_BACKGROUND_PARENT_END\n");
    pm.phaseEnd();

    // ------------------------------
    // LC decode loop child
    // ------------------------------

    pm.phaseBegin("LC_DECODE_LOOP", "LC");
    marker("MARKER_LC_DECODE_LOOP_BEGIN\n");

    for (int step = 0; step < decode_steps; ++step) {
        char phase_name[64];
        snprintf(phase_name, sizeof(phase_name), "LC_DECODE_STEP_%d", step);

        pm.phaseBegin(phase_name, "LC");

        char marker_begin[96];
        snprintf(marker_begin, sizeof(marker_begin),
                 "MARKER_LC_DECODE_STEP_%d_BEGIN\n", step);
        marker(marker_begin);

        compute_kernel<<<blocks, threads, 0, lc_stream>>>(d_lc, n, lc_iters);

        // Per-step sync makes each decode child a true submit+wait unit.
        CUDA_CHECK(cudaStreamSynchronize(lc_stream));

        char marker_end[96];
        snprintf(marker_end, sizeof(marker_end),
                 "MARKER_LC_DECODE_STEP_%d_END\n", step);
        marker(marker_end);

        pm.phaseEnd();
    }

    marker("MARKER_LC_DECODE_LOOP_END\n");
    pm.phaseEnd();

    // ------------------------------
    // LC final sync child
    // ------------------------------

    pm.phaseBegin("LC_FINAL_SYNC", "LC");
    marker("MARKER_LC_FINAL_SYNC_BEGIN\n");

    CUDA_CHECK(cudaStreamSynchronize(lc_stream));

    marker("MARKER_LC_FINAL_SYNC_END\n");
    pm.phaseEnd();

    marker("MARKER_LC_REQUEST_END\n");
    pm.phaseEnd();

    // ------------------------------------------------------------
    // BE FINAL SYNC PHASE
    //
    // This happens after LC_REQUEST closes.
    // ------------------------------------------------------------

    pm.phaseBegin("BE_FINAL_SYNC", "BE");
    marker("MARKER_BE_FINAL_SYNC_BEGIN\n");

    CUDA_CHECK(cudaStreamSynchronize(be_stream));

    marker("MARKER_BE_FINAL_SYNC_END\n");
    pm.phaseEnd();

    // ------------------------------------------------------------
    // CLEANUP
    // ------------------------------------------------------------

    pm.phaseBegin("CLEANUP", "UNKNOWN");

    CUDA_CHECK(cudaStreamDestroy(lc_stream));
    CUDA_CHECK(cudaStreamDestroy(be_stream));

    CUDA_CHECK(cudaFree(d_lc));
    CUDA_CHECK(cudaFree(d_be));

    pm.phaseEnd();

    printf("done\n");
    return 0;
}
