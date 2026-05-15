// gpu_phase_worker.cu
#include "../../../include/eventHandler.h"
#include "../../../include/gpuPhaseTypes.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

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

__global__ void compute_heavy_kernel(float* data, int n, int inner_iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float x = data[idx];

        for (int i = 0; i < inner_iters; ++i) {
            x = x * 1.000001f + 0.000001f;
            x = __sinf(x);
            x = x * x + 0.1f;
        }

        data[idx] = x;
    }
}

struct Args {
    std::string klass = "LC";
    std::string mode = "lc";
    int iters = 50;
    int chunks = 16;
    int n = 1 << 22;
    int lc_inner = 64;
    int be_inner = 512;
    int sleep_us = 0;
};

static Args parse_args(int argc, char** argv) {
    Args a;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--class") && i + 1 < argc) {
            a.klass = argv[++i];
        } else if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            a.mode = argv[++i];
        } else if (!strcmp(argv[i], "--iters") && i + 1 < argc) {
            a.iters = std::atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--chunks") && i + 1 < argc) {
            a.chunks = std::atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--n") && i + 1 < argc) {
            a.n = std::atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--lc-inner") && i + 1 < argc) {
            a.lc_inner = std::atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--be-inner") && i + 1 < argc) {
            a.be_inner = std::atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--sleep-us") && i + 1 < argc) {
            a.sleep_us = std::atoi(argv[++i]);
        }
    }

    return a;
}

static void run_lc(
    phaseManager& pm,
    cudaStream_t stream,
    float* d,
    int blocks,
    int threads,
    const Args& args
) {
    for (int r = 0; r < args.iters; ++r) {
        pm.phaseBegin("LC_REQUEST", "LC");

        pm.phaseBegin("LC_PREFILL_SUBMISSION", "LC");
        compute_heavy_kernel<<<blocks, threads, 0, stream>>>(d, args.n, args.lc_inner * 4);
        pm.phaseEnd();

        pm.phaseBegin("LC_PREFILL_SYNC", "LC");
        CUDA_CHECK(cudaStreamSynchronize(stream));
        pm.phaseEnd();

        pm.phaseBegin("LC_DECODE_STEP", "LC");
        compute_heavy_kernel<<<blocks, threads, 0, stream>>>(d, args.n, args.lc_inner);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        pm.phaseEnd();

        pm.phaseEnd();

        if (args.sleep_us > 0) {
            usleep(args.sleep_us);
        }
    }
}

static void run_be_long(
    phaseManager& pm,
    cudaStream_t stream,
    float* d,
    int blocks,
    int threads,
    const Args& args
) {
    for (int i = 0; i < args.iters; ++i) {
        pm.phaseBegin("BE_LONG_BATCH", "BE");

        compute_heavy_kernel<<<blocks, threads, 0, stream>>>(d, args.n, args.be_inner * args.chunks);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        pm.phaseEnd();
    }
}

static void run_be_chunked(
    phaseManager& pm,
    cudaStream_t stream,
    float* d,
    int blocks,
    int threads,
    const Args& args
) {
    for (int i = 0; i < args.iters; ++i) {
        pm.phaseBegin("BE_CHUNKED_BATCH", "BE");

        for (int c = 0; c < args.chunks; ++c) {
            pm.phaseBegin("BE_CHUNK", "BE");

            compute_heavy_kernel<<<blocks, threads, 0, stream>>>(d, args.n, args.be_inner);
            CUDA_CHECK(cudaStreamSynchronize(stream));

            pm.phaseEnd();
        }

        pm.phaseEnd();
    }
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    fprintf(stderr,
            "class=%s mode=%s iters=%d chunks=%d n=%d lc_inner=%d be_inner=%d\n",
            args.klass.c_str(),
            args.mode.c_str(),
            args.iters,
            args.chunks,
            args.n,
            args.lc_inner,
            args.be_inner);

    phaseManager pm;
    pm.initPhaseManager();

    size_t bytes = (size_t)args.n * sizeof(float);

    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, bytes));
    CUDA_CHECK(cudaMemset(d, 0, bytes));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    int threads = 256;
    int blocks = (args.n + threads - 1) / threads;

    pm.phaseBegin("SETUP", "UNK");
    compute_heavy_kernel<<<blocks, threads, 0, stream>>>(d, args.n, 8);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    pm.phaseEnd();

    if (args.mode == "lc") {
        marker("MARKER_LC_WORKER_BEGIN\n");
        run_lc(pm, stream, d, blocks, threads, args);
        marker("MARKER_LC_WORKER_END\n");
    } else if (args.mode == "be-long") {
        marker("MARKER_BE_LONG_WORKER_BEGIN\n");
        run_be_long(pm, stream, d, blocks, threads, args);
        marker("MARKER_BE_LONG_WORKER_END\n");
    } else if (args.mode == "be-chunked") {
        marker("MARKER_BE_CHUNKED_WORKER_BEGIN\n");
        run_be_chunked(pm, stream, d, blocks, threads, args);
        marker("MARKER_BE_CHUNKED_WORKER_END\n");
    } else {
        fprintf(stderr, "Unknown mode: %s\n", args.mode.c_str());
        return 1;
    }

    pm.phaseBegin("CLEANUP", "UNK");
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d));
    pm.phaseEnd();

    return 0;
}
