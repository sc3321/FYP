#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>

#define ITERS 10000
#define ALLOC 10000

static void mark(const char* s) {
    write(2, s, __builtin_strlen(s));
    write(2, "\n", 1);
}

static void check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

__global__ void saxpy(float* a, float* b, int vecSize) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadId < vecSize) {
        for (int i = 0; i < ITERS; ++i) {
            a[threadId] = a[threadId] + 6.7f;
        }
        b[threadId] = 2.0f * a[threadId];
    }
}

int main() {
    mark("MARK: PROGRAM_START");

    float* hostA = nullptr;
    float* hostB = nullptr;
    float* devA = nullptr;
    float* devB = nullptr;
    cudaStream_t expStream;

    mark("MARK: BEFORE_STREAM_CREATE");
    check(cudaStreamCreate(&expStream), "cudaStreamCreate");
    mark("MARK: AFTER_STREAM_CREATE");

    mark("MARK: BEFORE_HOST_ALLOC");
    check(cudaMallocHost(&hostA, ALLOC * sizeof(float)), "cudaMallocHost hostA");
    check(cudaMallocHost(&hostB, ALLOC * sizeof(float)), "cudaMallocHost hostB");
    mark("MARK: AFTER_HOST_ALLOC");

    mark("MARK: BEFORE_DEVICE_ALLOC");
    check(cudaMalloc(&devA, ALLOC * sizeof(float)), "cudaMalloc devA");
    check(cudaMalloc(&devB, ALLOC * sizeof(float)), "cudaMalloc devB");
    mark("MARK: AFTER_DEVICE_ALLOC");

    for (int i = 0; i < ALLOC; ++i) {
        hostA[i] = 3.14f;
        hostB[i] = 2.76f;
    }

    mark("MARK: BEFORE_H2D");
    check(cudaMemcpyAsync(devA, hostA, ALLOC * sizeof(float), cudaMemcpyHostToDevice, expStream), "H2D devA");
    check(cudaMemcpyAsync(devB, hostB, ALLOC * sizeof(float), cudaMemcpyHostToDevice, expStream), "H2D devB");
    check(cudaStreamSynchronize(expStream), "sync after H2D");
    mark("MARK: AFTER_H2D");

    int threadsPerBlock = 256;
    int numBlocks = (ALLOC + threadsPerBlock - 1) / threadsPerBlock;

    mark("MARK: BEFORE_KERNEL1");
    saxpy<<<numBlocks, threadsPerBlock, 0, expStream>>>(devA, devB, ALLOC);
    check(cudaGetLastError(), "kernel1 launch");
    mark("MARK: AFTER_KERNEL1_LAUNCH");

    mark("MARK: BEFORE_SYNC1");
    check(cudaStreamSynchronize(expStream), "sync1");
    mark("MARK: AFTER_SYNC1");

    mark("MARK: BEFORE_KERNEL2");
    saxpy<<<numBlocks, threadsPerBlock, 0, expStream>>>(devA, devB, ALLOC);
    check(cudaGetLastError(), "kernel2 launch");
    mark("MARK: AFTER_KERNEL2_LAUNCH");

    mark("MARK: BEFORE_SYNC2");
    check(cudaStreamSynchronize(expStream), "sync2");
    mark("MARK: AFTER_SYNC2");

    mark("MARK: BEFORE_D2H");
    check(cudaMemcpyAsync(hostA, devA, ALLOC * sizeof(float), cudaMemcpyDeviceToHost, expStream), "D2H hostA");
    check(cudaMemcpyAsync(hostB, devB, ALLOC * sizeof(float), cudaMemcpyDeviceToHost, expStream), "D2H hostB");
    check(cudaStreamSynchronize(expStream), "sync after D2H");
    mark("MARK: AFTER_D2H");

    mark("MARK: BEFORE_CLEANUP");
    check(cudaFree(devA), "cudaFree devA");
    check(cudaFree(devB), "cudaFree devB");
    check(cudaFreeHost(hostA), "cudaFreeHost hostA");
    check(cudaFreeHost(hostB), "cudaFreeHost hostB");
    check(cudaStreamDestroy(expStream), "cudaStreamDestroy");
    mark("MARK: PROGRAM_END");

    return 0;
}
