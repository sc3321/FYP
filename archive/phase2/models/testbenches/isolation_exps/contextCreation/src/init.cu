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

int main(){

    mark("MARK: PROGRAM START");

    mark("MARK: Before cudaFree");
    check(cudaFree(0), "cudaFree");
    mark("MARK: After cudaFree");
    
    mark("MARK: PROGRAM END");
    return 0;
}



