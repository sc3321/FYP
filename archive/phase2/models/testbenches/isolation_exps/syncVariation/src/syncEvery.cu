#include <cstdio>
#include <cstdlib>
#include <assert.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <unistd.h>

#define GLOBSIZE 8192 
#define ITERS 10000000

float randomNumber = 6.7f;

void write_marker(const char* label){
    write(2, label, __builtin_strlen(label));
}

void log_time(const char* label, int iterationNumber){
   struct timespec startTime;
   clock_gettime(CLOCK_MONOTONIC,&startTime);
   fprintf(stderr, "[%ld.%09ld] MARKER: %s\n Iteration Number: %d\n", startTime.tv_sec, startTime.tv_nsec, label, iterationNumber);
   fflush(stderr);
}

// SAXPY- single precision A times X plus Y.

__global__ void saxpy(float* a, float* b, int vecSize, float randomNumber){
    
    int workindex = threadIdx.x + blockIdx.x * blockDim.x;
    if(workindex < vecSize){
        for(int i = 0; i < 10000000; ++i){
            a[workindex] = (0.0001f * a[workindex]) + a[workindex];
        }
        b[workindex] = a[workindex] + randomNumber;
    }
    
}

int main(){ 
    
    cudaStream_t expStream;
    cudaStreamCreate(&expStream);
    

    float* x = nullptr;
    float* y = nullptr;
    cudaMallocHost(&x, GLOBSIZE*sizeof(float));
    cudaMallocHost(&y, GLOBSIZE*sizeof(float));
    
    for(int iteration = 0; iteration < GLOBSIZE; iteration++){
        x[iteration] = 1.0f;
        y[iteration] = 1.0f;
    }
    float* d_x = nullptr; 
    float* d_y = nullptr;
    cudaMalloc(&d_x, GLOBSIZE*sizeof(float));
    cudaMalloc(&d_y, GLOBSIZE*sizeof(float));
   
    int blockSize = 256;
    int gridSize = (GLOBSIZE + blockSize - 1) / blockSize;
    
    cudaMemcpyAsync(d_x, x, GLOBSIZE * sizeof(float), cudaMemcpyHostToDevice, expStream);
    cudaMemcpyAsync(d_y, y, GLOBSIZE * sizeof(float), cudaMemcpyHostToDevice, expStream);
    for(int iteration = 0; iteration < ITERS; iteration++){
        saxpy<<<gridSize, blockSize,0, expStream>>>(d_x, d_y, GLOBSIZE, randomNumber);
        
        write_marker("SYNC_START\n");
        cudaStreamSynchronize(expStream);
        write_marker("SYNC_END\n");

    }
    cudaMemcpyAsync(y, d_y, GLOBSIZE * sizeof(float), cudaMemcpyDeviceToHost, expStream);
    
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
