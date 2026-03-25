#include <cstdio>
#include <cstdlib>
#include <assert.h>
#include <ctime>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>

#define ITERS       1000
#define GLOBSIZE    10000000 
#define ALLOC_SIZE  10000

void write_marker(const char* label){
    write(2, label, __builtin_strlen(label));
}

__global__ void saxpy(float* a, float* b){
    
    int workindex = threadIdx.x + blockIdx.x * blockDim.x;
    if(workindex < ALLOC_SIZE){
        for(int i = 0; i < ITERS; ++i){
           a[workindex] = (0.0001f * a[workindex]) + a[workindex];
        }
        b[workindex] = a[workindex] + 3.14f;
    }
}

int main(){

    cudaStream_t expStream;
    cudaStreamCreate(&expStream);
    
    

    float* x = nullptr;
    float* y = nullptr;
    cudaMallocHost(&x, GLOBSIZE*sizeof(float));
    cudaMallocHost(&y, GLOBSIZE*sizeof(float));
    
    for(int i = 0; i < GLOBSIZE; ++i){
        x[i] = 3.14f;
        y[i] = 6.7f;
    }


    int allocationCount = GLOBSIZE / ALLOC_SIZE;    
    
    float* d_X = nullptr;
    float* d_Y = nullptr;
    int startIndex = 0;
    int blockSize = 256;
    int gridSize = (ALLOC_SIZE + blockSize - 1) / blockSize;

    for(int i = 0; i < allocationCount; ++i){
        
        startIndex = i * ALLOC_SIZE;
       
        write_marker("START_MALLOC\n");
        cudaMalloc(&d_X, ALLOC_SIZE * sizeof(float));
        cudaMalloc(&d_Y, ALLOC_SIZE * sizeof(float));
        write_marker("END_MALLOC\n");

        cudaMemcpyAsync(d_X, &x[startIndex], ALLOC_SIZE * sizeof(float), cudaMemcpyHostToDevice, expStream);
        cudaMemcpyAsync(d_Y, &y[startIndex], ALLOC_SIZE * sizeof(float), cudaMemcpyHostToDevice, expStream);
        
        saxpy<<<gridSize, blockSize,0, expStream>>>(d_X, d_Y);

        cudaMemcpyAsync(&x[startIndex], d_X, ALLOC_SIZE * sizeof(float), cudaMemcpyDeviceToHost, expStream);
        cudaMemcpyAsync(&y[startIndex], d_Y, ALLOC_SIZE * sizeof(float), cudaMemcpyDeviceToHost, expStream);
    
        cudaStreamSynchronize(expStream);
        cudaFree(d_X);
        cudaFree(d_Y);
    }

    cudaStreamSynchronize(expStream);
    
    cudaFreeHost(x);
    cudaFreeHost(y);

    return 0;
}
