#include <cstdio>
#include <cstdlib>
#include <assert.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>

#define ITERS 8192 
#define WORK 10000000
#define SUBS 1

void write_marker(const char* label){
    write(2, label, __builtin_strlen(label));
}

__global__ void saxpy(float* a, float* b,int startIndex, int endIndex){
    
    int workindex = threadIdx.x + blockIdx.x * blockDim.x;
    int globalIndex = workindex + startIndex;
    if(globalIndex < endIndex){
       for(int i = 0; i < ITERS; ++i){
           a[globalIndex] = (0.0001f * a[globalIndex]) + a[globalIndex];
       }
       b[globalIndex] = a[globalIndex] + 3.14f;
    }
}


int main(){

    cudaStream_t expStream;
    cudaStreamCreate(&expStream);
    
    float* x = nullptr;
    float* y = nullptr;
    
    cudaMallocHost(&x, WORK*sizeof(float));
    cudaMallocHost(&y, WORK*sizeof(float));
    
    for(int i = 0; i < WORK; ++i){
        x[i] = 2.3f;
        y[i] = 4.5f;
    }

    float* d_X;
    float* d_Y;

    cudaMalloc(&d_X, WORK*sizeof(float));
    cudaMalloc(&d_Y, WORK*sizeof(float));
    
    cudaMemcpyAsync(d_X, x, WORK*sizeof(float), cudaMemcpyHostToDevice, expStream);
    
    int startIndex = 0;
    int blockSize = 256;
    int submissionSize = WORK / SUBS; 
    int gridSize = (submissionSize + blockSize - 1)/ blockSize; 
   
    for(int subIndex = 0; subIndex < SUBS; ++subIndex){
        startIndex = subIndex * submissionSize;
        int endIndex = startIndex + submissionSize;
        
        write_marker("SUB_START");
        saxpy<<<gridSize, blockSize,0, expStream>>>(d_X, d_Y, startIndex, endIndex);
        write_marker("SUB_END");

    }
    
    cudaMemcpyAsync(y, d_Y, WORK * sizeof(float), cudaMemcpyDeviceToHost, expStream);
    cudaStreamSynchronize(expStream);
        
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFreeHost(x);
    cudaFreeHost(y);
    
    return 0;
}

