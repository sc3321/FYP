#include <cstdio>
#include <cstdlib>
#include <assert.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thread>

#define GLOBSIZE 1024

__global__ void vecAdd(float* a, float* b, float* c, int vectorLength){
    int workindex = threadIdx.x + blockIdx.x * blockDim.x;
    if(workindex < vectorLength){
        c[workindex] = a[workindex] + b[workindex];
    }
}

__global__ void vecMul(float* a, float* b, float* d, int vectorLength){
    int workindex = threadIdx.x + blockIdx.x * blockDim.x;
    if(workindex < vectorLength){
        d[workindex] = a[workindex] * b[workindex];
    }
}

int main(){

    float* a = (float*)std::malloc(GLOBSIZE * sizeof(float));
    float* b = (float*)std::malloc(GLOBSIZE * sizeof(float));
    float* c = (float*)std::malloc(GLOBSIZE * sizeof(float));
    float* d = (float*)std::malloc(GLOBSIZE * sizeof(float));

    for (int i = 0; i <GLOBSIZE; i++){
        if(i % 2){
            a[i] = (float)i;
            b[i] = 0;
        }
        else{
            a[i] = 0;
            b[i] = (float)i;
        }
    }

    int threadsPerBlock = 64;
    int blocks = (GLOBSIZE + threadsPerBlock -1) / threadsPerBlock;
    
    float *d_A, *d_B, *d_C, *d_D;
    cudaMalloc(&d_A, GLOBSIZE * sizeof(float));
    cudaMalloc(&d_B, GLOBSIZE * sizeof(float));
    cudaMalloc(&d_C, GLOBSIZE * sizeof(float));
    cudaMalloc(&d_D, GLOBSIZE * sizeof(float));
    
    cudaMemcpy(d_A, a, GLOBSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b, GLOBSIZE*sizeof(float), cudaMemcpyHostToDevice);
    
    vecAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, GLOBSIZE);    
    cudaError_t e =cudaGetLastError();
    assert(e == cudaSuccess);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_C, GLOBSIZE*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < GLOBSIZE; ++i){
        assert(c[i] == a[i] + b[i]);
    }
    
    printf("all good");
    
    vecMul<<<blocks, threadsPerBlock>>>(d_A, d_B, d_D, GLOBSIZE);
    e =cudaGetLastError();
    assert(e == cudaSuccess);
    cudaDeviceSynchronize();

    cudaMemcpy(d, d_D, GLOBSIZE*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < GLOBSIZE; ++i){
        assert(d[i] == a[i] * b[i]);
    }
    
    printf("all good");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);

    return 0;
}
