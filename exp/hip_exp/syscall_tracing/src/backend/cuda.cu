#include "../../include/cuda.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <cstddef>
#include <stdio.h>

__global__ void add_one_kernel(int* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) data[i] += 1;
}


static void check_error_msg(cudaError_t err, char* funct){
    if(err != cudaSuccess){
        std::fprintf(stderr, "cuda error at %s: %s\n", funct, cudaGetErrorString(err));
    std::exit(1);
    }
        
}

const char* cuda_backend::name() const{
    return "cuda";
}

void* cuda_backend::dev_malloc(std::size_t bytes) {
    void* p = nullptr; 
    check_error_msg(cudaMalloc(&p, bytes), "cudaMalloc"); 
    return p; 
};

void  cuda_backend::dev_free(void* p) {
    check_error_msg(cudaFree(p), "cudaFree");
};

void  cuda_backend::h2d(void* d, const void* h, std::size_t bytes) {
    check_error_msg(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice), "cudaMemCpyH2D");

};

void  cuda_backend::d2h(void* h, const void* d, std::size_t bytes) {
         check_error_msg(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost), "cudaMemCpyD2H");

};

void  cuda_backend::sync() {
    check_error_msg(cudaDeviceSynchronize(), "cudaSync");
};


void cuda_backend::add_one(void* d, int N) {
    int* data = (int*)d;
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    add_one_kernel<<<blocks, threads>>>(data, N);
    check_error_msg(cudaGetLastError(), "add_one_kernel launch");
}



Backend* make_backend(){
    return new cuda_backend();
};




