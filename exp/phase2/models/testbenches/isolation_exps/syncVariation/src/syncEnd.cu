#include <cstdio>
#include <cstdlib>
#include <assert.h>
#include <ctime>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/syscall.h>

//
//static inline void mark_start(void) {syscall(SYS_getpid); }
//static inline void mark_end(void)   {syscall(SYS_gettid); }

void write_marker(const char* label){
    write(2, label, __builtin_strlen(label)); 
}

void log_time(const char* label){
   struct timespec startTime;

   clock_gettime(CLOCK_REALTIME,&startTime);
   fprintf(stderr, "[%ld.%09ld] MARKER: %s\n", startTime.tv_sec, startTime.tv_nsec, label);
   fflush(stderr);
}


#define GLOBSIZE 8192 
#define ITERS 1000000
float randomNumber = 6.7f;
    

// SAXPY- single precision A times X plus Y.

__global__ void saxpy(float* a, float* b, int vecSize, float randomNumber){
    
    int workindex = threadIdx.x + blockIdx.x * blockDim.x;
    if(workindex < vecSize){
        for(int i = 0; i < 100000; ++i){
            a[workindex] =  a[workindex] + 0.000001f;
        }
    }
    b[workindex] = randomNumber * a[workindex]; 
}

int main(){ 
    
    cudaStream_t expStream;
    cudaStreamCreate(&expStream);
    

    float* x = nullptr;
    float* y = nullptr;
    cudaMallocHost(&x, GLOBSIZE*sizeof(float));
    cudaMallocHost(&y, GLOBSIZE*sizeof(float));
    
    for(int i = 0; i < GLOBSIZE; i++){
        x[i] = 1.0f;
        y[i] = 1.0f;
    }
    float* d_x = nullptr; 
    float* d_y = nullptr;
    cudaMalloc(&d_x, GLOBSIZE*sizeof(float));
    cudaMalloc(&d_y, GLOBSIZE*sizeof(float));
   
    int blockSize = 256;
    int gridSize = (GLOBSIZE + blockSize - 1) / blockSize;
    
    cudaMemcpyAsync(d_x, x, GLOBSIZE * sizeof(float), cudaMemcpyHostToDevice, expStream);
    cudaMemcpyAsync(d_y, y, GLOBSIZE * sizeof(float), cudaMemcpyHostToDevice, expStream);
    for(int i = 0; i < ITERS; i++){
        saxpy<<<gridSize, blockSize,0, expStream>>>(d_x, d_y, GLOBSIZE, randomNumber);
    }
    cudaMemcpyAsync(y, d_y, GLOBSIZE * sizeof(float),cudaMemcpyDeviceToHost, expStream);
    
    write_marker("START_SYNC\n"); 
    cudaStreamSynchronize(expStream);
    write_marker("END_SYNC\n"); 


    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
