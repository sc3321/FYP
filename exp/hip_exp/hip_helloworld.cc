// Loop iteration program for exploring GPU programming model. C program for testing 
#include <cstddef>
#include <assert.h>
#include <cstdio>
#include <iterator>
#include "/Users/shreechan/Library/HIP-CPU/include/hip/hip_runtime.h"

__global__
void kernel(const int* arr_A, const int* arr_B, int* arr_C, int n){
    
    // The following line used to for every thread to work out "Which index am I working on"
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
   
    size_t arr_max = n; 

    if(thread_idx < arr_max){
        arr_C[thread_idx] =  arr_A[thread_idx] + (thread_idx * arr_B[thread_idx]);
   }

}


 int main(){
     int A[512];
     int B[512];
     int C[512];
     
     int C_cpu[512];

     size_t n = sizeof(A)/sizeof(A[0]);

     for(int i = 0; i < n; ++i){
         A[i] = i + (n%(i+1));
         B[i] = n%(i+1);
         C_cpu[i] = A[i] + (i * B[i]);
         printf("The C_cpu value is: %d \n", C_cpu[i]);
     }


   
    // hipMalloc takes 2 params- (void**, size_t)
    // Must declare device device (GPU) pointers
    
    int* device_A;
    int* device_B;
    int* device_C;

    hipMalloc(&device_A, n * sizeof(int));
    hipMalloc(&device_B, n * sizeof(int));
    hipMalloc(&device_C, n * sizeof(int));
    
    hipMemcpy(device_A, A, n * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(device_B, B, n * sizeof(int), hipMemcpyHostToDevice);
    
    
    int threadsPerBlock = 128;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    // Given we launch t threads per block and b number of blocks, the total threads launched = t * b
    // We require t * b >= N
    // We want to only launch the minimal number of blocks, so we do that b such that t * b is as close as possible but >= N
    // tldr: instead of doing blocks = N / t = b, by adding the (threads + 1), we are doing the ceiling of the division which is what we want.  
    
    hipLaunchKernelGGL(
        kernel,
        dim3(blocks), dim3(threadsPerBlock),
        0, 0,            // shared mem, stream
        device_A, device_B, device_C, n    // kernel args
    );

    hipDeviceSynchronize();

    // 6. Copy back result
    hipMemcpy(C, device_C, n * sizeof(int), hipMemcpyDeviceToHost);
    
    // 7. Assertion checking to see if the GPU program worked as intended:
   


    for(int i=0; i < n; ++i){
        printf("Device: %d , CPU: %d \n", C[i], C_cpu[i]);
        assert(C_cpu[i] == C[i]);
    }
    
     int ret = 0;

     for(int i = 0; i < n; ++i){
         ret += C[i];
     }

    hipFree(device_A);
    hipFree(device_B);
    hipFree(device_C);


    return ret;
 }
