#include <hip/hip_runtime.h>
#include <cstdio>
#include <iterator>




#ifdef USE_HIP
__global__ void add_one(int* data){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < 256){
            data[i] += 1;
        }
}
#endif

#ifndef KERNEL_ITERS
#define KERNEL_ITERS 1
#endif

#ifndef ALLOC_ITERS
#define ALLOC_ITERS 1
#endif


int main(){

    hipFree(nullptr);

    const int N = 256;
    int *h = (int*)std::malloc(N * sizeof(int));
    for(int i = 0; i < N; ++i){
        h[i] = i;
    } 

    #ifdef BASELINE
        std::printf("Baseline syscall run. Printing first element of h...: %d", h[0]);
    #endif
    

    #ifdef ALLOC
       int* dev_ptr = nullptr; // allocating device pointer 
       hipMalloc(&dev_ptr, N * sizeof(int));
       hipMemcpy(dev_ptr, h, N * sizeof(int), hipMemcpyHostToDevice);
       hipMemcpy(h, dev_ptr, N * sizeof(int), hipMemcpyDeviceToHost);
       hipFree(dev_ptr);
       std::printf("alloc_copy finished");  
    #endif // ALLOC
   

    #ifdef MODE_KERNELS
      int* d = nullptr;
      hipMalloc(&d, N * sizeof(int));
      hipMemcpy(d, h, N * sizeof(int), hipMemcpyHostToDevice);

      for (int it = 0; it < KERNEL_ITERS; it++) {
        hipLaunchKernelGGL(add_one, dim3(1), dim3(256), 0, 0, d);
      }
      hipDeviceSynchronize();

      hipMemcpy(h, d, N * sizeof(int), hipMemcpyDeviceToHost);
      hipFree(d);
      std::printf("kernels iters=%d out=%d\n", KERNEL_ITERS, h[0]);
    #endif

    std::free(h);

    return 0;











}
