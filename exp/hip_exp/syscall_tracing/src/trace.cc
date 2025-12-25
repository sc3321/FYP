#include <hip/hip_enums.h>
#include <hip/hip_runtime.h>
#include <assert.h>
#include <cstdint>

#ifdef USE_HIP
__global__ void add_one(int* data, int N){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < N){
            data[i] += 1;
        }
}
#endif

#define HIP_CHECK(call)               \
    do {                              \
        hipError_t _err = (call);      \
        if(_err != hipSuccess){        \
            std::fprintf(stderr,"HIP error at %s:%d: %s -> %s\n", __FILE__, __LINE__, #call, hipGetErrorString(_err));             \
            return 1;                 \
        }                             \
                                      \
    } while(0)                        \

int main(int argc, char *argv[]){
   
    if(argc != 3){
        printf("Expected 3 arguments \n");
        return 0;
    } 
    
    int bytes       = atoi(argv[1]);
    int iterations  = atoi(argv[2]);
    
    HIP_CHECK(hipFree(nullptr));

    const int N = bytes / sizeof(int);
    if (N <= 0) { std::fprintf(stderr, "N=%d (bytes=%d) too small\n", N, bytes); std::exit(1); }
    int *h = (int*)std::malloc(N * sizeof(int));
    if(h == NULL){
        return 0;
    }
    for(int i = 0; i < N; ++i){
        h[i] = i;
    } 

    #ifdef BASELINE
        std::printf("Baseline syscall run. Configuration: bytes: %d, iterations: %d. Printing last element of h...: %d\n",bytes, iterations, h[N-1]);
    #endif
    

    #ifdef ALLOC
        for(int it = 0; it < iterations; ++it){
            int* dev_ptr = nullptr; // allocating device pointer 
            HIP_CHECK(hipMalloc(&dev_ptr, N * sizeof(int)));
            // hipMemcpy(dev_ptr, h, N * sizeof(int), hipMemcpyHostToDevice);
            // hipMemcpy(h, dev_ptr, N * sizeof(int), hipMemcpyDeviceToHost);
            HIP_CHECK(hipFree(dev_ptr));
    }
        std::printf("alloc_copy finished. Configuration: bytes : %d, iterations: %d. \n", bytes, iterations);  
    #endif // ALLOC
   

    #ifdef MODE_KERNELS
      int* d = nullptr;
      HIP_CHECK(hipMalloc(&d, N * sizeof(int)));
      HIP_CHECK(hipMemcpy(d, h, N * sizeof(int), hipMemcpyHostToDevice));
      const int threads_per_block = 256;
      const int blocks = (N + threads_per_block - 1) / threads_per_block;
      for (int it = 0; it < iterations; it++) {
        hipLaunchKernelGGL(add_one, dim3(blocks), dim3(threads_per_block), 0, 0, d, N);
      }
      HIP_CHECK(hipDeviceSynchronize());

      HIP_CHECK(hipMemcpy(h, d, N * sizeof(int), hipMemcpyDeviceToHost));
      HIP_CHECK(hipFree(d));
      std::printf("kernels finished. Configuration: bytes : %d, iterations: %d, Output is: %d. \n", bytes, iterations, h[0]);  
      assert(h[0] == iterations);
    #endif

    std::free(h);

    return 0;

}
