#include "../include/patterns.h"
#include <cstring>

void Patterns::device_alloc(int bytes, int iterations, int* base_array){
  Backend* DuT = new Backend();
  for(int i = 0; i < iterations; ++i){
      int* dev_ptr = nullptr;
      DuT->dev_malloc();
      DuT->dev_free();
  }
}


void Patterns::device_baseline(int bytes, int iterations, int* base_array){
    Backend* DuT = new Backend();
    int* bptr = (int*)std::malloc(bytes * sizeof(int));
    memcpy(bptr, base_array, bytes * sizeof(int));
    for(int i = 0; i < iterations; ++i){
        bptr[i]++;
    }
}


void Patterns::device_kernels(int bytes, int iterations, int* base_array){
    int* d = nullptr;
    Backend* DuT = new Backend();
    DuT->dev_malloc();
    DuT->dev_memcpy();
//      HIP_CHECK(hipMalloc(&d, N * sizeof(int)));
 //     HIP_CHECK(hipMemcpy(d, h, N * sizeof(int), hipMemcpyHostToDevice));
      const int threads_per_block = 256;
      const int blocks = (bytes + threads_per_block - 1) / threads_per_block;
      for (int it = 0; it < iterations; it++) {
       DuT->dev_addone();
       DuT->dev_sync();
       DuT->dev_memcpy();
       DuT->dev_free();
       //   HIP_CHECK(hipDeviceSynchronize());

      //HIP_CHECK(hipMemcpy(h, d, N * sizeof(int), hipMemcpyDeviceToHost));
      //HIP_CHECK(hipFree(d));
      std::printf("kernels finished. Configuration: bytes : %d, iterations: %d, Output is: %d. \n", bytes, iterations, h[0]);  
      assert(base_array[0] == iterations);

}





