#include "../include/patterns.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>


void Patterns::device_alloc(Backend& b, int bytes, int iterations, int* base_array) {
    for (int it = 0; it < iterations; ++it) {
        void* d = b.dev_malloc((std::size_t)bytes);        
        if(!d){ 
            std::fprintf(stderr, "dev_malloc returned null\n"); 
            std::exit(1); 
        }
        b.dev_free(d);
    }
}

void Patterns::device_baseline(Backend& b, int bytes, int iterations, int* base_array) {
    const int N = bytes / (int)sizeof(int);
    if (N <= 0) { 
        std::fprintf(stderr, "baseline: N=%d too small\n", N); 
        std::exit(1); 
    }

    int* work = (int*)std::malloc((std::size_t)N * sizeof(int));
    if (!work) { 
        std::fprintf(stderr, "baseline: malloc failed\n"); 
        std::exit(1); 
    }

    std::memcpy(work, base_array, (std::size_t)N * sizeof(int));

    for (int it = 0; it < iterations; ++it) {
        for (int i = 0; i < N; ++i) work[i] += 1;
    }

    assert(work[0] == iterations);
    std::free(work);
}

void Patterns::device_kernels(Backend& b, int bytes, int iterations, int* base_array) {
    
    const int N = bytes / (int)sizeof(int);
    if (N <= 0) { 
        std::fprintf(stderr, "kernels: N=%d too small\n", N); 
        std::exit(1); 
    }

    int* host = (int*)std::malloc((std::size_t)N * sizeof(int));
    if (!host) { 
        std::fprintf(stderr, "kernels: malloc host failed\n"); 
        std::exit(1); 
    }
    std::memcpy(host, base_array, (std::size_t)N * sizeof(int));

    void* d = b.dev_malloc((std::size_t)bytes);
    if (!d) { 
        std::fprintf(stderr, "kernels: dev_malloc returned null\n"); 
        std::exit(1); 
    }

    b.h2d(d, host, (std::size_t)bytes);

    for (int it = 0; it < iterations; ++it) {
        b.add_one(d, N);
        b.sync();
        
        if(it % 1000 == 0){
            b.d2h(base_array, d, (std::size_t)bytes);
            std::printf("progress it=%d host[0]=%d\n", it, base_array[0]);
        }


    }
    
    b.d2h(host, d, (std::size_t)bytes);
    b.dev_free(d);

    assert(host[0] == iterations);

    std::printf("kernels finished. backend=%s bytes=%d iterations=%d output=%d\n", b.name(), bytes, iterations, host[0]);

    std::free(host);
}


