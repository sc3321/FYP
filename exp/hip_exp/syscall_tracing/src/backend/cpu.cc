#include "../../include/cpu.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>

const char* Cpu_Backend::name() const {
    return "cpu";
}

void* Cpu_Backend::dev_malloc(std::size_t bytes) {
    void* p = std::malloc(bytes);
    if (!p && bytes != 0) {
        std::fprintf(stderr, "cpu_backend: malloc failed (%zu bytes)\n", bytes);
        std::exit(1);
    }
    return p;
}

void Cpu_Backend::dev_free(void* p) {
    std::free(p);
}

void Cpu_Backend::h2d(void* d, const void* h, std::size_t bytes) {
    if (!d || !h) {
        std::fprintf(stderr, "cpu_backend: h2d null pointer (d=%p h=%p)\n", d, h);
        std::exit(1);
    }
    std::memcpy(d, h, bytes);
}

void Cpu_Backend::d2h(void* h, const void* d, std::size_t bytes) {
    if (!h || !d) {
        std::fprintf(stderr, "cpu_backend: d2h null pointer (h=%p d=%p)\n", h, d);
        std::exit(1);
    }
    std::memcpy(h, d, bytes);
}

void Cpu_Backend::sync() {
    // CPU execution is synchronous by definition
}

void Cpu_Backend::add_one(void* d, int N){
    int count = 0;
    int* bptr = (int*)d;
    for(count = 0; count < N; ++count){
        bptr[count]++;
    }
}

Backend* make_backend() {
    return new Cpu_Backend();
}
