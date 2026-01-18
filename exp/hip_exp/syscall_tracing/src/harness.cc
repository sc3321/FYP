#include "../include/harness.h"
#include "../include/patterns.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static bool streq(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

int run(int bytes, int iterations, const char* variant, Backend& B) {
    const int N = bytes / (int)sizeof(int);
    if (N <= 0) {
        std::fprintf(stderr, "bytes too small: bytes=%d -> N=%d\n", bytes, N);
        return 1;
    }

    int* base = (int*)std::malloc((std::size_t)N * sizeof(int));
    if (!base) {
        std::fprintf(stderr, "malloc failed for base array\n");
        return 1;
    }
    for (int i = 0; i < N; ++i) base[i] = 0;

    Patterns p;

    if (streq(variant, "baseline")) {
        p.device_baseline(B, bytes, iterations, base);
        std::free(base);
        return 0;
    }

    if (streq(variant, "alloc")) {
        p.device_alloc(B, bytes, iterations, base);
    }

    else if (streq(variant, "kernels")) {
        p.device_kernels(B, bytes, iterations, base);
    } 

    else {
        std::fprintf(stderr, "Unknown variant '%s' (expected alloc|kernels|baseline)\n", variant);
        delete &B;
        std::free(base);
        return 1;
    }

    delete &B;
    std::free(base);
    return 0;
}

