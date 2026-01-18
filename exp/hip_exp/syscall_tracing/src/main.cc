#include <cstring>
#include <assert.h>
#include <stdlib.h>
#include <cstdio>
#include "../include/harness.h"
#include "../include/backend_factory.h"

int main(int argc, char *argv[]){
   
    if(argc != 4){
        printf("Expected 3 arguments \n");
        return 0;
    } 
    
    int bytes           = atoi(argv[1]);
    int iterations      = atoi(argv[2]);
    char* variant       = argv[3];    
   
    const int N = bytes / sizeof(int);
    if (N <= 0) { std::fprintf(stderr, "N=%d (bytes=%d) too small\n", N, bytes); std::exit(1); }
      return 0;

    static Backend* B = make_backend();
    
    return run(N, iterations, variant, *B);
        
}
