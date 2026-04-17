#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thread>
#include "kernels.cuh"
#include <vector>
#include <unistd.h>

#define ITERS 500
#define ROWS 512 
#define COLS 512 

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)


enum class syncMode {
    perIter,
    final,
    none
};

struct Config{
    int iters               = ITERS;
    int rows                = ROWS;
    int cols                = COLS;
    int warmup              = 10;
    Launch_Mode launch_mode = Launch_Mode::large;
    syncMode sync_mode      = syncMode::perIter;
    bool async_copy         = false;
};


double randMToN(double M, double N)
{
    return M + (rand() / ( RAND_MAX / (N-M) ) ) ;  
}

static void printMarker(const char* label, int iter) {
    
    char buffer[strlen(label)*2];
    int len = snprintf(buffer, sizeof(buffer),"%s %d\n", label, iter);
    if(len > 0){
        write(2, buffer, len);
    } 
}

static int chunkRows(const Config& cfg) {
    switch (cfg.launch_mode) {
        case Launch_Mode::large:  return cfg.rows;
        case Launch_Mode::medium: return 64;
        case Launch_Mode::small:  return 16;
        case Launch_Mode::tiny:   return 8;
        case Launch_Mode::micro:  return 1;
    }
    return cfg.rows;
}


int main(int argc, char** argv){
    Config cfg;

    if (argc > 1) {
        if (std::strcmp(argv[1], "small") == 0) cfg.launch_mode = Launch_Mode::small;
        if (std::strcmp(argv[1], "medium") == 0) cfg.launch_mode = Launch_Mode::medium;
        if (std::strcmp(argv[1], "large") == 0) cfg.launch_mode = Launch_Mode::large;
        if (std::strcmp(argv[1], "tiny") == 0) cfg.launch_mode = Launch_Mode::tiny;
        if (std::strcmp(argv[1], "micro") == 0) cfg.launch_mode = Launch_Mode::micro;
    }
    if (argc > 2) {
        if (std::strcmp(argv[2], "per_iter") == 0) cfg.sync_mode = syncMode::perIter;
        if (std::strcmp(argv[2], "final") == 0) cfg.sync_mode = syncMode::final;
        if (std::strcmp(argv[2], "none") == 0) cfg.sync_mode = syncMode::none;
    }
   	if (argc > 3) cfg.iters = std::atoi(argv[3]);
	if (argc > 4) cfg.rows  = std::atoi(argv[4]);
	if (argc > 5) cfg.cols  = std::atoi(argv[5]); 

    matrix A;
    matrix B;
    matrix C;
    
    A.height = cfg.rows;
    A.width = cfg.cols;
    B.height = cfg.rows;
    B.width = cfg.cols;
    C.height = cfg.rows;
    C.width = cfg.cols;
    A.stride = A.width;
    B.stride = B.width;
    C.stride = C.width;
	
	int matrixSize = cfg.rows * cfg.cols;
    
    A.elements = (float*)std::malloc(matrixSize * sizeof(float));
    B.elements = (float*)std::malloc(matrixSize * sizeof(float));
    C.elements = (float*)std::malloc(matrixSize * sizeof(float));
 
    for(int i = 0; i < matrixSize; ++i){
       A.elements[i] = randMToN(0.0f, 1.0f); 
       B.elements[i] = randMToN(0.0f, 1.0f); 
       C.elements[i] = 0.0f;
    }
    
    matrix d_A{A.width, A.height, A.stride, nullptr};
    matrix d_B{B.width, B.height, B.stride, nullptr};
    matrix d_C{C.width, C.height, C.stride, nullptr};

    cudaMalloc(&d_A.elements, matrixSize * sizeof(float)); 
    cudaMalloc(&d_B.elements, matrixSize * sizeof(float)); 
    cudaMalloc(&d_C.elements, matrixSize * sizeof(float)); 
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    CUDA_CHECK(cudaMemcpyAsync(d_A.elements, A.elements, matrixSize*sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B.elements, B.elements, matrixSize*sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(d_C.elements, 0, matrixSize*sizeof(float), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream)); 

   	const int rows_per_launch = chunkRows(cfg);
	for (int iter = 0; iter < cfg.warmup + cfg.iters; ++iter) {
        bool measure = (iter >= cfg.warmup);
        int logical_iter = iter - cfg.warmup;

        if (measure) printMarker("ITER_START", logical_iter);

        // Phase A: optional setup/reset for this iteration
        if (measure) printMarker("PHASE_A_START", logical_iter);
        CUDA_CHECK(cudaMemsetAsync(d_C.elements, 0, matrixSize * sizeof(float), stream));
        if (measure) printMarker("PHASE_A_END", logical_iter);

        // Phase B: GEMM launches
        if (measure) printMarker("PHASE_B_START", logical_iter);
        for (int row0 = 0; row0 < cfg.rows; row0 += rows_per_launch) {
            int numRows = rows_per_launch;
            if (row0 + numRows > cfg.rows) {
                numRows = cfg.rows - row0;
            }

            if (measure) printMarker("SUBMIT_GEMM_START", logical_iter);
            launchGemm(d_A, d_B, d_C, row0, numRows, stream);
            if (measure) printMarker("SUBMIT_GEMM_END", logical_iter);
        }
        if (measure) printMarker("PHASE_B_END", logical_iter);

        // Phase C: normalization launches
        if (measure) printMarker("PHASE_C_START", logical_iter);
        for (int row0 = 0; row0 < cfg.rows; row0 += rows_per_launch) {
            int numRows = rows_per_launch;
            if (row0 + numRows > cfg.rows) {
                numRows = cfg.rows - row0;
            }

            if (measure) printMarker("SUBMIT_NORM_START", logical_iter);
            launchNormalize(d_C, row0, numRows, stream);
            if (measure) printMarker("SUBMIT_NORM_END", logical_iter);
        }
        if (measure) printMarker("PHASE_C_END", logical_iter);

        // Sync policy
        if (cfg.sync_mode == syncMode::perIter) {
            if (measure) printMarker("SYNC_START", logical_iter);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (measure) printMarker("SYNC_END", logical_iter);
        }

        if (measure) printMarker("ITER_END", logical_iter);
    }

    if (cfg.sync_mode == syncMode::final) {
        printMarker("FINAL_SYNC_START", -1);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        printMarker("FINAL_SYNC_END", -1);
    }

    CUDA_CHECK(cudaMemcpy(C.elements, d_C.elements,
                          matrixSize * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_A.elements));
    CUDA_CHECK(cudaFree(d_B.elements));
    CUDA_CHECK(cudaFree(d_C.elements));

    std::free(A.elements);
    std::free(B.elements);
    std::free(C.elements);

    return 0;	
}
