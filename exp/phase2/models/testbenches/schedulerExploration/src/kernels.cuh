#pragma once
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

typedef struct{
    int width;
    int height;
    int stride;
    float* elements;
} matrix;

enum class Launch_Mode{
    small,
    medium,
    large
};

__global__ void matMulKernel(matrix A, matrix B, matrix C, int rowOffset);

__global__ void normalizeKernel(matrix C, int rowOffset, int numRows);

void launchGemm(matrix d_A, matrix d_B, matrix d_C,
                int rowOffset, int numRows,
                cudaStream_t stream);

void launchNormalize(matrix d_C,
                     int rowOffset, int numRows,
                     cudaStream_t stream);
