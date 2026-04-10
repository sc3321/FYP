#include <__clang_cuda_builtin_vars.h>
#include <cstdlib>
#include <filesystem>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thread>

#define BLOCK_SIZE 16

typedef struct{
    int width;
    int height;
    int stride;
    float* elements;
} matrix;

__device__ float getElement(const matrix A, int row, int col){
    return A.elements[row * A.stride + col];
}

__device__ void setElement(const matrix A, int row, int col, float val){
     A.elements[row * A.stride + col] = val;
}

//row and col refer to TILE row and col
__device__ matrix getSubMatrix(matrix A, int row, int col)
{
    matrix aSub;
    aSub.width  = BLOCK_SIZE;
    aSub.height = BLOCK_SIZE;
    aSub.stride = A.stride;
    aSub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return aSub;
}

// forward declaration
__global__ void matMulKernel(const matrix A, const matrix B, matrix C);

void MatMul(const matrix A, const matrix B, matrix C)
{
    // Load A and B to device memory
    matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
    cudaMemcpyHostToDevice);
    // Allocate C in device memory
    matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    matMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

__global__ void matMulKernel(matrix A, matrix B, matrix C){

   int blockRow = blockIdx.y;
   int blockCol = blockIdx.x;

   int row = threadIdx.x;
   int col = threadIdx.y;

   matrix cSub = getSubMatrix(C, blockRow, blockCol);
   float cValue;

   for(int m = 0; m < A.height / BLOCK_SIZE; ++m){
        matrix aSub = getSubMatrix(A, blockRow, m);
        matrix bSub = getSubMatrix(B, m, blockCol);
        
        __shared__ float a_sub[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float b_sub[BLOCK_SIZE][BLOCK_SIZE];

        a_sub[row][col] = getElement(aSub, row, col);
        b_sub[row][col] = getElement(bSub, row, col);

        __syncthreads();

        for(int e = 0; e < BLOCK_SIZE; ++e){
            cValue += a_sub[row][e] * b_sub[e][col];
            __syncthreads();
        }
        setElement(cSub, row, col, cValue);
   }    

}














