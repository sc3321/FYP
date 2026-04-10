#include "kernels.cuh"
#include <cstdio>

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

__global__ void matMulKernel(matrix A, matrix B, matrix C, int rowOffset){

   int blockRow = blockIdx.y + rowOffset / BLOCK_SIZE;
   int blockCol = blockIdx.x;

   int row = threadIdx.x;
   int col = threadIdx.y;

   matrix cSub = getSubMatrix(C, blockRow, blockCol);
   float cValue = 0.0f;   for(int m = 0; m < A.height / BLOCK_SIZE; ++m){
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
    }    
    setElement(cSub, row, col, cValue);

}

void launchGemm(matrix d_A, matrix d_B, matrix d_C,
                int rowOffset, int numRows,
                cudaStream_t stream) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((d_B.width + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (numRows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matMulKernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, rowOffset);
}


__global__ void normalizeKernel(matrix C, int rowOffset, int numRows) {
    int localRow = blockIdx.x;
    int row = rowOffset + localRow;

    if (localRow >= numRows || row >= C.height) return;

    float sum = 0.0f;
    for (int col = threadIdx.x; col < C.width; col += blockDim.x) {
        sum += C.elements[row * C.stride + col];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float norm = partial[0] + 1e-6f;

    for (int col = threadIdx.x; col < C.width; col += blockDim.x) {
        C.elements[row * C.stride + col] /= norm;
    }
}

void launchNormalize(matrix d_C,
                     int rowOffset, int numRows,
                     cudaStream_t stream) {
    dim3 block(256);
    dim3 grid(numRows);
    normalizeKernel<<<grid, block, 0, stream>>>(d_C, rowOffset, numRows);
}
