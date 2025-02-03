#include <stdio.h>
#include <cuda.h>

#define N 1024  // Size of the vectors

// CUDA Kernel for vector addition with one block of size N
__global__ void vectorAdd_BlockSizeN(float *A, float *B, float *C, int n) {
    int i = threadIdx.x;  // Single block, so we only use threadIdx.x
    if (i < n)
        C[i] = A[i] + B[i];
}

// CUDA Kernel for vector addition with N threads across multiple blocks
__global__ void vectorAdd_NThreads(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread index
    if (i < n)
        C[i] = A[i] + B[i];
}

int main() {
    int size = N * sizeof(float);

    // Allocate memory on host
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = (N - i) * 1.0f;
    }

    // Allocate memory on GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // **Approach 1: One block of size N**
    vectorAdd_BlockSizeN<<<1, N>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Result from Approach 1 (Block Size = N):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_C[i]);  // Print first 10 results for verification
    }
    printf("\n");

    // **Approach 2: N threads (Multiple Blocks)**
    int threadsPerBlock = 256;  // Number of threads per block
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd_NThreads<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Result from Approach 2 (N Threads, Multiple Blocks):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_C[i]);  // Print first 10 results for verification
    }
    printf("\n");

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

