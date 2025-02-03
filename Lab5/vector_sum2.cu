#include <stdio.h>
#include <cuda.h>

#define N 1024  // Size of the vectors (can be modified)
#define THREADS_PER_BLOCK 256  // Fixed number of threads per block

// CUDA Kernel for vector addition
__global__ void vectorAdd(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread index
    if (i < n)  // Prevent out-of-bounds access
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

    // **Calculate number of blocks**
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernel
    vectorAdd<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print the first 10 results for verification
    printf("Result (First 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_C[i]);
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

