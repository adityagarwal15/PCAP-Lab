#include <stdio.h>
#include <cuda.h>
#include <math.h>

#define N 1024  // Number of elements in the array
#define THREADS_PER_BLOCK 256  // Number of threads per block

// CUDA kernel to compute sine of the angles in radians
__global__ void computeSine(float *angles, float *sineValues, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread index
    if (i < n) {  // Ensure we don't go out of bounds
        sineValues[i] = sinf(angles[i]);  // Compute sine of angle at index i
    }
}

int main() {
    int size = N * sizeof(float);

    // Allocate memory on host
    float *h_angles = (float*)malloc(size);
    float *h_sineValues = (float*)malloc(size);

    // Initialize angles in radians (e.g., [0, π/2, π, 3π/2, ...])
    for (int i = 0; i < N; i++) {
        h_angles[i] = i * (M_PI / 180.0f);  // Example: Convert i to radians
    }

    // Allocate memory on GPU
    float *d_angles, *d_sineValues;
    cudaMalloc(&d_angles, size);
    cudaMalloc(&d_sineValues, size);

    // Copy data from host to device
    cudaMemcpy(d_angles, h_angles, size, cudaMemcpyHostToDevice);

    // Calculate number of blocks needed
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch the kernel to compute sine values
    computeSine<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_angles, d_sineValues, N);

    // Copy the result back to host
    cudaMemcpy(h_sineValues, d_sineValues, size, cudaMemcpyDeviceToHost);

    // Print the first 10 results for verification
    printf("Sine of angles (First 10 values):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", h_sineValues[i]);
    }
    printf("\n");

    // Free memory
    cudaFree(d_angles);
    cudaFree(d_sineValues);
    free(h_angles);
    free(h_sineValues);

    return 0;
}

