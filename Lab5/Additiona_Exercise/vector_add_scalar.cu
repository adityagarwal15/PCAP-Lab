#include <stdio.h>
#include <cuda.h>

// Define vector length and scalar value
#define N 1024  // Length of the vectors
#define a 2.5   // Scalar value

// CUDA kernel for performing the operation y = a * x + y
__global__ void vectorAddScalar(float *x, float *y, int n, float scalar) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread index
    if (i < n) {  // Ensure we don't go out of bounds
        y[i] = scalar * x[i] + y[i];  // Update y based on the formula
    }
}

int main() {
    int size = N * sizeof(float);  // Size of each vector in bytes

    // Allocate memory for the vectors on host
    float *h_x = (float*)malloc(size);
    float *h_y = (float*)malloc(size);

    // Initialize vectors x and y with some values
    for (int i = 0; i < N; i++) {
        h_x[i] = i * 1.0f;  // Example values for x: [0, 1, 2, 3, ...]
        h_y[i] = i * 2.0f;  // Example values for y: [0, 2, 4, 6, ...]
    }

    // Allocate memory for the vectors on device (GPU)
    float *d_x, *d_y;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);

    // Copy vectors from host to device
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

    // Calculate the number of blocks and threads per block
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel to perform the operation y = a * x + y
    vectorAddScalar<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, N, a);

    // Copy the result from device to host
    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

    // Print the first 10 elements of the result vector y
    printf("Updated values of y (First 10 values):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_y[i]);
    }
    printf("\n");

    // Free device and host memory
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y);

    return 0;
}

