#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// CUDA Kernel to transform the matrix
__global__ void transform_matrix(int m, int n, float *matrix) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        int index = row * n + col;
        matrix[index] = __powf(matrix[index], row + 1);
    }
}

// Function to print the matrix
void print_matrix(float *matrix, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", matrix[i * n + j]);
        }
        printf("\n");
    }
}

int main() {
    int m, n;

    // Input matrix dimensions
    printf("Enter matrix dimensions (M N): ");
    scanf("%d %d", &m, &n);

    // Allocate host matrix
    float *h_matrix = (float*)malloc(m * n * sizeof(float));

    // Read matrix elements
    printf("Enter matrix elements:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%f", &h_matrix[i * n + j]);
        }
    }

    // Allocate device memory
    float *d_matrix;
    CHECK_CUDA(cudaMalloc((void**)&d_matrix, m * n * sizeof(float)));

    // Copy matrix to device
    CHECK_CUDA(cudaMemcpy(d_matrix, h_matrix, m * n * sizeof(float), cudaMemcpyHostToDevice));

    // Define CUDA kernel execution configuration
    dim3 blockDim(16, 16);  // 16x16 threads per block
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);

    // Launch the CUDA kernel
    transform_matrix<<<gridDim, blockDim>>>(m, n, d_matrix);
    CHECK_CUDA(cudaGetLastError());  // Check for kernel launch errors
    CHECK_CUDA(cudaDeviceSynchronize());  // Ensure kernel execution completes

    // Copy the modified matrix back to the host
    CHECK_CUDA(cudaMemcpy(h_matrix, d_matrix, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // Print transformed matrix
    printf("\nTransformed Matrix:\n");
    print_matrix(h_matrix, m, n);

    // Free memory
    free(h_matrix);
    CHECK_CUDA(cudaFree(d_matrix));

    return 0;
}

/*
SAMPLE OUTPUT:
---------------
Enter matrix dimensions (M N): 3 3
Enter matrix elements:
1 2 3
4 5 6
7 8 9

Transformed Matrix:
1.00 2.00 3.00
16.00 25.00 36.00
343.00 512.00 729.00
*/

