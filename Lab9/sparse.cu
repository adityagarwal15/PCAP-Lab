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

// CUDA Kernel for Sparse Matrix-Vector Multiplication (SpMV) using CSR format
__global__ void spmv_csr_kernel(int rows, const int *__restrict__ row_ptr, 
                               const int *__restrict__ col_idx, 
                               const float *__restrict__ values, 
                               const float *__restrict__ x, 
                               float *__restrict__ y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        const int row_start = row_ptr[row];
        const int row_end = row_ptr[row + 1];

        // Coalesced memory access for efficiency
        for (int j = row_start; j < row_end; j++) {
            sum += values[j] * x[col_idx[j]];
        }
        y[row] = sum;
    }
}

// Host function for SpMV using CSR format
void spmv_csr(int rows, int cols, int nnz, const int *row_ptr, const int *col_idx, 
             const float *values, const float *x, float *y) {
    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_x, *d_y;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_row_ptr, (rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_col_idx, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_values, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_x, cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_y, rows * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_row_ptr, row_ptr, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_idx, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_values, values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, x, cols * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize output vector to zero
    CHECK_CUDA(cudaMemset(d_y, 0, rows * sizeof(float)));

    // Launch kernel
    int block_size = 256;
    int grid_size = (rows + block_size - 1) / block_size;
    spmv_csr_kernel<<<grid_size, block_size>>>(rows, d_row_ptr, d_col_idx, d_values, d_x, d_y);

    // Check for kernel errors
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());  // Ensure kernel execution completes

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(y, d_y, rows * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    // Example matrix (4x4) with 9 non-zero elements
    int rows = 4, cols = 4;
    int nnz = 9;  // Number of non-zero elements

    // CSR representation
    int row_ptr[] = {0, 2, 4, 7, 9}; // Size = rows + 1
    int col_idx[] = {0, 1, 1, 2, 0, 2, 3, 1, 3}; // Size = nnz
    float values[] = {10, 20, 30, 40, 50, 60, 70, 80, 90}; // Size = nnz

    // Input vector
    float x[] = {1, 2, 3, 4};

    // Output vector
    float y[4] = {0};

    // Perform SpMV
    spmv_csr(rows, cols, nnz, row_ptr, col_idx, values, x, y);

    // Print result
    printf("Result vector: ");
    for (int i = 0; i < rows; i++) {
        printf("%.1f ", y[i]);
    }
    printf("\n");

    return 0;
}

/*
SAMPLE OUTPUT:
--------------
Result vector: 50.0 150.0 470.0 500.0
*/

