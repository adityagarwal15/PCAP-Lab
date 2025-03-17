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

// CUDA Kernel to compute matrix B
__global__ void compute_matrix_B(int m, int n, int *A, int *B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        int index = row * n + col;
        if (row == 0 || row == m - 1 || col == 0 || col == n - 1) {
            B[index] = A[index];  // Copy border elements as is
        } else {
            B[index] = ~A[index];  // Compute 1's complement for non-border elements
        }
    }
}

// Function to print a matrix
void print_matrix(const char *label, int *matrix, int m, int n) {
    printf("\n%s:\n", label);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%4d ", matrix[i * n + j]);
        }
        printf("\n");
    }
}

int main() {
    int m, n;

    // Input matrix dimensions
    printf("Enter matrix dimensions (M N): ");
    if (scanf("%d %d", &m, &n) != 2 || m <= 0 || n <= 0) {
        fprintf(stderr, "Invalid matrix dimensions!\n");
        return EXIT_FAILURE;
    }

    // Allocate host memory for matrices A and B
    int *h_A = (int*)malloc(m * n * sizeof(int));
    int *h_B = (int*)malloc(m * n * sizeof(int));

    if (!h_A || !h_B) {
        fprintf(stderr, "Memory allocation failed!\n");
        return EXIT_FAILURE;
    }

    // Read matrix A elements
    printf("Enter matrix A elements:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (scanf("%d", &h_A[i * n + j]) != 1) {
                fprintf(stderr, "Invalid input for matrix elements!\n");
                free(h_A);
                free(h_B);
                return EXIT_FAILURE;
            }
        }
    }

    // Allocate device memory for matrices A and B
    int *d_A, *d_B;
    CHECK_CUDA(cudaMalloc((void**)&d_A, m * n * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_B, m * n * sizeof(int)));

    // Copy matrix A to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, m * n * sizeof(int), cudaMemcpyHostToDevice));

    // Define CUDA kernel execution configuration
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);

    // Launch CUDA kernel
    compute_matrix_B<<<gridDim, blockDim>>>(m, n, d_A, d_B);
    CHECK_CUDA(cudaGetLastError());  
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy the result matrix B back to the host
    CHECK_CUDA(cudaMemcpy(h_B, d_B, m * n * sizeof(int), cudaMemcpyDeviceToHost));

    // Print matrices
    print_matrix("Input Matrix A", h_A, m, n);
    print_matrix("Output Matrix B", h_B, m, n);

    // Free memory
    free(h_A);
    free(h_B);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));

    return 0;
}
