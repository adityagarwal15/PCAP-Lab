#include <stdio.h>
#include <cuda_runtime.h>

// CUDA error check macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                   __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Kernel for Row-per-thread approach
__global__ void matrixMulRowPerThread(float *A, float *B, float *C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rowsA) {
        for (int col = 0; col < colsB; col++) {
            float sum = 0;
            for (int k = 0; k < colsA; k++) {
                sum += A[row * colsA + k] * B[k * colsB + col];
            }
            C[row * colsB + col] = sum;
        }
    }
}

// Kernel for Column-per-thread approach
__global__ void matrixMulColPerThread(float *A, float *B, float *C, int rowsA, int colsA, int colsB) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < colsB) {
        for (int row = 0; row < rowsA; row++) {
            float sum = 0;
            for (int k = 0; k < colsA; k++) {
                sum += A[row * colsA + k] * B[k * colsB + col];
            }
            C[row * colsB + col] = sum;
        }
    }
}

// Kernel for Element-per-thread approach
__global__ void matrixMulElementPerThread(float *A, float *B, float *C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        float sum = 0;
        for (int k = 0; k < colsA; k++) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

// Function to print matrices
void printMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    int rowsA, colsA, rowsB, colsB, choice;

    // User input for matrix dimensions
    printf("Enter rows and columns for Matrix A: ");
    scanf("%d %d", &rowsA, &colsA);
    printf("Enter rows and columns for Matrix B: ");
    scanf("%d %d", &rowsB, &colsB);

    // Check for valid matrix multiplication condition
    if (colsA != rowsB) {
        printf("Matrix multiplication not possible. Columns of A must match rows of B.\n");
        return 1;
    }

    // User selects computation approach
    printf("Select computation approach:\n");
    printf("1. Row-per-thread\n");
    printf("2. Column-per-thread\n");
    printf("3. Element-per-thread\n");
    printf("Enter choice (1-3): ");
    scanf("%d", &choice);

    // Size of matrices
    int SIZE_A = rowsA * colsA * sizeof(float);
    int SIZE_B = rowsB * colsB * sizeof(float);
    int SIZE_C = rowsA * colsB * sizeof(float);

    // Allocate memory for matrices
    float *h_A = (float *)malloc(SIZE_A);
    float *h_B = (float *)malloc(SIZE_B);
    float *h_C = (float *)malloc(SIZE_C);

    // Initialize matrices A and B
    for (int i = 0; i < rowsA * colsA; i++) h_A[i] = i % 10 + 1;
    for (int i = 0; i < rowsB * colsB; i++) h_B[i] = (i % 5) + 1;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, SIZE_A));
    CUDA_CHECK(cudaMalloc(&d_B, SIZE_B));
    CUDA_CHECK(cudaMalloc(&d_C, SIZE_C));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, SIZE_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, SIZE_B, cudaMemcpyHostToDevice));

    // Kernel launch based on choice
    if (choice == 1) { // Row-per-thread
        int threadsPerBlock = 256;
        int blocks = (rowsA + threadsPerBlock - 1) / threadsPerBlock;
        matrixMulRowPerThread<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, rowsA, colsA, colsB);
        printf("Using Row-per-Thread Approach\n");
    } 
    else if (choice == 2) { // Column-per-thread
        int threadsPerBlock = 256;
        int blocks = (colsB + threadsPerBlock - 1) / threadsPerBlock;
        matrixMulColPerThread<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, rowsA, colsA, colsB);
        printf("Using Column-per-Thread Approach\n");
    } 
    else if (choice == 3) { // Element-per-thread
        dim3 threadsPerBlock(16, 16);
        dim3 blocks((colsB + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (rowsA + threadsPerBlock.y - 1) / threadsPerBlock.y);
        matrixMulElementPerThread<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, rowsA, colsA, colsB);
        printf("Using Element-per-Thread Approach\n");
    } 
    else {
        printf("Invalid choice! Exiting...\n");
        exit(1);
    }

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, SIZE_C, cudaMemcpyDeviceToHost));

    // Print matrices
    printf("Matrix A:\n");
    printMatrix(h_A, rowsA, colsA);
    printf("Matrix B:\n");
    printMatrix(h_B, rowsB, colsB);
    printf("Result Matrix C:\n");
    printMatrix(h_C, rowsA, colsB);

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}

