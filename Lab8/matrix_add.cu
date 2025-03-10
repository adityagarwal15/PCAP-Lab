#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
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
__global__ void matrixAddRowPerThread(float *A, float *B, float *C, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        for (int col = 0; col < cols; col++) {
            int idx = row * cols + col;
            C[idx] = A[idx] + B[idx];
        }
    }
}

// Kernel for Column-per-thread approach
__global__ void matrixAddColPerThread(float *A, float *B, float *C, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        for (int row = 0; row < rows; row++) {
            int idx = row * cols + col;
            C[idx] = A[idx] + B[idx];
        }
    }
}

// Kernel for Element-per-thread approach
__global__ void matrixAddElementPerThread(float *A, float *B, float *C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

// Function to print the matrix
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
    int rows, cols, choice;
    
    // User input for matrix size
    printf("Enter number of rows: ");
    scanf("%d", &rows);
    printf("Enter number of columns: ");
    scanf("%d", &cols);

    // User selects computation approach
    printf("Select computation approach:\n");
    printf("1. Row-per-thread\n");
    printf("2. Column-per-thread\n");
    printf("3. Element-per-thread\n");
    printf("Enter choice (1-3): ");
    scanf("%d", &choice);

    size_t SIZE = rows * cols * sizeof(float);
    
    // Allocate host matrices
    float *h_A = (float *)malloc(SIZE);
    float *h_B = (float *)malloc(SIZE);
    float *h_C = (float *)malloc(SIZE);

    // Initialize matrices
    for (int i = 0; i < rows * cols; i++) {
        h_A[i] = i + 1;
        h_B[i] = (i + 1) * 2;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, SIZE));
    CUDA_CHECK(cudaMalloc(&d_B, SIZE));
    CUDA_CHECK(cudaMalloc(&d_C, SIZE));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, SIZE, cudaMemcpyHostToDevice));

    // Configure kernel launch parameters
    if (choice == 1) { // Row-per-thread
        int threadsPerBlock = 256;
        int blocks = (rows + threadsPerBlock - 1) / threadsPerBlock;
        matrixAddRowPerThread<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);
        printf("Using Row-per-Thread Approach\n");
    } 
    else if (choice == 2) { // Column-per-thread
        int threadsPerBlock = 256;
        int blocks = (cols + threadsPerBlock - 1) / threadsPerBlock;
        matrixAddColPerThread<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);
        printf("Using Column-per-Thread Approach\n");
    } 
    else if (choice == 3) { // Element-per-thread
        dim3 threadsPerBlock(16, 16);
        dim3 blocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
        matrixAddElementPerThread<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);
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
    CUDA_CHECK(cudaMemcpy(h_C, d_C, SIZE, cudaMemcpyDeviceToHost));

    // Print matrices
    printf("Matrix A:\n");
    printMatrix(h_A, rows, cols);
    printf("Matrix B:\n");
    printMatrix(h_B, rows, cols);
    printf("Result Matrix C:\n");
    printMatrix(h_C, rows, cols);

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}

