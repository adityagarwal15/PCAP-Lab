#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16 // Block size for tiling

// Matrix multiplication kernel using 2D grid and 2D block with shared memory optimization
__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int width) {
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    for (int i = 0; i < (width + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
        // Load tiles into shared memory
        if (row < width && (i * BLOCK_SIZE + threadIdx.x) < width)
            tileA[threadIdx.y][threadIdx.x] = A[row * width + i * BLOCK_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        
        if (col < width && (i * BLOCK_SIZE + threadIdx.y) < width)
            tileB[threadIdx.y][threadIdx.x] = B[(i * BLOCK_SIZE + threadIdx.y) * width + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        // Compute partial dot product using shared memory
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < width && col < width)
        C[row * width + col] = sum;
}

// Function to initialize a matrix with random values
void initializeMatrix(float *matrix, int width) {
    for (int i = 0; i < width * width; i++) {
        matrix[i] = (float)(rand() % 100) / 100.0f;
    }
}

// Function to print a matrix
void printMatrix(float *matrix, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f ", matrix[i * width + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    int width = 1024; // Matrix dimensions (width x width)
    size_t size = width * width * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        printf("Host memory allocation failed!\n");
        return -1;
    }
    
    // Initialize matrices
    srand(42);  // Set seed for reproducibility
    initializeMatrix(h_A, width);
    initializeMatrix(h_B, width);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaError_t err;
    
    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess) {
        printf("CUDA Error (d_A malloc): %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess) {
        printf("CUDA Error (d_B malloc): %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        return -1;
    }
    
    err = cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess) {
        printf("CUDA Error (d_C malloc): %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return -1;
    }
    
    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (width + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event
    cudaEventRecord(start);
    
    // Execute matrix multiplication kernel
    matrixMultiplyKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, width);
    
    // Record stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error (kernel launch): %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);
        return -1;
    }
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel Execution Time: %.2f ms\n", milliseconds);
    
    // Copy result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Print small section of matrices for verification (if matrix is small)
    if (width <= 10) {
        printf("Matrix A:\n");
        printMatrix(h_A, width);
        
        printf("Matrix B:\n");
        printMatrix(h_B, width);
        
        printf("Result Matrix C:\n");
        printMatrix(h_C, width);
    } else {
        // For large matrices, just print a small corner
        printf("Result Matrix C (top-left 5x5 corner):\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                printf("%.2f ", h_C[i * width + j]);
            }
            printf("\n");
        }
    }
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    
    printf("\nMatrix multiplication completed for %dx%d matrices.\n", width, width);
    return 0;
}
