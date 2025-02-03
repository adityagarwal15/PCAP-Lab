#include <stdio.h>
#include <cuda.h>

// CUDA kernel for odd-even transposition sort
__global__ void oddEvenTranspositionSort(int *matrix, int numRows, int numCols) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;  // Global thread index
    int temp;

    // Odd phase
    if (tid % 2 == 1 && tid < numRows * numCols - 1) {
        if (matrix[tid] > matrix[tid + 1]) {
            // Swap the elements
            temp = matrix[tid];
            matrix[tid] = matrix[tid + 1];
            matrix[tid + 1] = temp;
        }
    }
    __syncthreads();  // Ensure all threads have finished swapping

    // Even phase
    if (tid % 2 == 0 && tid < numRows * numCols - 1) {
        if (matrix[tid] > matrix[tid + 1]) {
            // Swap the elements
            temp = matrix[tid];
            matrix[tid] = matrix[tid + 1];
            matrix[tid + 1] = temp;
        }
    }
}

int main() {
    int numRows = 3, numCols = 5;
    int matrix[3][5] = {
        {12, 11, 15, 10, 14},
        {22, 21, 25, 19, 23},
        {32, 31, 35, 30, 34}
    };

    int *d_matrix;
    int size = numRows * numCols * sizeof(int);

    // Allocate memory on device
    cudaMalloc((void**)&d_matrix, size);

    // Copy the matrix from host to device
    cudaMemcpy(d_matrix, matrix, size, cudaMemcpyHostToDevice);

    // Launch the kernel for multiple iterations to achieve sorted output
    int numIterations = 10;  // Number of iterations to ensure sorting
    for (int i = 0; i < numIterations; i++) {
        oddEvenTranspositionSort<<<(numRows * numCols + 255) / 256, 256>>>(d_matrix, numRows, numCols);
    }

    // Copy the sorted matrix back to the host
    cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost);

    // Print the sorted matrix
    printf("Sorted Matrix:\n");
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_matrix);

    return 0;
}

