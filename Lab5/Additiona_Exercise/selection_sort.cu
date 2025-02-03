#include <stdio.h>
#include <cuda.h>

// CUDA kernel for selection sort on each row of the matrix
__global__ void selectionSortKernel(int *matrix, int numRows, int numCols) {
    int row = blockIdx.x;  // Each block handles one row
    int i, j, minIndex, temp;

    if (row < numRows) {
        for (i = 0; i < numCols - 1; i++) {
            minIndex = i;
            // Find the minimum element in the row starting from i
            for (j = i + 1; j < numCols; j++) {
                if (matrix[row * numCols + j] < matrix[row * numCols + minIndex]) {
                    minIndex = j;
                }
            }
            // Swap the found minimum element with the element at index i
            if (minIndex != i) {
                temp = matrix[row * numCols + i];
                matrix[row * numCols + i] = matrix[row * numCols + minIndex];
                matrix[row * numCols + minIndex] = temp;
            }
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

    // Launch the kernel with numRows blocks, each handling a row
    selectionSortKernel<<<numRows, 1>>>(d_matrix, numRows, numCols);

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

