#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define N 3 // Matrix size (3x3)

int main(int argc, char* argv[]) {
    int rank, size;
    int matrix[N][N], searchElement, localCount = 0, globalCount = 0;
    int rowsPerProcess = N / 3;  // Divide rows among 3 processes

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root process reads the matrix and the search element
    if (rank == 0) {
        printf("Enter the 3x3 matrix elements:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }

        printf("Enter the element to be searched: ");
        scanf("%d", &searchElement);
    }

    // Broadcast the search element to all processes
    MPI_Bcast(&searchElement, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatter rows of the matrix to all processes
    int subMatrix[rowsPerProcess][N];
    MPI_Scatter(matrix, rowsPerProcess * N, MPI_INT, subMatrix, rowsPerProcess * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process counts occurrences of the search element in its assigned rows
    for (int i = 0; i < rowsPerProcess; i++) {
        for (int j = 0; j < N; j++) {
            if (subMatrix[i][j] == searchElement) {
                localCount++;
            }
        }
    }

    // Reduce the local counts to the root process to get the global count
    MPI_Reduce(&localCount, &globalCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Root process prints the result
    if (rank == 0) {
        printf("The element %d occurred %d times in the matrix.\n", searchElement, globalCount);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}

/*
Question:
Write an MPI program to read a 3x3 matrix. Enter an element to be searched in the root process.
Find the number of occurrences of this element in the matrix using three processes.

Sample I/O:
---------------------
Enter the 3x3 matrix elements:
1 2 3
4 5 6
7 8 9
Enter the element to be searched: 5

Output:
---------------------
The element 5 occurred 1 times in the matrix.
*/

