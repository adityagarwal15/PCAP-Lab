#include <stdio.h>
#include <mpi.h>

#define N 4  // Matrix size (4x4)

int main(int argc, char *argv[]) {
    int rank, size;
    int matrix[N][N];  // Input matrix
    int result[N][N] = {0};  // Output matrix
    int row_start, row_end;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root process takes input from the user
    if (rank == 0) {
        printf("Enter the 4x4 matrix elements:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }
    }

    // Broadcast the matrix from the root process to all other processes
    MPI_Bcast(matrix, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Divide rows equally among processes
    int rows_per_process = N / size;
    row_start = rank * rows_per_process;
    row_end = (rank == size - 1) ? N : (rank + 1) * rows_per_process;

    // Perform the transformation on the assigned rows
    for (int i = row_start; i < row_end; i++) {
        for (int j = 0; j < N; j++) {
            // For each element, sum the current row with all rows above it
            for (int k = 0; k <= i; k++) {
                result[i][j] += matrix[k][j];
            }
        }
    }

    // Create a separate buffer for gathering results
    int gather_result[N * N];  // Separate buffer to avoid aliasing
    MPI_Gather(&result[row_start][0], rows_per_process * N, MPI_INT,
               gather_result, rows_per_process * N, MPI_INT,
               0, MPI_COMM_WORLD);

    // Root process prints the result matrix
    if (rank == 0) {
        printf("\nInput Matrix:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", matrix[i][j]);
            }
            printf("\n");
        }

        printf("\nOutput Matrix:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", gather_result[i * N + j]);  // Access the gathered results
            }
            printf("\n");
        }
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}

/*
Question:
Write an MPI program to read a 4x4 matrix and display the following output using four processes.
I/p matrix: 
1 2 3 4
1 2 3 1
1 1 1 1
2 1 2 1

O/p matrix:
1 2 3 4
2 4 6 5
3 5 7 6
5 6 9 7

Sample I/O:
---------------------
Enter the 4x4 matrix elements:
1 2 3 4
1 2 3 1
1 1 1 1
2 1 2 1

Output:
---------------------
Input Matrix:
1 2 3 4 
1 2 3 1 
1 1 1 1 
2 1 2 1 

Output Matrix:
1 2 3 4 
2 4 6 5 
3 5 7 6 
5 6 9 7 
*/

