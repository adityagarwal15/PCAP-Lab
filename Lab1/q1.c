#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int x;  // Integer constant 'x'
    double result;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank of the process and the total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Only the root process (rank 0) will ask for input
    if (rank == 0) {
        // Get the value of x from the user
        printf("Enter the value of x: ");
        scanf("%d", &x);
    }

    // Broadcast the value of x to all processes
    MPI_Bcast(&x, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate pow(x, rank) for each process
    result = pow(x, rank);

    // Print the result from each process
    printf("Process %d of %d: pow(%d, %d) = %f\n", rank, size, x, rank, result);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
