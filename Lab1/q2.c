#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank of the process and the total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if the rank is even or odd
    if (rank % 2 == 0) {
        printf("Process %d of %d: Hello\n", rank, size);
    } else {
        printf("Process %d of %d: World\n", rank, size);
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
