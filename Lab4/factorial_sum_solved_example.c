//write a mpi program using n processes to find 1!+2!+.....+N!. Use collective communication routines.

#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size, fact = 1, factsum = 0, i;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate factorial for rank + 1
    for (i = 1; i <= rank + 1; i++) {
        fact = fact * i;
    }

    // Use MPI_Reduce to sum up all factorials
    MPI_Reduce(&fact, &factsum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Rank 0 prints the result
    if (rank == 0) {
        printf("Sum of all factorials (1! + 2! + ... + N!) = %d\n", factsum);
    }

    MPI_Finalize();
    return 0;
}




