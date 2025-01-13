// 3) Write a MPI program to read N elements of the array in the root process (process 0)
// where N is equal to the total number of processes. The root process sends one value to
// each of the slaves. Let even ranked process finds square of the received element and
// odd ranked process finds cube of received element. Use Buffered send.

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int N;
    int *array = NULL;
    int rcvdval, res;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    N = size;

    int buffer_size = N * sizeof(int) + MPI_BSEND_OVERHEAD;
    int *buffer = (int *)malloc(buffer_size);
    MPI_Buffer_attach(buffer, buffer_size);

    if (rank == 0) {
        array = (int *)malloc(N * sizeof(int));
        printf("Enter %d elements for the array: ", N);
        for (int i = 0; i < N; i++) {
            scanf("%d", &array[i]);
        }
    }


    if (rank == 0) {
        for (int i = 1; i < N; i++) {
            MPI_Bsend(&array[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&rcvdval, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        if (rank % 2 == 0) {
            res = rcvdval * rcvdval;
            printf("Process %d (Even) received %d and squared it to %d\n", rank, rcvdval, res);
        } 
        else {
            res = rcvdval * rcvdval * rcvdval;
            printf("Process %d (Odd) received %d and cubed it to %d\n", rank, rcvdval, res);
        }
        MPI_Bsend(&res, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        for (int i = 1; i < N; i++) {
            MPI_Recv(&res, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
            printf("Root process received res from process %d: %d\n", i, res);
        }
        free(array); 
    }
    MPI_Buffer_detach(buffer, &buffer_size);
    MPI_Finalize();
    return 0;
}

// Sample I/O:

// Enter 3 elements for the array: 1 2 3
// Root process received res from process 1: 8
// Root process received res from process 2: 9
// Process 1 (Odd) received 2 and cubed it to 8
// Process 2 (Even) received 3 and squared it to 9
