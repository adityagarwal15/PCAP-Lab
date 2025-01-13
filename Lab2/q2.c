// Write a MPI program where the master process (process 0) sends a number to each of the slaves
// and the slave processes receive the number and prints it. Use standard send.

#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int number = 42;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            printf("Master (Process 0): Sending number %d to Process %d\n", number, i);
            MPI_Send(&number, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        printf("Process %d: Received number %d from Process 0\n", rank, number);
    }

    MPI_Finalize();
    return 0;
}

// NOTE - NUMBER OF PROCESSES >2 ADVISED.
// Sample I/0:

// Master (Process 0): Sending number 42 to Process 1
// Master (Process 0): Sending number 42 to Process 2
// Master (Process 0): Sending number 42 to Process 3
// Master (Process 0): Sending number 42 to Process 4
// Process 1: Received number 42 from Process 0
// Process 2: Received number 42 from Process 0
// Process 3: Received number 42 from Process 0
// Process 4: Received number 42 from Process 0
