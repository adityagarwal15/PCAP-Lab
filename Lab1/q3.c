#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    double a, b;
    double result;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank of the process and the total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Only root process (rank 0) will take user input
    if (rank == 0) {
        printf("Enter two numbers a and b: ");
        scanf("%lf %lf", &a, &b);
    }

    // Broadcast the input values to all processes
    MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform different operations in parallel based on rank
    if (rank == 0) {
        result = a + b;
        printf("Process %d: Addition (a + b) = %lf\n", rank, result);
    } else if (rank == 1) {
        result = a - b;
        printf("Process %d: Subtraction (a - b) = %lf\n", rank, result);
    } else if (rank == 2) {
        result = a * b;
        printf("Process %d: Multiplication (a * b) = %lf\n", rank, result);
    } else if (rank == 3) {
        if (b != 0) {
            result = a / b;
            printf("Process %d: Division (a / b) = %lf\n", rank, result);
        } else {
            printf("Process %d: Error! Division by zero.\n", rank);
        }
    } else if (rank == 4) {
        if ((int)b != 0) {  // Modulus requires integer division
            result = (int)a % (int)b;
            printf("Process %d: Modulus (a %% b) = %lf\n", rank, result);
        } else {
            printf("Process %d: Error! Modulus by zero.\n", rank);
        }
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
