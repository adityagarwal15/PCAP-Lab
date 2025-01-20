#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Function to calculate factorial of a number
long long factorial(int num) {
    long long result = 1;
    for (int i = 1; i <= num; i++) {
        result *= i;
    }
    return result;
}

int main(int argc, char *argv[]) {
    int rank, size, N;
    int *values = NULL;     // Array to hold the N values in the root process
    long long local_result; // Holds the factorial calculated by each process
    long long total_sum;    // Holds the final sum of factorials in the root process

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Root process reads N and the values
        printf("Enter the number of values (N): ");
        scanf("%d", &N);

        if (N != size) {
            printf("Error: Number of processes must be equal to N.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        values = (int *)malloc(N * sizeof(int));
        printf("Enter %d values: ", N);
        for (int i = 0; i < N; i++) {
            scanf("%d", &values[i]);
        }
    }

    // Root process sends one value to each process
    int value;
    MPI_Scatter(values, 1, MPI_INT, &value, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process calculates the factorial of the received value
    local_result = factorial(value);

    // Gather the factorial results in the root process
    MPI_Reduce(&local_result, &total_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Root process prints the total sum of factorials
    if (rank == 0) {
        printf("The sum of factorials is: %lld\n", total_sum);
        free(values); // Free the dynamically allocated memory
    }

    MPI_Finalize();
    return 0;
}

