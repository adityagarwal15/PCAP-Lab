#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size, M, N;
    double *array = NULL; // Array to hold N x M elements in the root process
    double *sub_array = NULL; // Sub-array for each process to work on
    double local_avg = 0.0; // Average of M elements in each process
    double total_avg = 0.0; // Final average computed by the root process

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    N = size; // Number of processes

    if (rank == 0) {
        // Root process reads M and the N x M elements
        printf("Enter the value of M (number of elements per process): ");
        scanf("%d", &M);

        // Allocate memory for the array
        array = (double *)malloc(N * M * sizeof(double));

        printf("Enter %d elements (1D array of size %d): ", N * M, N * M);
        for (int i = 0; i < N * M; i++) {
            scanf("%lf", &array[i]);
        }
    }

    // Broadcast M to all processes
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for the sub-array
    sub_array = (double *)malloc(M * sizeof(double));

    // Scatter the data: each process gets M elements
    MPI_Scatter(array, M, MPI_DOUBLE, sub_array, M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Each process computes the average of its M elements
    double local_sum = 0.0;
    for (int i = 0; i < M; i++) {
        local_sum += sub_array[i];
    }
    local_avg = local_sum / M;

    // Gather all local averages at the root process
    double *local_avgs = NULL;
    if (rank == 0) {
        local_avgs = (double *)malloc(N * sizeof(double));
    }
    MPI_Gather(&local_avg, 1, MPI_DOUBLE, local_avgs, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Root process computes the total average
    if (rank == 0) {
        double sum_of_avgs = 0.0;
        for (int i = 0; i < N; i++) {
            sum_of_avgs += local_avgs[i];
        }
        total_avg = sum_of_avgs / N;

        printf("The total average is: %.2lf\n", total_avg);

        // Free allocated memory in the root process
        free(array);
        free(local_avgs);
    }

    // Free allocated memory in each process
    free(sub_array);

    MPI_Finalize();
    return 0;
}

