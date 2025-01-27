/*
 * Question:
 * Write an MPI program using N processes to calculate the sum of factorials:
 * 1! + 2! + ... + N!
 * 
 * - Use the `MPI_Scan` function to perform the calculation.
 * - Implement error handling using MPI's error handling routines to address potential issues such as invalid input, 
 *   broadcast failures, scan errors, or scenarios where N exceeds the number of processes.
 * - Each process should compute the factorial for its rank (if within the range 1 to N) and participate in the scan operation.
 *
 * Sample I/O:
 * Input: 
 *    Enter the value of N: 5
 * Output:
 *    Sum of factorials (1! + 2! + ... + N!) = 153
 * 
 * Input:
 *    Enter the value of N: -3
 * Output:
 *    Invalid input. Please enter a positive integer.
 * 
 * Input:
 *    Enter the value of N: 10 (with 8 processes)
 * Output:
 *    Error: N cannot be greater than the number of processes (8).
 */
 
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

// Function to calculate factorial
int factorial(int n) {
    int result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

int main(int argc, char* argv[]) {
    int rank, size, n, local_fact = 0, scan_result = 0, errcode;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Set the error handler to return errors instead of aborting
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    errcode = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (errcode != MPI_SUCCESS) {
        char error_string[BUFSIZ];
        int length_of_error_string;
        MPI_Error_string(errcode, error_string, &length_of_error_string);
        fprintf(stderr, "Rank Error: %s\n", error_string);
        MPI_Abort(MPI_COMM_WORLD, errcode);
        return EXIT_FAILURE;
    }

    errcode = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (errcode != MPI_SUCCESS) {
        char error_string[BUFSIZ];
        int length_of_error_string;
        MPI_Error_string(errcode, error_string, &length_of_error_string);
        fprintf(stderr, "Size Error: %s\n", error_string);
        MPI_Abort(MPI_COMM_WORLD, errcode);
        return EXIT_FAILURE;
    }

    // Root process inputs the value of N
    if (rank == 0) {
        printf("Enter the value of N: ");
        if (scanf("%d", &n) != 1 || n <= 0) {
            fprintf(stderr, "Invalid input. Please enter a positive integer.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return EXIT_FAILURE;
        }
    }

    // Broadcast the value of N to all processes
    errcode = MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (errcode != MPI_SUCCESS) {
        char error_string[BUFSIZ];
        int length_of_error_string;
        MPI_Error_string(errcode, error_string, &length_of_error_string);
        fprintf(stderr, "Broadcast Error: %s\n", error_string);
        MPI_Abort(MPI_COMM_WORLD, errcode);
        return EXIT_FAILURE;
    }

    // Ensure that N does not exceed the number of processes
    if (n > size) {
        if (rank == 0) {
            fprintf(stderr, "Error: N cannot be greater than the number of processes (%d).\n", size);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return EXIT_FAILURE;
    }

    // Calculate local factorial for each process (if within range)
    if (rank + 1 <= n) {
        local_fact = factorial(rank + 1);
    }

    // Perform MPI_Scan to compute the partial sum of factorials
    errcode = MPI_Scan(&local_fact, &scan_result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (errcode != MPI_SUCCESS) {
        char error_string[BUFSIZ];
        int length_of_error_string;
        MPI_Error_string(errcode, error_string, &length_of_error_string);
        fprintf(stderr, "Scan Error: %s\n", error_string);
        MPI_Abort(MPI_COMM_WORLD, errcode);
        return EXIT_FAILURE;
    }

    // Output the result from the last process
    if (rank == n - 1) {
        printf("Sum of factorials (1! + 2! + ... + N!) = %d\n", scan_result);
    }

    // Finalize MPI
    MPI_Finalize();
    return EXIT_SUCCESS;
}

