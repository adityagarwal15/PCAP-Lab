#include <stdio.h>
#include <mpi.h>
#include <ctype.h>  // For the tolower() and toupper() functions

int main(int argc, char *argv[]) {
    int rank, size;
    char str[] = "HELLO";  // The string to be toggled

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank of the process and the total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ensure the number of processes doesn't exceed the length of the string
    if (size > sizeof(str) - 1) {
        if (rank == 0) {
            printf("Error: Number of processes cannot exceed the length of the string.\n");
        }
        MPI_Finalize();
        return 0;
    }

    // Each process toggles the character at its index (rank)
    if (rank < sizeof(str) - 1) {  // Ensure rank is within bounds of the string
        // Toggle the character
        if (isupper(str[rank])) {
            str[rank] = tolower(str[rank]);  // If it's uppercase, make it lowercase
        } else {
            str[rank] = toupper(str[rank]);  // If it's lowercase, make it uppercase
        }

        // Print the modified string from each process
        printf("Process %d: Modified string = %s\n", rank, str);
    }

    // Synchronize processes to ensure all have printed before finalizing
    MPI_Barrier(MPI_COMM_WORLD);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
