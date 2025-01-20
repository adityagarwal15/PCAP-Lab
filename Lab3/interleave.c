#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int rank, size, chunk_size;
    char *S1 = NULL;            // String 1 (root process)
    char *S2 = NULL;            // String 2 (root process)
    char *sub_S1 = NULL;        // Substring from S1 for each process
    char *sub_S2 = NULL;        // Substring from S2 for each process
    char *sub_result = NULL;    // Resultant substring for each process
    char *result = NULL;        // Final resultant string (root process)

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Root process reads the strings
        printf("Enter String S1: ");
        char input1[1000];
        scanf("%s", input1);

        printf("Enter String S2: ");
        char input2[1000];
        scanf("%s", input2);

        // Check if strings are of the same length
        if (strlen(input1) != strlen(input2)) {
            printf("Error: Strings must be of the same length.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int length = strlen(input1);

        // Check if the length is evenly divisible by the number of processes
        if (length % size != 0) {
            printf("Error: String length must be evenly divisible by the number of processes.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        chunk_size = length / size;

        // Allocate memory for strings
        S1 = (char *)malloc((length + 1) * sizeof(char));
        S2 = (char *)malloc((length + 1) * sizeof(char));
        result = (char *)malloc((length + 1) * sizeof(char));

        // Copy input strings into allocated memory
        strcpy(S1, input1);
        strcpy(S2, input2);
    }

    // Broadcast the chunk size to all processes
    MPI_Bcast(&chunk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for substrings in each process
    sub_S1 = (char *)malloc((chunk_size + 1) * sizeof(char));
    sub_S2 = (char *)malloc((chunk_size + 1) * sizeof(char));
    sub_result = (char *)malloc((2 * chunk_size + 1) * sizeof(char));

    // Scatter substrings of S1 and S2 to all processes
    MPI_Scatter(S1, chunk_size, MPI_CHAR, sub_S1, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(S2, chunk_size, MPI_CHAR, sub_S2, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Null-terminate the substrings for safety
    sub_S1[chunk_size] = '\0';
    sub_S2[chunk_size] = '\0';

    // Interleave characters from sub_S1 and sub_S2
    for (int i = 0; i < chunk_size; i++) {
        sub_result[2 * i] = sub_S1[i];
        sub_result[2 * i + 1] = sub_S2[i];
    }
    sub_result[2 * chunk_size] = '\0';

    // Gather the resultant substrings at the root process
    MPI_Gather(sub_result, 2 * chunk_size, MPI_CHAR, result, 2 * chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Root process displays the final resultant string
    if (rank == 0) {
        result[2 * strlen(S1)] = '\0'; // Null-terminate the result
        printf("Resultant String: %s\n", result);

        // Free allocated memory in the root process
        free(S1);
        free(S2);
        free(result);
    }

    // Free allocated memory in each process
    free(sub_S1);
    free(sub_S2);
    free(sub_result);

    MPI_Finalize();
    return 0;
}

