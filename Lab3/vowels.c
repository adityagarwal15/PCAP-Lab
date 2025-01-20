#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Function to check if a character is a vowel
int is_vowel(char ch) {
    ch = tolower(ch); // Convert to lowercase for case-insensitivity
    return (ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u');
}

int main(int argc, char *argv[]) {
    int rank, size, chunk_size;
    char *string = NULL;       // String to be processed (in root process)
    char *sub_string = NULL;   // Substring to be processed by each process
    int local_non_vowels = 0;  // Number of non-vowels counted by each process
    int total_non_vowels = 0;  // Total number of non-vowels (calculated in root)

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Root process reads the string
        printf("Enter a string: ");
        char input[1000];
        scanf("%s", input);

        int length = strlen(input);

        // Check if string length is evenly divisible by the number of processes
        if (length % size != 0) {
            printf("Error: String length must be evenly divisible by the number of processes.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        chunk_size = length / size;

        // Allocate memory for the string and copy the input into it
        string = (char *)malloc((length + 1) * sizeof(char));
        strcpy(string, input);
    }

    // Broadcast the chunk size to all processes
    MPI_Bcast(&chunk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for the substring in each process
    sub_string = (char *)malloc((chunk_size + 1) * sizeof(char));

    // Scatter the string across processes
    MPI_Scatter(string, chunk_size, MPI_CHAR, sub_string, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Null-terminate the substring for processing
    sub_string[chunk_size] = '\0';

    // Each process counts non-vowels in its substring
    for (int i = 0; i < chunk_size; i++) {
        if (!is_vowel(sub_string[i]) && isalpha(sub_string[i])) {
            local_non_vowels++;
        }
    }

    // Gather the results at the root process
    int *non_vowel_counts = NULL;
    if (rank == 0) {
        non_vowel_counts = (int *)malloc(size * sizeof(int));
    }
    MPI_Gather(&local_non_vowels, 1, MPI_INT, non_vowel_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Root process prints the results
    if (rank == 0) {
        printf("Non-vowels counted by each process:\n");
        for (int i = 0; i < size; i++) {
            printf("Process %d: %d non-vowels\n", i, non_vowel_counts[i]);
            total_non_vowels += non_vowel_counts[i];
        }
        printf("Total number of non-vowels: %d\n", total_non_vowels);

        // Free allocated memory in the root process
        free(string);
        free(non_vowel_counts);
    }

    // Free allocated memory in each process
    free(sub_string);

    MPI_Finalize();
    return 0;
}

