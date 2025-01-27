#include <stdio.h>
#include <mpi.h>
#include <string.h>

#define MAX_LEN 100  // Maximum length of the word

int main(int argc, char *argv[]) {
    int rank, size;
    char word[MAX_LEN];  // Input word
    int N;  // Length of the input word
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter a word: ");
        scanf("%s", word);  // Read the input word
        N = strlen(word);  // Calculate the length of the word
    }

    // Broadcast the length of the word to all processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Broadcast the word to all processes
    MPI_Bcast(word, N, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Calculate how many letters each process will handle
    int letters_per_process = N / size;
    int start = rank * letters_per_process;
    int end = (rank == size - 1) ? N : (rank + 1) * letters_per_process;

    // Store the output string locally for each process
    char local_result[MAX_LEN] = {0};

    // Each process adds the characters
    for (int i = start; i < end; i++) {
        int repeat_count = i + 1;
        for (int j = 0; j < repeat_count; j++) {
            local_result[i * N + j] = word[i];  // Add character to local result
        }
    }

    // Gather the results from all processes into the root process
    char gather_result[MAX_LEN * MAX_LEN] = {0};
    MPI_Gather(local_result, MAX_LEN, MPI_CHAR,
               gather_result, MAX_LEN, MPI_CHAR,
               0, MPI_COMM_WORLD);

    // Root process displays the final result
    if (rank == 0) {
        printf("Output: ");
        for (int i = 0; i < N; i++) {
            // Concatenate the repeated characters to the final result
            for (int j = 0; j <= i; j++) {
                printf("%c", word[i]);
            }
        }
        printf("\n");
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}

/*
Question:
Write an MPI program to read a word of length N. Using N processes including the root, get an output word with the pattern as shown in the example. The resultant output word should be displayed in the root process.

Example:

Input: PCAP
Output: PCCAAAPPPP

Sample I/O:
---------------------
Enter a word: PCAP

Output:
---------------------
PCCAAAPPPP
*/

