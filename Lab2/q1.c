//  Write a MPI program using synchronous send. The sender process sends a word to the
// receiver. The second process receives the word, toggles each letter of the word
// and sends it back to the first process. Both processes use synchronous send operations.

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <mpi.h>

#define maxlen 100

void toggle_case(char* word) {
    for (int i = 0; word[i] != '\0'; i++) {
        if (isupper(word[i])) {
            word[i] = tolower(word[i]);
        } else if (islower(word[i])) {
            word[i] = toupper(word[i]);
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char word[maxlen];

    if (rank == 0) {
        printf("Process 0: Enter a word to send: ");
        fgets(word, maxlen, stdin);
        word[strcspn(word, "\n")] = '\0';

        MPI_Ssend(word, strlen(word) + 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        printf("Process 0: Sent word \"%s\" to process 1.\n", word);

        MPI_Recv(word, maxlen, MPI_CHAR, 1, 1, MPI_COMM_WORLD, &status);
        printf("Process 0: Received modified word \"%s\" from process 1.\n", word);
    }
    else if (rank == 1) {

        MPI_Recv(word, maxlen, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        printf("Process 1: Received word \"%s\" from process 0.\n", word);

        toggle_case(word);
        printf("Process 1: Toggled word to \"%s\".\n", word);

        MPI_Ssend(word, strlen(word) + 1, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
        printf("Process 1: Sent modified word \"%s\" back to process 0.\n", word);
    }

    MPI_Finalize();
    return 0;
}

// Sample I/O:

// Process 0: Enter a word to send: hello
// Process 0: Sent word "hello" to process 1.
// Process 0: Received modified word "HELLO" from process 1.
// Process 1: Received word "hello" from process 0.
// Process 1: Toggled word to "HELLO".
// Process 1: Sent modified word "HELLO" back to process 0.
