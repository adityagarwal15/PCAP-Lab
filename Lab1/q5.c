#include <stdio.h>
#include <mpi.h>

// Function to compute factorial
long long factorial(int n) {
    long long result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

// Function to compute Fibonacci number
long long fibonacci(int n) {
    if (n == 0) return 0;
    if (n == 1) return 1;

    long long a = 0, b = 1, temp;
    for (int i = 2; i <= n; i++) {
        temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

int main(int argc, char *argv[]) {
    int rank, size;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank of the process and the total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Even ranked processes calculate factorial
    if (rank % 2 == 0) {
        long long fact = factorial(rank);
        printf("Process %d: Factorial of %d is %lld\n", rank, rank, fact);
    }
    // Odd ranked processes calculate Fibonacci number
    else {
        long long fib = fibonacci(rank);
        printf("Process %d: Fibonacci number at position %d is %lld\n", rank, rank, fib);
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
