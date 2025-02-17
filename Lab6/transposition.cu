#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<stdio.h>

__global__ void add_vec(int*da,int s,int k){
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Calculate global thread index
    int temp = i * 2;  // Calculate the pair of elements to compare
    
    if (k == 0) {
        temp++;  // If k is 0, compare elements at odd indices
    }

    if (temp + 1 > s - 1)  // If the pair exceeds bounds, return
        return;

    // If the pair is out of order, swap them
    if (da[temp] > da[temp + 1]) {
        da[temp] = da[temp] ^ da[temp + 1];
        da[temp + 1] = da[temp] ^ da[temp + 1];
        da[temp] = da[temp + 1] ^ da[temp];
    }
}

int main() {
    int n;
    
    // Ask for the length of the vector
    printf("Length of the vector : ");
    scanf("%d", &n);  // Sample Input: 6

    int a[n];
    int *da;

    // Allocate memory on the device (GPU) for the input vector
    cudaMalloc((void**)&da, n * sizeof(int));

    // Ask the user to input the elements of the vector
    printf("Enter vector one : ");
    for (int i = 0; i < n; i++)
        scanf("%d", &a[i]);  // Sample Input: 6 5 4 3 2 1
    
    // Copy the input vector from host to device memory
    cudaMemcpy(da, a, n * sizeof(int), cudaMemcpyHostToDevice);

    // Set grid and block dimensions for CUDA kernel
    dim3 grid(n / 2, 1, 1);  // n/2 blocks, each with 1 thread
    dim3 blk(1, 1, 1);        // 1 thread per block

    // Perform the sorting operation using the kernel
    for (int i = 1; i < n + 1; i++) {
        add_vec<<<grid, blk>>>(da, n, i % 2);  // Alternate sorting passes
    }

    // Copy the result back to host memory
    cudaMemcpy(a, da, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the sorted vector
    printf("Sorted Vector : ");
    for (int i = 0; i < n; i++)
        printf("%d\t", a[i]);  // Sample Output: 1   2   3   4   5   6
    
    printf("\n");

    // Free the device memory
    cudaFree(da);
}

/*
Sample Input:
Length of the vector : 6
Enter vector one : 6 5 4 3 2 1

Sample Output:
Sorted Vector : 1   2   3   4   5   6   
*/
