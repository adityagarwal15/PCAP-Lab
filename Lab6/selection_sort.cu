#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<stdio.h>

__global__ void add_vec(int*da,int*dc,int s){
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Calculate global thread index
    int k = 0;
    
    // Count how many elements in da are smaller than da[i] or equal to da[i] and have a larger index
    for (int j = 0; j < s; j++) {
        if ((da[j] < da[i]) || (da[j] == da[i] && j > i)) 
            k++;
    }

    dc[k] = da[i];  // Assign da[i] to the corresponding position in dc
}

int main(){
    int n;
    
    // Ask the user for the length of the vector
    printf("Length of the vector : ");
    scanf("%d", &n);  // Sample Input: 5

    int a[n], c[n];
    int *da, *dc;

    // Allocate memory on the GPU for the input and output vectors
    cudaMalloc((void**)&da, n * sizeof(int));
    cudaMalloc((void**)&dc, n * sizeof(int));

    // Ask the user to input the elements of the vector
    printf("Enter vector one : ");
    for (int i = 0; i < n; i++)
        scanf("%d", &a[i]);  // Sample Input: 4 3 5 1 2
    
    // Copy the input vector from host to device
    cudaMemcpy(da, a, n * sizeof(int), cudaMemcpyHostToDevice);

    // Set grid and block dimensions for CUDA kernel
    dim3 grid(n, 1, 1);  // n blocks, each with 1 thread
    dim3 blk(1, 1, 1);    // 1 thread per block

    // Launch the kernel
    add_vec<<<grid, blk>>>(da, dc, n);

    // Copy the result back from device to host
    cudaMemcpy(c, dc, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Output the sorted array (this is the "ranked" order of the original elements)
    printf("Sorted Vector : ");
    for (int i = 0; i < n; i++)
        printf("%d\t", c[i]);  // Sample Output: 1    2    3    4    5
    
    printf("\n");

    // Free the allocated memory on the device
    cudaFree(da);
    cudaFree(dc);
}

/*
Sample Input:
Length of the vector : 5
Enter vector one : 4 3 5 1 2

Sample Output:
Sorted Vector : 1   2   3   4   5   
*/
