#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256 // Optimized block size
#define KERNEL_SIZE 5  // Convolution kernel size

__constant__ float d_kernel[KERNEL_SIZE]; // Constant memory for kernel

// CUDA kernel for 1D convolution using shared memory and constant memory
__global__ void conv1DKernel(float *input, float *output, int dataSize) {
    __shared__ float sharedInput[BLOCK_SIZE + KERNEL_SIZE - 1];
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int sharedIdx = threadIdx.x + KERNEL_SIZE / 2;

    // Load input data into shared memory
    if (globalIdx < dataSize)
        sharedInput[sharedIdx] = input[globalIdx];
    else
        sharedInput[sharedIdx] = 0.0f;

    if (threadIdx.x < KERNEL_SIZE / 2) {
        int leftIdx = globalIdx - KERNEL_SIZE / 2;
        sharedInput[threadIdx.x] = (leftIdx >= 0) ? input[leftIdx] : 0.0f;
    }

    if (threadIdx.x >= blockDim.x - KERNEL_SIZE / 2) {
        int rightIdx = globalIdx + KERNEL_SIZE / 2;
        sharedInput[sharedIdx + blockDim.x] = (rightIdx < dataSize) ? input[rightIdx] : 0.0f;
    }
    __syncthreads();

    // Perform convolution
    float sum = 0.0f;
    for (int k = 0; k < KERNEL_SIZE; k++) {
        sum += sharedInput[sharedIdx - KERNEL_SIZE / 2 + k] * d_kernel[k];
    }

    if (globalIdx < dataSize)
        output[globalIdx] = sum;
}

void initializeArray(float *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = (float)(rand() % 100) / 100.0f;
    }
}

int main() {
    int dataSize = 1 << 20; // 1M elements for benchmarking
    size_t size = dataSize * sizeof(float);
    float h_kernel[KERNEL_SIZE] = {0.2, 0.2, 0.2, 0.2, 0.2}; // Example averaging filter
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);
    initializeArray(h_input, dataSize);

    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_kernel, h_kernel, KERNEL_SIZE * sizeof(float));

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((dataSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    conv1DKernel<<<gridDim, blockDim>>>(d_input, d_output, dataSize);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution Time: %f ms\n", milliseconds);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    printf("1D convolution completed for %d elements.\n", dataSize);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
