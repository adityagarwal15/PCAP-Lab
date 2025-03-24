#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Kernel for block-level inclusive scan
__global__ void inclusiveScanKernel(int *d_input, int *d_output, int *d_block_sums, int n) {
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    temp[tid] = (gid < n) ? d_input[gid] : 0;
    __syncthreads();
    
    for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        int val = (tid >= offset) ? temp[tid - offset] : 0;
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }
    
    if (gid < n) {
        d_output[gid] = temp[tid];
    }
    if (tid == BLOCK_SIZE - 1 && blockIdx.x < gridDim.x) {
        d_block_sums[blockIdx.x] = temp[tid];
    }
}

__global__ void addBlockSums(int *d_output, int *d_block_sums, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && gid < n) {
        d_output[gid] += d_block_sums[blockIdx.x - 1];
    }
}

void initializeArray(int *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = rand() % 10 + 1;
    }
}

bool verifyResult(int *input, int *output, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += input[i];
        if (output[i] != sum) {
            printf("Verification failed at index %d: expected %d, got %d\n", i, sum, output[i]);
            return false;
        }
    }
    return true;
}

void inclusiveScan(int *h_input, int *h_output, int n, float *time_ms) {
    size_t size = n * sizeof(int);
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    int *d_input, *d_output, *d_block_sums;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_block_sums, num_blocks * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(num_blocks);
    
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    inclusiveScanKernel<<<gridDim, blockDim, BLOCK_SIZE * sizeof(int)>>>(d_input, d_output, d_block_sums, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    if (num_blocks > 1) {
        int *h_block_sums = (int*)malloc(num_blocks * sizeof(int));
        CHECK_CUDA_ERROR(cudaMemcpy(h_block_sums, d_block_sums, num_blocks * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 1; i < num_blocks; i++) {
            h_block_sums[i] += h_block_sums[i-1];
        }
        CHECK_CUDA_ERROR(cudaMemcpy(d_block_sums, h_block_sums, num_blocks * sizeof(int), cudaMemcpyHostToDevice));
        addBlockSums<<<gridDim, blockDim>>>(d_output, d_block_sums, n);
        CHECK_CUDA_ERROR(cudaGetLastError());
        free(h_block_sums);
    }
    
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(time_ms, start, stop));
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    CHECK_CUDA_ERROR(cudaFree(d_block_sums));
}

int main() {
    const int n = 1 << 20;
    size_t size = n * sizeof(int);
    
    int *h_input = (int*)malloc(size);
    int *h_output = (int*)malloc(size);
    initializeArray(h_input, n);
    float execution_time = 0;
    inclusiveScan(h_input, h_output, n, &execution_time);
    
    printf("Inclusive scan completed for %d elements\n", n);
    printf("Execution Time: %.3f ms\n", execution_time);
    
    if (verifyResult(h_input, h_output, n)) {
        printf("Verification passed!\n");
    }
    
    free(h_input);
    free(h_output);
    return 0;
}

