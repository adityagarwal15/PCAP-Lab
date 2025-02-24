#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_LENGTH 1024

__global__ void generate_pattern_kernel(const char* input, char* output, int input_length, int* positions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_chars = input_length + (input_length - 1) + (input_length - 2) + (input_length - 3);
    
    if (idx < total_chars) {
        // Find which section we're in
        int section = 0;
        int pos = idx;
        
        // Determine which section this thread is working on
        while (pos >= positions[section] && section < 4) {
            pos -= positions[section];
            section++;
        }
        
        // Copy character based on section
        output[idx] = input[pos];
    }
}

void generate_pattern(const char* input) {
    int input_length = strlen(input);
    
    // Calculate positions for each section
    int positions[4];
    positions[0] = input_length;       // PCAP
    positions[1] = input_length - 1;   // PCA
    positions[2] = input_length - 2;   // PC
    positions[3] = input_length - 3;   // P
    
    // Calculate total output length
    int total_length = 0;
    for (int i = 0; i < 4; i++) {
        total_length += positions[i];
    }
    
    // Allocate device memory
    char *d_input, *d_output;
    int *d_positions;
    cudaMalloc((void**)&d_input, (input_length + 1) * sizeof(char));
    cudaMalloc((void**)&d_output, (total_length + 1) * sizeof(char));
    cudaMalloc((void**)&d_positions, 4 * sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_input, input, (input_length + 1) * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, positions, 4 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Calculate grid and block dimensions
    int blockSize = 256;
    int numBlocks = (total_length + blockSize - 1) / blockSize;
    
    // Launch kernel
    generate_pattern_kernel<<<numBlocks, blockSize>>>(d_input, d_output, input_length, d_positions);
    
    // Wait for completion and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Allocate host memory for result
    char* result = (char*)malloc((total_length + 1) * sizeof(char));
    
    // Copy result back to host
    cudaMemcpy(result, d_output, total_length * sizeof(char), cudaMemcpyDeviceToHost);
    result[total_length] = '\0';
    
    // Format the output with spaces between sections
    printf("Input string S: %s\n", input);
    printf("Output string RS: ");
    
    int current_pos = 0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < positions[i]; j++) {
            printf("%c", result[current_pos++]);
        }
        if (i < 3) printf(" ");  // Add space between sections
    }
    printf("\n");
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_positions);
    free(result);
}

int main() {
    char input[MAX_LENGTH];
    
    // Get input string
    printf("Enter the input string: ");
    scanf("%s", input);
    
    // Validate input
    if (strlen(input) == 0) {
        printf("Error: Input string cannot be empty.\n");
        return 1;
    }
    
    if (strlen(input) < 4) {
        printf("Error: Input string must be at least 4 characters long.\n");
        return 1;
    }
    
    generate_pattern(input);
    return 0;
}
