#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_WORD_LENGTH 100
#define MAX_SENTENCE_LENGTH 1024  // Maximum sentence length

__device__ int compare_words(const char* word1, const char* word2) {
    int i = 0;
    while (word1[i] != '\0' && word2[i] != '\0') {
        if (word1[i] != word2[i]) return 0;
        i++;
    }
    return (word1[i] == '\0' && word2[i] == '\0');
}

__global__ void count_word_kernel(const char* sentence, const char* target_word, int* count, int sentence_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < sentence_length) {
        // Only process if we're at the start of a word (first char or after space)
        if (idx == 0 || sentence[idx - 1] == ' ') {
            char current_word[MAX_WORD_LENGTH];
            int word_idx = 0;
            
            // Extract the word starting at this position
            while (idx + word_idx < sentence_length && 
                   sentence[idx + word_idx] != ' ' && 
                   sentence[idx + word_idx] != '\0' && 
                   word_idx < MAX_WORD_LENGTH - 1) {
                current_word[word_idx] = sentence[idx + word_idx];
                word_idx++;
            }
            current_word[word_idx] = '\0';
            
            // Compare with target word and increment atomic counter if match
            if (compare_words(current_word, target_word)) {
                atomicAdd(count, 1);
            }
        }
    }
}

void count_word_occurrences(const char* sentence, const char* target_word) {
    int sentence_length = strlen(sentence);
    
    // Allocate device memory
    char *d_sentence, *d_target_word;
    int *d_count;
    
    cudaMalloc((void**)&d_sentence, (sentence_length + 1) * sizeof(char));
    cudaMalloc((void**)&d_target_word, (strlen(target_word) + 1) * sizeof(char));
    cudaMalloc((void**)&d_count, sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_sentence, sentence, (sentence_length + 1) * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_word, target_word, (strlen(target_word) + 1) * sizeof(char), cudaMemcpyHostToDevice);
    
    int initial_count = 0;
    cudaMemcpy(d_count, &initial_count, sizeof(int), cudaMemcpyHostToDevice);
    
    // Calculate grid and block dimensions
    int blockSize = 256;
    int numBlocks = (sentence_length + blockSize - 1) / blockSize;
    
    // Launch kernel
    count_word_kernel<<<numBlocks, blockSize>>>(d_sentence, d_target_word, d_count, sentence_length);
    
    // Wait for completion and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Get result
    int count_result;
    cudaMemcpy(&count_result, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("The word '%s' appears %d times in the sentence.\n", target_word, count_result);
    
    // Cleanup
    cudaFree(d_sentence);
    cudaFree(d_target_word);
    cudaFree(d_count);
}

int main() {
    char sentence[MAX_SENTENCE_LENGTH];
    char target_word[MAX_WORD_LENGTH];
    
    // Get sentence input from user
    printf("Enter a sentence (max %d characters): ", MAX_SENTENCE_LENGTH - 1);
    fgets(sentence, MAX_SENTENCE_LENGTH, stdin);
    
    // Remove trailing newline from fgets if present
    size_t len = strlen(sentence);
    if (len > 0 && sentence[len-1] == '\n') {
        sentence[len-1] = '\0';
    }
    
    // Get target word from user
    printf("Enter the word to count: ");
    scanf("%s", target_word);
    
    // Check if inputs are not empty
    if (strlen(sentence) == 0 || strlen(target_word) == 0) {
        printf("Error: Sentence and word cannot be empty.\n");
        return 1;
    }
    
    count_word_occurrences(sentence, target_word);
    return 0;
}
