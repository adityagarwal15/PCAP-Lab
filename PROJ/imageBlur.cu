#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLUR_SIZE 1
#define BLOCK_SIZE 16

int xDimension;
int yDimension;

// Remove the custom uchar4 definition since it's already defined by CUDA

void readImage(char *filename, uchar4 **image);
void writeImage(uchar4 *image, char *filename, int width, int height);

__global__ void unsharedBlurring(uchar4 *image, uchar4 *imageOutput, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        uchar4 pixel = make_uchar4(0, 0, 0, 0);
        float4 average = make_float4(0, 0, 0, 0);

        for (int i = -BLUR_SIZE; i <= BLUR_SIZE; i++) {
            for (int j = -BLUR_SIZE; j <= BLUR_SIZE; j++) {
                int blurRow = row + i;
                int blurCol = col + j;

                // Boundary check
                if (blurRow >= 0 && blurRow < height && blurCol >= 0 && blurCol < width) {
                    pixel = image[blurRow * width + blurCol];
                } else {
                    pixel = make_uchar4(0, 0, 0, 0); // Set to black if out of bounds
                }
                average.x += pixel.x;  // Change .r to .x
                average.y += pixel.y;  // Change .g to .y
                average.z += pixel.z;  // Change .b to .z
                // Note: Leave out pixel.a since you're not using it
            }
        }

        // Divide summation by number of pixels
        int numPixels = (BLUR_SIZE * 2 + 1) * (BLUR_SIZE * 2 + 1);
        average.x /= numPixels;
        average.y /= numPixels;
        average.z /= numPixels;

        imageOutput[row * width + col] = make_uchar4((unsigned char)average.x, (unsigned char)average.y, (unsigned char)average.z, 255);
    }
}

__global__ void sharedBlurring(uchar4 *image, uchar4 *imageOutput, int width, int height) {
    int col = threadIdx.x + blockIdx.x * (blockDim.x - 2 * BLUR_SIZE);
    int row = threadIdx.y + blockIdx.y * (blockDim.y - 2 * BLUR_SIZE);

    if (col < width && row < height) {
        __shared__ uchar4 chunk[BLOCK_SIZE + 2 * BLUR_SIZE][BLOCK_SIZE + 2 * BLUR_SIZE];

        // Load elements into shared memory
        int relativeRow = threadIdx.y + BLUR_SIZE;
        int relativeCol = threadIdx.x + BLUR_SIZE;
        if (row < height && col < width) {
            chunk[relativeRow][relativeCol] = image[row * width + col];
        } else {
            chunk[relativeRow][relativeCol] = make_uchar4(0, 0, 0, 0); // Set to black if out of bounds
        }

        __syncthreads();

        // Perform the blur operation within the shared memory
        if (threadIdx.x >= BLUR_SIZE && threadIdx.x < blockDim.x - BLUR_SIZE && threadIdx.y >= BLUR_SIZE && threadIdx.y < blockDim.y - BLUR_SIZE) {
            uchar4 pixel = make_uchar4(0, 0, 0, 0);
            float4 average = make_float4(0, 0, 0, 0);

            for (int i = -BLUR_SIZE; i <= BLUR_SIZE; i++) {
                for (int j = -BLUR_SIZE; j <= BLUR_SIZE; j++) {
                    int blurRow = threadIdx.y + i;
                    int blurCol = threadIdx.x + j;

                    pixel = chunk[blurRow][blurCol];
                    average.x += pixel.x;  // Change .r to .x
                    average.y += pixel.y;  // Change .g to .y
                    average.z += pixel.z;  // Change .b to .z
                }
            }

            // Divide summation by number of pixels
            int numPixels = (BLUR_SIZE * 2 + 1) * (BLUR_SIZE * 2 + 1);
            average.x /= numPixels;
            average.y /= numPixels;
            average.z /= numPixels;

            imageOutput[row * width + col] = make_uchar4((unsigned char)average.x, (unsigned char)average.y, (unsigned char)average.z, 255);
        }
    }
}

void readImage(char *filename, uchar4 **image) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error reading file %s\n", filename);
        exit(1);
    }

    // Simple PPM image parsing (only P6 format)
    char format[3];
    fscanf(file, "%2s\n", format);
    if (strcmp(format, "P6") != 0) {
        printf("Only P6 PPM format is supported\n");
        exit(1);
    }

    int width, height, maxColorValue;
    fscanf(file, "%d %d\n%d\n", &width, &height, &maxColorValue);

    *image = (uchar4 *)malloc(width * height * sizeof(uchar4));
    for (int i = 0; i < width * height; i++) {
        (*image)[i].x = fgetc(file);  // Change .r to .x
        (*image)[i].y = fgetc(file);  // Change .g to .y
        (*image)[i].z = fgetc(file);  // Change .b to .z
        (*image)[i].w = 255;          // Change .a to .w
    }

    fclose(file);
    xDimension = width;
    yDimension = height;
}

void writeImage(uchar4 *image, char *filename, int width, int height) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        printf("Error writing to file %s\n", filename);
        exit(1);
    }

    fprintf(file, "P6\n%d %d\n255\n", width, height);

    for (int i = 0; i < width * height; i++) {
        fputc(image[i].x, file);  // Change .r to .x
        fputc(image[i].y, file);  // Change .g to .y
        fputc(image[i].z, file);  // Change .b to .z
    }

    fclose(file);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        printf("Usage: ./imageBlur <input_image> <output_image> <blur_type (0 for unshared, 1 for shared)>\n");
        exit(1);
    }

    char *inputImageFile = argv[1];
    char *outputImageFile = argv[2];
    int blurType = atoi(argv[3]);

    uchar4 *hostImage;
    uchar4 *deviceImage, *deviceImageOutput;

    // Read input image
    readImage(inputImageFile, &hostImage);
    
    size_t imageSize = xDimension * yDimension * sizeof(uchar4);

    // Allocate memory on the device
    cudaMalloc((void **)&deviceImage, imageSize);
    cudaMalloc((void **)&deviceImageOutput, imageSize);

    cudaMemcpy(deviceImage, hostImage, imageSize, cudaMemcpyHostToDevice);

    // Setup kernel dimensions
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((xDimension + BLOCK_SIZE - 1) / BLOCK_SIZE, (yDimension + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (blurType == 0) {
        printf("Using unshared memory for blurring...\n");
        unsharedBlurring<<<blocksPerGrid, threadsPerBlock>>>(deviceImage, deviceImageOutput, xDimension, yDimension);
    } else {
        printf("Using shared memory for blurring...\n");
        sharedBlurring<<<blocksPerGrid, threadsPerBlock>>>(deviceImage, deviceImageOutput, xDimension, yDimension);
    }

    // Copy output back to host
    cudaMemcpy(hostImage, deviceImageOutput, imageSize, cudaMemcpyDeviceToHost);

    // Write output image
    writeImage(hostImage, outputImageFile, xDimension, yDimension);

    // Free memory
    cudaFree(deviceImage);
    cudaFree(deviceImageOutput);
    free(hostImage);

    return 0;
}
