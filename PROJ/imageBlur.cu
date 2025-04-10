#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

// Define constants
#define DEFAULT_BLUR_SIZE 5
#define DEFAULT_BLUR_PASSES 1
#define DEFAULT_BLOCK_SIZE 16
#define DEFAULT_OUTPUT_WIDTH 0
#define DEFAULT_OUTPUT_HEIGHT 0

typedef enum {
    BLUR_BOX = 0,
    BLUR_GAUSSIAN = 1,
    BLUR_MEDIAN = 2
} BlurType;

// Function prototypes
void printHelp();
uchar4* readImage(const char *filename, int *width, int *height);
void writeImage(const uchar4 *image, const char *filename, int width, int height);
void applyBlur(const uchar4 *input, uchar4 *output, int width, int height, 
              int blurSize, float sigma, BlurType blurType, 
              int passes, bool useSharedMemory, int blockSize);
void resizeAndBlur(const uchar4 *input, uchar4 *output, 
                  int inputWidth, int inputHeight,
                  int outputWidth, int outputHeight,
                  int blurSize, int blockSize);
__global__ void blurKernel(const uchar4 *input, uchar4 *output, 
                          int width, int height, int blurSize);

// Help function
void printHelp() {
    printf("Enhanced Image Blur - CUDA Implementation\n");
    printf("Usage: ./imageBlur [options] <input_image> <output_image>\n\n");
    printf("Options:\n");
    printf("  -t <type>        Blur type (0=box, 1=gaussian, 2=median) [default: %d]\n", BLUR_BOX);
    printf("  -r <radius>      Blur radius [default: %d]\n", DEFAULT_BLUR_SIZE);
    printf("  -s <sigma>       Sigma value for Gaussian blur [default: 1.0]\n");
    printf("  -p <passes>      Number of blur passes [default: %d]\n", DEFAULT_BLUR_PASSES);
    printf("  -b <block_size>  CUDA block size [default: %d]\n", DEFAULT_BLOCK_SIZE);
    printf("  -m <mode>        Memory mode (0=global, 1=shared) [default: 1]\n");
    printf("  -w <width>       Output image width (0=use input width) [default: 0]\n");
    printf("  -h <height>      Output image height (0=use input height) [default: 0]\n");
}

// Image I/O functions
uchar4* readImage(const char *filename, int *width, int *height) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error: Cannot open file %s\n", filename);
        return NULL;
    }

    char magic[3];
    int maxval;
    if (fscanf(fp, "%2s\n%d %d\n%d\n", magic, width, height, &maxval) != 4 ||
        strcmp(magic, "P6") != 0 || maxval != 255) {
        printf("Error: Invalid PPM file format (must be P6 with maxval 255)\n");
        fclose(fp);
        return NULL;
    }

    size_t num_pixels = (*width) * (*height);
    uchar4 *image = (uchar4*)malloc(num_pixels * sizeof(uchar4));
    if (!image) {
        printf("Error: Memory allocation failed\n");
        fclose(fp);
        return NULL;
    }

    unsigned char *temp = (unsigned char*)malloc(num_pixels * 3);
    if (fread(temp, 1, num_pixels * 3, fp) != num_pixels * 3) {
        printf("Error: Failed to read image data\n");
        free(temp);
        free(image);
        fclose(fp);
        return NULL;
    }

    for (size_t i = 0; i < num_pixels; i++) {
        image[i].x = temp[i*3];     // R
        image[i].y = temp[i*3 + 1]; // G
        image[i].z = temp[i*3 + 2]; // B
        image[i].w = 255;           // A
    }

    free(temp);
    fclose(fp);
    printf("Loaded image: %s (%d x %d)\n", filename, *width, *height);
    return image;
}

void writeImage(const uchar4 *image, const char *filename, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        printf("Error: Cannot create file %s\n", filename);
        return;
    }

    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    
    size_t num_pixels = width * height;
    unsigned char *temp = (unsigned char*)malloc(num_pixels * 3);
    
    for (size_t i = 0; i < num_pixels; i++) {
        temp[i*3] = image[i].x;
        temp[i*3 + 1] = image[i].y;
        temp[i*3 + 2] = image[i].z;
    }
    
    fwrite(temp, 1, num_pixels * 3, fp);
    free(temp);
    fclose(fp);
    printf("Wrote image: %s (%d x %d)\n", filename, width, height);
}

// CUDA kernel for basic box blur
__global__ void blurKernel(const uchar4 *input, uchar4 *output, 
                          int width, int height, int blurSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        int count = 0;
        
        // Box blur: average all pixels in the (2*blurSize + 1) x (2*blurSize + 1) window
        for (int i = -blurSize; i <= blurSize; i++) {
            for (int j = -blurSize; j <= blurSize; j++) {
                int curY = row + i;
                int curX = col + j;
                
                if (curY >= 0 && curY < height && curX >= 0 && curX < width) {
                    uchar4 pixel = input[curY * width + curX];
                    sum.x += pixel.x;
                    sum.y += pixel.y;
                    sum.z += pixel.z;
                    count++;
                }
            }
        }
        
        uchar4 result;
        result.x = (unsigned char)(sum.x / count);
        result.y = (unsigned char)(sum.y / count);
        result.z = (unsigned char)(sum.z / count);
        result.w = 255;
        
        output[row * width + col] = result;
    }
}

// Updated applyBlur to use CUDA kernel
void applyBlur(const uchar4 *input, uchar4 *output, int width, int height, 
              int blurSize, float sigma, BlurType blurType, 
              int passes, bool useSharedMemory, int blockSize) {
    uchar4 *d_input, *d_output, *d_temp;
    size_t size = width * height * sizeof(uchar4);
    
    // Allocate device memory
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);
    cudaMalloc((void **)&d_temp, size);
    
    // Copy input to device
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blocksPerGrid((width + blockSize - 1) / blockSize,
                      (height + blockSize - 1) / blockSize);
    
    // Apply blur for specified number of passes
    for (int p = 0; p < passes; p++) {
        blurKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height, blurSize);
        
        // Swap buffers for next pass
        if (p < passes - 1) {
            cudaMemcpy(d_temp, d_output, size, cudaMemcpyDeviceToDevice);
            uchar4 *temp = d_input;
            d_input = d_temp;
            d_temp = temp;
        }
    }
    
    // Copy result back to host
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
}

// Kernel for resize and blur
__global__ void resizeAndBlurKernel(const uchar4 *input, uchar4 *output, 
                                   int inputWidth, int inputHeight,
                                   int outputWidth, int outputHeight,
                                   int blurSize) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (outCol < outputWidth && outRow < outputHeight) {
        float inX = (float)outCol * inputWidth / outputWidth;
        float inY = (float)outRow * inputHeight / outputHeight;
        
        int inCol = (int)inX;
        int inRow = (int)inY;
        
        float dx = inX - inCol;
        float dy = inY - inRow;
        
        uchar4 p00 = input[min(inRow, inputHeight-1) * inputWidth + min(inCol, inputWidth-1)];
        uchar4 p01 = input[min(inRow, inputHeight-1) * inputWidth + min(inCol+1, inputWidth-1)];
        uchar4 p10 = input[min(inRow+1, inputHeight-1) * inputWidth + min(inCol, inputWidth-1)];
        uchar4 p11 = input[min(inRow+1, inputHeight-1) * inputWidth + min(inCol+1, inputWidth-1)];
        
        if (blurSize > 0) {
            float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            int count = 0;
            
            for (int i = -blurSize; i <= blurSize; i++) {
                for (int j = -blurSize; j <= blurSize; j++) {
                    int curY = inRow + i;
                    int curX = inCol + j;
                    
                    if (curY >= 0 && curY < inputHeight && curX >= 0 && curX < inputWidth) {
                        uchar4 pixel = input[curY * inputWidth + curX];
                        sum.x += pixel.x;
                        sum.y += pixel.y;
                        sum.z += pixel.z;
                        count++;
                    }
                }
            }
            
            if (count > 0) {
                p00.x = p01.x = p10.x = p11.x = (unsigned char)(sum.x / count);
                p00.y = p01.y = p10.y = p11.y = (unsigned char)(sum.y / count);
                p00.z = p01.z = p10.z = p11.z = (unsigned char)(sum.z / count);
            }
        }
        
        uchar4 result;
        result.x = (unsigned char)((1-dx)*(1-dy)*p00.x + dx*(1-dy)*p01.x + (1-dx)*dy*p10.x + dx*dy*p11.x);
        result.y = (unsigned char)((1-dx)*(1-dy)*p00.y + dx*(1-dy)*p01.y + (1-dx)*dy*p10.y + dx*dy*p11.y);
        result.z = (unsigned char)((1-dx)*(1-dy)*p00.z + dx*(1-dy)*p01.z + (1-dx)*dy*p10.z + dx*dy*p11.z);
        result.w = 255;
        
        output[outRow * outputWidth + outCol] = result;
    }
}

void resizeAndBlur(const uchar4 *input, uchar4 *output, 
                  int inputWidth, int inputHeight,
                  int outputWidth, int outputHeight,
                  int blurSize, int blockSize) {
    uchar4 *d_input, *d_output;
    float kernelTime = 0.0f;
    
    size_t inputSize = inputWidth * inputHeight * sizeof(uchar4);
    size_t outputSize = outputWidth * outputHeight * sizeof(uchar4);
    
    cudaMalloc((void **)&d_input, inputSize);
    cudaMalloc((void **)&d_output, outputSize);
    
    cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blocksPerGrid((outputWidth + blockSize - 1) / blockSize,
                      (outputHeight + blockSize - 1) / blockSize);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    resizeAndBlurKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_input, d_output, inputWidth, inputHeight, outputWidth, outputHeight, blurSize);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&kernelTime, start, stop);
    
    printf("Resize and blur kernel execution time: %.3f ms\n", kernelTime);
    
    cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Main function
int main(int argc, char **argv) {
    int xDimension, yDimension;
    char inputImageFile[256] = {0};
    char outputImageFile[256] = {0};
    uchar4 *hostImage = NULL;

    int blurSize = DEFAULT_BLUR_SIZE;
    int blockSize = DEFAULT_BLOCK_SIZE;
    int passes = DEFAULT_BLUR_PASSES;
    float sigma = 1.0f;
    BlurType blurType = BLUR_BOX;
    bool useSharedMemory = true;
    int outputWidth = DEFAULT_OUTPUT_WIDTH;
    int outputHeight = DEFAULT_OUTPUT_HEIGHT;
    
    int arg = 1;
    while (arg < argc && argv[arg][0] == '-') {
        switch (argv[arg][1]) {
            case 't': if (++arg < argc) blurType = (BlurType)atoi(argv[arg]); break;
            case 'r': if (++arg < argc) blurSize = atoi(argv[arg]); break;
            case 's': if (++arg < argc) sigma = atof(argv[arg]); break;
            case 'p': if (++arg < argc) passes = atoi(argv[arg]); break;
            case 'b': if (++arg < argc) blockSize = atoi(argv[arg]); break;
            case 'm': if (++arg < argc) useSharedMemory = (atoi(argv[arg]) == 1); break;
            case 'w': if (++arg < argc) outputWidth = atoi(argv[arg]); break;
            case 'h': 
                if (strcmp(argv[arg], "-h") == 0) { printHelp(); return 0; }
                if (++arg < argc) outputHeight = atoi(argv[arg]); 
                break;
            default: printf("Unknown option: %s\n", argv[arg]); printHelp(); return 1;
        }
        arg++;
    }
    
    if (arg >= argc) { printf("Error: Input image file not specified.\n"); printHelp(); return 1; }
    strcpy(inputImageFile, argv[arg++]);
    if (arg >= argc) { printf("Error: Output image file not specified.\n"); printHelp(); return 1; }
    strcpy(outputImageFile, argv[arg]);
    
    hostImage = readImage(inputImageFile, &xDimension, &yDimension);
    if (!hostImage) return 1;
    
    int finalWidth = (outputWidth > 0) ? outputWidth : xDimension;
    int finalHeight = (outputHeight > 0) ? outputHeight : yDimension;
    
    uchar4 *outputImage = (uchar4 *)malloc(finalWidth * finalHeight * sizeof(uchar4));
    if (!outputImage) {
        printf("Error: Failed to allocate memory for output image.\n");
        free(hostImage);
        return 1;
    }
    
    printf("Processing image: %s -> %s\n", inputImageFile, outputImageFile);
    printf("Input dimensions: %d x %d\n", xDimension, yDimension);
    
    if (finalWidth != xDimension || finalHeight != yDimension) {
        printf("Output dimensions: %d x %d (resizing enabled)\n", finalWidth, finalHeight);
        resizeAndBlur(hostImage, outputImage, xDimension, yDimension, 
                     finalWidth, finalHeight, blurSize, blockSize);
    } else {
        printf("Output dimensions: %d x %d\n", finalWidth, finalHeight);
        applyBlur(hostImage, outputImage, xDimension, yDimension, 
                 blurSize, sigma, blurType, passes, useSharedMemory, blockSize);
    }
    
    writeImage(outputImage, outputImageFile, finalWidth, finalHeight);
    
    free(hostImage);
    free(outputImage);
    return 0;
}
