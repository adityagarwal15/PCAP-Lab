#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLUR_SIZE 1

int xDimension = 0;
int yDimension = 0;

void readImage(char *filename, unsigned char **image);
void writeImage(unsigned char *image, char *filename, int width, int height);

void unsharedBlurring(unsigned char *image, unsigned char *imageOutput, int width, int height) {
    printf("Starting unshared blurring...\n");
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int index = (row * width + col) * 3;
            if (index + 2 >= width * height * 3) {
                fprintf(stderr, "Error: Out of bounds access at index %d\n", index);
                exit(EXIT_FAILURE);
            }

            float averageR = 0.0f, averageG = 0.0f, averageB = 0.0f;
            int count = 0;

            for (int i = -BLUR_SIZE; i <= BLUR_SIZE; i++) {
                for (int j = -BLUR_SIZE; j <= BLUR_SIZE; j++) {
                    int blurRow = row + i;
                    int blurCol = col + j;

                    if (blurRow >= 0 && blurRow < height && blurCol >= 0 && blurCol < width) {
                        int blurIndex = (blurRow * width + blurCol) * 3;
                        averageR += image[blurIndex];
                        averageG += image[blurIndex + 1];
                        averageB += image[blurIndex + 2];
                        count++;
                    }
                }
            }

            imageOutput[index] = (unsigned char)(averageR / count);
            imageOutput[index + 1] = (unsigned char)(averageG / count);
            imageOutput[index + 2] = (unsigned char)(averageB / count);
        }
    }
    printf("Unshared blurring completed.\n");
}

void sharedBlurring(unsigned char *image, unsigned char *imageOutput, int width, int height) {
    printf("Starting shared blurring...\n");

    int blockSize = 16;
    int paddedWidth = width + 2 * BLUR_SIZE;
    int paddedHeight = height + 2 * BLUR_SIZE;

    unsigned char *sharedChunk = (unsigned char *)calloc(paddedWidth * paddedHeight * 3, sizeof(unsigned char));
    if (!sharedChunk) {
        perror("Error allocating shared memory");
        exit(EXIT_FAILURE);
    }

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int index = (row * width + col) * 3;
            if (index + 2 >= width * height * 3) {
                fprintf(stderr, "Error: Out of bounds access at index %d\n", index);
                exit(EXIT_FAILURE);
            }

            float averageR = 0.0f, averageG = 0.0f, averageB = 0.0f;
            int count = 0;

            for (int i = -BLUR_SIZE; i <= BLUR_SIZE; i++) {
                for (int j = -BLUR_SIZE; j <= BLUR_SIZE; j++) {
                    int blurRow = row + i;
                    int blurCol = col + j;

                    if (blurRow >= 0 && blurRow < height && blurCol >= 0 && blurCol < width) {
                        int blurIndex = (blurRow * width + blurCol) * 3;
                        averageR += image[blurIndex];
                        averageG += image[blurIndex + 1];
                        averageB += image[blurIndex + 2];
                        count++;
                    }
                }
            }

            int outputIndex = (row * width + col) * 3;
            imageOutput[outputIndex] = (unsigned char)(averageR / count);
            imageOutput[outputIndex + 1] = (unsigned char)(averageG / count);
            imageOutput[outputIndex + 2] = (unsigned char)(averageB / count);
        }
    }

    free(sharedChunk);
    printf("Shared blurring completed.\n");
}

void readImage(char *filename, unsigned char **image) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    char format[3];
    fscanf(file, "%2s\n", format);
    if (strcmp(format, "P6") != 0) {
        fprintf(stderr, "Only P6 PPM format is supported.\n");
        exit(EXIT_FAILURE);
    }

    int width, height, maxColorValue;
    fscanf(file, "%d %d\n%d\n", &width, &height, &maxColorValue);

    if (width <= 0 || height <= 0) {
        fprintf(stderr, "Invalid image dimensions.\n");
        exit(EXIT_FAILURE);
    }

    xDimension = width;
    yDimension = height;

    *image = (unsigned char *)malloc(width * height * 3);
    if (!*image) {
        perror("Memory allocation failed for image");
        exit(EXIT_FAILURE);
    }

    fread(*image, 3, width * height, file);
    fclose(file);
}

void writeImage(unsigned char *image, char *filename, int width, int height) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Error opening output file");
        exit(EXIT_FAILURE);
    }

    fprintf(file, "P6\n%d %d\n255\n", width, height);
    fwrite(image, 3, width * height, file);
    fclose(file);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: ./cpu <input_image> <output_image> <blur_type (0 for unshared, 1 for shared)>\n");
        return EXIT_FAILURE;
    }

    char *inputImageFile = argv[1];
    char *outputImageFile = argv[2];
    int blurType = atoi(argv[3]);

    unsigned char *hostImage;
    unsigned char *hostImageOutput = (unsigned char *)calloc(xDimension * yDimension * 3, sizeof(unsigned char));
    if (!hostImageOutput) {
        perror("Memory allocation failed for output image");
        exit(EXIT_FAILURE);
    }

    readImage(inputImageFile, &hostImage);

    if (!hostImage) {
        fprintf(stderr, "Error: Image data is NULL.\n");
        exit(EXIT_FAILURE);
    }

    printf("Image dimensions: %d x %d\n", xDimension, yDimension);

    if (blurType == 0) {
        printf("Using unshared memory for blurring...\n");
        unsharedBlurring(hostImage, hostImageOutput, xDimension, yDimension);
    } else if (blurType == 1) {
        printf("Using shared memory for blurring...\n");
        sharedBlurring(hostImage, hostImageOutput, xDimension, yDimension);
    } else {
        fprintf(stderr, "Invalid blur type. Use 0 (unshared) or 1 (shared).\n");
        return EXIT_FAILURE;
    }

    writeImage(hostImageOutput, outputImageFile, xDimension, yDimension);

    free(hostImage);
    free(hostImageOutput);

    return EXIT_SUCCESS;
}
