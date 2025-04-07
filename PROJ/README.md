# CUDA Image Blur

A CUDA-accelerated image blurring application that demonstrates both shared and unshared memory implementations for efficient image processing.

## Overview

This application applies a blur effect to a PPM image using CUDA parallel processing capabilities. It offers two different implementation approaches:

1. **Unshared Memory Implementation**: Each thread independently processes its assigned pixel, fetching neighbor pixels directly from global memory.

2. **Shared Memory Implementation**: Optimized version that loads blocks of the image into shared memory to reduce global memory access latency.

## Features

- Fast parallel image blurring using GPU acceleration
- Support for standard PPM (P6) format images
- Two memory optimization approaches for performance comparison
- Configurable blur radius (via BLUR_SIZE constant)

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (compatible version)
- C compiler

## Usage

```bash
./imageBlur <input_image> <output_image> <blur_type>
```

Where:

<input_image>: Path to the input PPM image (e.g., input.ppm)
<output_image>: Path where the blurred image will be saved
<blur_type>: Blur implementation to use

0: Unshared memory implementation
1: Shared memory implementation

Example

```bash
./imageBlur input.ppm blurred_output.ppm 1
```

This will process input.ppm using the shared memory implementation and save the result as blurred_output.ppm.
Input Image Format
The application supports PPM (P6) format images. This is a simple uncompressed image format that consists of:

A header with format identifier, dimensions, and maximum color value
Raw RGB pixel data

Implementation Details
Memory Models
Unshared Memory
Each thread:

Calculates the position of its assigned pixel
Fetches neighbor pixels directly from global memory
Computes the average color values
Writes the result to the output image

Shared Memory
Each thread block:

Loads a chunk of the image into shared memory
Synchronizes to ensure all data is loaded
Processes pixels using the faster shared memory
Writes results back to global memory

Performance Considerations
The shared memory implementation generally provides better performance due to:

Reduced global memory access latency
Better memory access coalescing
Lower memory bandwidth requirements

Building from Source

nvcc -o imageBlur imageBlur.cu
