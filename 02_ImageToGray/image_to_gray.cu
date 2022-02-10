
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <cassert>

#include "stb_image.h"
#include "stb_image_write.h"

struct Pixel
{
    unsigned char r, g, b, a;
};

void ConvertImageToGrayCpu(unsigned char* imageRGBA, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            Pixel* ptrPixel = (Pixel*)&imageRGBA[y * width * 4 + 4 * x];
            unsigned char pixelValue = (unsigned char)(ptrPixel->r * 0.2126f + ptrPixel->g * 0.7152f + ptrPixel->b * 0.0722f);
            ptrPixel->r = pixelValue;
            ptrPixel->g = pixelValue;
            ptrPixel->b = pixelValue;
            ptrPixel->a = 255;
        }
    }
}

__global__ void ConvertImageToGrayGpu(unsigned char* imageRGBA)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t idx = y * blockDim.x * gridDim.x + x;

    Pixel* ptrPixel = (Pixel*)&imageRGBA[idx * 4];
    unsigned char pixelValue = (unsigned char)
        (ptrPixel->r * 0.2126f + ptrPixel->g * 0.7152f + ptrPixel->b * 0.0722f);
    ptrPixel->r = pixelValue;
    ptrPixel->g = pixelValue;
    ptrPixel->b = pixelValue;
    ptrPixel->a = 255;
}

int main(int argc, char** argv)
{
    // Check argument count
    if (argc < 2)
    {
        std::cout << "Usage: 02_ImageToGray <filename>";
        return -1;
    }

    // Open image
    int width, height, componentCount;
    std::cout << "Loading png file...";
    unsigned char* imageData = stbi_load(argv[1], &width, &height, &componentCount, 4);
    if (!imageData)
    {
        std::cout << std::endl << "Failed to open \"" << argv[1] << "\"";
        return -1;
    }
    std::cout << " DONE" << std::endl;

    // Validate image sizes
    if (width % 32 || height % 32)
    {
        // NOTE: Leaked memory of "imageData"
        std::cout << "Width and/or Height is not dividable by 32!";
        return -1;
    }

    /*
    // Process image on cpu
    std::cout << "Processing image...";
    ConvertImageToGrayCpu(imageData, width, height);
    std::cout << " DONE" << std::endl;
    */

    // Copy data to the gpu
    std::cout << "Copy data to GPU...";
    unsigned char* ptrImageDataGpu = nullptr;
    assert(cudaMalloc(&ptrImageDataGpu, width * height * 4) == cudaSuccess);
    assert(cudaMemcpy(ptrImageDataGpu, imageData, width * height * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    std::cout << " DONE" << std::endl;

    // Process image on gpu
    std::cout << "Running CUDA Kernel...";
    dim3 blockSize(32, 32);
    dim3 gridSize(width / blockSize.x, height / blockSize.y);
    ConvertImageToGrayGpu<<<gridSize, blockSize>>>(ptrImageDataGpu);
    auto err = cudaGetLastError();
    std::cout << " DONE" << std::endl; 

    // Copy data from the gpu
    std::cout << "Copy data from GPU...";
    assert(cudaMemcpy(imageData, ptrImageDataGpu, width * height * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    std::cout << " DONE" << std::endl;

    // Build output filename
    std::string fileNameOut = argv[1];
    fileNameOut = fileNameOut.substr(0, fileNameOut.find_last_of('.')) + "_gray.png";

    // Write image back to disk
    std::cout << "Writing png to disk...";
    stbi_write_png(fileNameOut.c_str(), width, height, 4, imageData, 4 * width);
    std::cout << " DONE";

    // Free memory
    cudaFree(ptrImageDataGpu);
    stbi_image_free(imageData);
}
