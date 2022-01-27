
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <random>
#include <cassert>
#include <iomanip>

// Default 4 component vector
struct Vector
{
    float x, y, z, w;
};

// CPU Vector addition KERNEL
void CpuVectorArrAdd(const Vector* arrA, const Vector* arrB, Vector* arrR, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        arrR[i].x = arrA[i].x + arrB[i].x;
        arrR[i].y = arrA[i].y + arrB[i].y;
        arrR[i].z = arrA[i].z + arrB[i].z;
        arrR[i].w = arrA[i].w + arrB[i].w;
    }
}

__global__ void GpuVectorArrAdd(const Vector* arrA, const Vector* arrB, Vector* arrR)
{
    // Calculare index
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;

    // Do computations
    arrR[i].x = arrA[i].x + arrB[i].x;
    arrR[i].y = arrA[i].y + arrB[i].y;
    arrR[i].z = arrA[i].z + arrB[i].z;
    arrR[i].w = arrA[i].w + arrB[i].w;
}

int main()
{
    // Print some infos
    const size_t WORKINGSET_COUT = 256 * 1024;
    std::cout << " === THIS IS OUR FIRST CUDA APPLICATION === " << std::endl;

    // Prepare a workset 
    std::cout << "Preparing workset..." << std::endl;
    Vector* arrVecA = (Vector*)malloc(sizeof(Vector) * WORKINGSET_COUT);
    Vector* arrVecB = (Vector*)malloc(sizeof(Vector) * WORKINGSET_COUT);
    Vector* arrVecR = (Vector*)malloc(sizeof(Vector) * WORKINGSET_COUT);
    assert(arrVecA && arrVecB && arrVecR && "Malloc failed for working set");
    std::srand(std::time(nullptr));
    for (size_t i = 0; i < WORKINGSET_COUT; i++)
    {
        arrVecA[i].x = 1.0f / (std::rand() % 200 + 1);
        arrVecB[i].x = 1.0f / (std::rand() % 200 + 1);
        arrVecA[i].y = 1.0f / (std::rand() % 200 + 1);
        arrVecB[i].y = 1.0f / (std::rand() % 200 + 1);
        arrVecA[i].z = 1.0f / (std::rand() % 200 + 1);
        arrVecB[i].z = 1.0f / (std::rand() % 200 + 1);
        arrVecA[i].w = 1.0f / (std::rand() % 200 + 1);
        arrVecB[i].w = 1.0f / (std::rand() % 200 + 1);
    }

    // Process the data
    std::cout << "Computing Result..." << std::endl;
    // CpuVectorArrAdd(arrVecA, arrVecB, arrVecR, WORKINGSET_COUT);
    // -- Allocate gpu memory
    Vector* gpuArrVecA = nullptr, *gpuArrVecB = nullptr, *gpuArrVecR = nullptr;
    cudaMalloc(&gpuArrVecA, sizeof(Vector) * WORKINGSET_COUT);
    cudaMalloc(&gpuArrVecB, sizeof(Vector) * WORKINGSET_COUT);
    cudaMalloc(&gpuArrVecR, sizeof(Vector) * WORKINGSET_COUT);
    assert(gpuArrVecA && gpuArrVecB && gpuArrVecR && "cudaMalloc failed for working set");
    // -- Copy arrays to gpu
    cudaMemcpy(gpuArrVecA, arrVecA, sizeof(Vector) * WORKINGSET_COUT, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuArrVecB, arrVecB, sizeof(Vector) * WORKINGSET_COUT, cudaMemcpyHostToDevice);
    // -- Execute gpu kernel
    const size_t blockSize = 256;
    const size_t blockCount = WORKINGSET_COUT / blockSize;
    GpuVectorArrAdd<<<blockCount, blockSize>>>(gpuArrVecA, gpuArrVecB, gpuArrVecR);
    // -- Fetch result from gpu
    cudaMemcpy(arrVecR, gpuArrVecR, sizeof(Vector) * WORKINGSET_COUT, cudaMemcpyDeviceToHost);

    // Print some values from result
    for (size_t i = 0; i < 5; i++)
    {
        std::cout << std::setw(8) << std::fixed <<
            "X=" << arrVecR[i].x << "\tY=" << arrVecR[i].y << "\tZ=" << arrVecR[i].z << "\tW=" << arrVecR[i].w << std::endl;
    }
    std::cout << "..." << std::endl;
    for (size_t i = WORKINGSET_COUT - 5; i < WORKINGSET_COUT; i++)
    {
        std::cout << std::setw(8) << std::fixed <<
            "X=" << arrVecR[i].x << "\tY=" << arrVecR[i].y << "\tZ=" << arrVecR[i].z << "\tW=" << arrVecR[i].w << std::endl;
    }

    // Free memory
    cudaFree(gpuArrVecA);
    cudaFree(gpuArrVecB);
    cudaFree(gpuArrVecR);
    free(arrVecA);
    free(arrVecB);
    free(arrVecR);
}
