#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>

#define BLOCK_SIZE 32

void getInfoCUDADevice(cudaDeviceProp& prop, int id) {
    printf("CUDA device %i name  - %s\n", id, prop.name);
    printf("CUDA device %i Warp size in threads  - %i\n", id, prop.warpSize);
    printf("CUDA device %i Maximum number of threads per block  - %i\n", id, prop.maxThreadsPerBlock);
    printf("CUDA device %i multiprocessors count  - %i\n", id, prop.multiProcessorCount);
    printf("CUDA device %i Maximum size of each dimension of a block  - %i %i %i\n", id, prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("CUDA device %i Maximum size of each dimension of a grid  - %i %i %i\n", id, prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
}

__global__ void matrixMult(const int16_t *A, const int16_t *B, int16_t *result, int size) {
    int bx = blockIdx.x;  
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int ia = size * (gridDim.y * by + ty);
    int ib = gridDim.x * bx + tx;
    int ic = ia + ib;
    
    int16_t sum = 0;
    
    for (int k = 0; k < size; k++) {
        sum += A[ia + k] * B[k * size + ib];
    }
    result[ic] = sum;
}

void compareMatrix(const int16_t* f, const int16_t* s, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (f[i * size + j] != s[i * size + j]) {
                std::cout << "Matrices are not equal\n";
                return;
            }
        }
    }
    std::cout << "Matrices are equal\n";
}

int main() {
    int count;
    cudaDeviceProp prop;
    cudaGetDeviceCount(&count);
    printf("Count CUDA devices - %i\n", count);
    cudaGetDeviceProperties(&prop, count - 1);
    getInfoCUDADevice(prop, count - 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);    
    int size = 1024;
    for (int iter = 0; iter < 10; iter++) {
        std::cout << "Itteration number: " << iter + 1 << "\n";
        size_t byte_size = size * size * sizeof(int16_t);

        int16_t* h_A = (int16_t*)malloc(byte_size);
        int16_t* h_B = (int16_t*)malloc(byte_size);
        int16_t* h_C = (int16_t*)malloc(byte_size);
        int16_t* CPU_C = (int16_t*)malloc(byte_size);

        for (int i = 0; i < size * size; ++i) {
            h_A[i] = rand() % 100;
            h_B[i] = rand() % 100;
            CPU_C[i] = 0;
        }

        std::cout << "CPU: \n";
        cudaEventRecord(start, 0);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                for (int k = 0; k < size; ++k) {
                    CPU_C[i * size + j] += h_A[i * size + k] * h_B[k * size + j];
                }
            }
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float result_time_cpu;
        cudaEventElapsedTime(&result_time_cpu, start, stop);
		std::cout << "Elapsed time: " << result_time_cpu << " ms\n";

        std::cout << "GPU: \n";

        int16_t* d_A = NULL;
        cudaMalloc((void**)&d_A, byte_size);
        cudaMemcpy(d_A, h_A, byte_size, cudaMemcpyHostToDevice);

        int16_t* d_B = NULL;
        cudaMalloc((void**)&d_B, byte_size);
        cudaMemcpy(d_B, h_B, byte_size, cudaMemcpyHostToDevice);

        int16_t* d_C = NULL;
        cudaMalloc((void**)&d_C, byte_size);

        cudaEventRecord(start, 0);

        const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        const dim3 grid(size / block.x, size / block.y);
        matrixMult <<< grid, block >>> (d_A, d_B, d_C, size);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float result_time_gpu;
        cudaEventElapsedTime(&result_time_gpu, start, stop);
        std::cout << "Elapsed time: " << result_time_gpu << " ms\n";

        cudaMemcpy(h_C, d_C, byte_size, cudaMemcpyDeviceToHost);
        compareMatrix(h_C, CPU_C, size);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);     
        free(h_C); 
        free(CPU_C);
    }
    cudaEventDestroy(start);  
    cudaEventDestroy(stop);

    return 0;
}
