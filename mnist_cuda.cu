// =========================================================
// Copyright © 2024 SEU Linux Association ASC Programming Contest Team
// All rights reserved.
// =========================================================

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// Define error checking macro
#define CUDA_CHECK(error) \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Matrix multiplication kernel function
__global__ void matrix_multiplication(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < N && col < N) {
        float sum = 0.0f;
        for(int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Vector addition kernel function
__global__ void vector_addition(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // Define matrix size
    int N = 8192; // Adjust size based on GPU memory, e.g., 4096
    size_t size = N * N * sizeof(float);
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    if (!h_A || !h_B || !h_C) {
        std::cerr << "Host memory allocation failed!" << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize matrices
    for(int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f; // Simplified initialization
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaError_t err;
    err = cudaMalloc(&d_A, size);
    CUDA_CHECK(err);
    err = cudaMalloc(&d_B, size);
    CUDA_CHECK(err);
    err = cudaMalloc(&d_C, size);
    CUDA_CHECK(err);

    // Copy data to device
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);

    // Define block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((N + block.x -1)/block.x, (N + block.y -1)/block.y);

    // Define number of iterations
    int iterations = 10; // Adjust number of iterations as needed

    // Initialize vectors for addition
    int vec_size = N;
    size_t vec_bytes = vec_size * sizeof(float);
    float *h_VA, *h_VB, *h_VC;
    h_VA = (float*)malloc(vec_bytes);
    h_VB = (float*)malloc(vec_bytes);
    h_VC = (float*)malloc(vec_bytes);

    if (!h_VA || !h_VB || !h_VC) {
        std::cerr << "Host vector memory allocation failed!" << std::endl;
        return EXIT_FAILURE;
    }

    for(int i = 0; i < vec_size; ++i) {
        h_VA[i] = 1.0f; // Simplified initialization
        h_VB[i] = 2.0f;
    }

    // Allocate device memory for vectors
    float *d_VA, *d_VB, *d_VC;
    err = cudaMalloc(&d_VA, vec_bytes);
    CUDA_CHECK(err);
    err = cudaMalloc(&d_VB, vec_bytes);
    CUDA_CHECK(err);
    err = cudaMalloc(&d_VC, vec_bytes);
    CUDA_CHECK(err);

    // Copy vectors to device
    err = cudaMemcpy(d_VA, h_VA, vec_bytes, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);
    err = cudaMemcpy(d_VB, h_VB, vec_bytes, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);

    // Define thread block and grid dimensions for vector addition
    int threads_per_block = 256;
    int blocks_per_grid = (vec_size + threads_per_block - 1) / threads_per_block;

    // Record start time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Execute multiple iterations of matrix multiplication and vector addition
    for(int i = 0; i < iterations; ++i) {
        matrix_multiplication<<<grid, block>>>(d_A, d_B, d_C, N);
        err = cudaGetLastError();
        CUDA_CHECK(err);

        vector_addition<<<blocks_per_grid, threads_per_block>>>(d_VA, d_VB, d_VC, vec_size);
        err = cudaGetLastError();
        CUDA_CHECK(err);
    }

    // Wait for all kernels to finish
    err = cudaDeviceSynchronize();
    CUDA_CHECK(err);

    // Record end time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Total time for " << iterations << " iterations: " << milliseconds << " ms" << std::endl;

    // Copy results back to host
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);
    err = cudaMemcpy(h_VC, d_VC, vec_bytes, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);

    // Print results for verification
    std::cout << "C[0] = " << h_C[0] << ", VC[0] = " << h_VC[0] << std::endl;
    std::cout << "C[1] = " << h_C[1] << ", VC[1] = " << h_VC[1] << std::endl;

    // Free memory
    free(h_A); free(h_B); free(h_C);
    free(h_VA); free(h_VB); free(h_VC);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_VA); cudaFree(d_VB); cudaFree(d_VC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
