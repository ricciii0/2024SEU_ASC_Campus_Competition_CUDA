// =========================================================
// Copyright © 2024 SEU Linux Association ASC Programming Contest Team
// All rights reserved.
// =========================================================
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstdlib>

// 定义错误检查宏
#define CUDA_CHECK(error) \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// 定义MNIST数据结构
struct MNISTData {
    std::vector<float> images;          // 标准化后的图像数据
    std::vector<uint8_t> labels;        // 标签数据
    int num_samples;
    int image_size;                     // 每张图像的像素数量（28*28=784）
};

// 函数声明
MNISTData load_mnist(const std::string& image_path, const std::string& label_path);
__global__ void matrix_multiplication(float* A, float* B, float* C, int N);
__global__ void vector_addition(float* A, float* B, float* C, int N);

// 主函数
int main() {
    // 加载MNIST数据
    std::string image_path = "data/train-images-idx3-ubyte";
    std::string label_path = "data/train-labels-idx1-ubyte";
    MNISTData train_data = load_mnist(image_path, label_path);
    std::cout << "Loaded " << train_data.num_samples << " training samples." << std::endl;

    // 定义矩阵规模
    int N = 4096;
    size_t size = N * N * sizeof(float);
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    if (!h_A || !h_B || !h_C) {
        std::cerr << "Host memory allocation failed!" << std::endl;
        return EXIT_FAILURE;
    }

    // 初始化矩阵
    for(int i = 0; i < N * N; ++i) {
        if(i < train_data.images.size()) {
            h_A[i] = train_data.images[i];
            h_B[i] = train_data.images[i] * 2.0f;
        } else {
            h_A[i] = 0.0f;
            h_B[i] = 0.0f;
        }
    }

    // 打印初始化后的部分数据
    std::cout << "Initialization:" << std::endl;
    std::cout << "h_A[0] = " << h_A[0] << ", h_B[0] = " << h_B[0] << std::endl;
    std::cout << "h_A[1] = " << h_A[1] << ", h_B[1] = " << h_B[1] << std::endl;

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaError_t err;
    err = cudaMalloc(&d_A, size);
    CUDA_CHECK(err);
    err = cudaMalloc(&d_B, size);
    CUDA_CHECK(err);
    err = cudaMalloc(&d_C, size);
    CUDA_CHECK(err);

    // 拷贝数据到设备
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);

    // 定义线程块和网格尺寸
    dim3 block(16, 16);
    dim3 grid((N + block.x -1)/block.x, (N + block.y -1)/block.y);

    // 定义循环次数
    int iterations = 10; 

    // 向量加法初始化
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
        if(i < train_data.images.size()) {
            h_VA[i] = train_data.images[i];
            h_VB[i] = train_data.images[i] * 2.0f;
        } else {
            h_VA[i] = 1.0f;
            h_VB[i] = 2.0f;
        }
    }

    // 打印向量初始化后的部分数据
    std::cout << "Vector Initialization:" << std::endl;
    std::cout << "h_VA[0] = " << h_VA[0] << ", h_VB[0] = " << h_VB[0] << std::endl;
    std::cout << "h_VA[1] = " << h_VA[1] << ", h_VB[1] = " << h_VB[1] << std::endl;

    // 分配设备向量内存
    float *d_VA, *d_VB, *d_VC;
    err = cudaMalloc(&d_VA, vec_bytes);
    CUDA_CHECK(err);
    err = cudaMalloc(&d_VB, vec_bytes);
    CUDA_CHECK(err);
    err = cudaMalloc(&d_VC, vec_bytes);
    CUDA_CHECK(err);

    // 拷贝向量数据到设备
    err = cudaMemcpy(d_VA, h_VA, vec_bytes, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);
    err = cudaMemcpy(d_VB, h_VB, vec_bytes, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);

    // 定义向量加法线程块和网格尺寸
    int threads_per_block = 8;
    int blocks_per_grid = (vec_size + threads_per_block - 1) / threads_per_block;

    // 记录开始时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 执行多次矩阵乘法和向量加法
    for(int i = 0; i < iterations; ++i) {
        matrix_multiplication<<<grid, block>>>(d_A, d_B, d_C, N);
        err = cudaGetLastError();
        CUDA_CHECK(err);

        vector_addition<<<blocks_per_grid, threads_per_block>>>(d_VA, d_VB, d_VC, vec_size);
        err = cudaGetLastError();
        CUDA_CHECK(err);
    }

    // 等待所有核函数执行完毕
    err = cudaDeviceSynchronize();
    CUDA_CHECK(err);

    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Total time for " << iterations << " iterations: " << milliseconds << " ms" << std::endl;

    // 拷贝结果回主机
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);
    err = cudaMemcpy(h_VC, d_VC, vec_bytes, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);

    // 打印结果验证
    std::cout << "C[0] = " << h_C[0] << ", VC[0] = " << h_VC[0] << std::endl;
    std::cout << "C[1] = " << h_C[1] << ", VC[1] = " << h_VC[1] << std::endl;

    // 释放内存
    free(h_A); free(h_B); free(h_C);
    free(h_VA); free(h_VB); free(h_VC);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_VA); cudaFree(d_VB); cudaFree(d_VC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

// 加载MNIST数据的实现
MNISTData load_mnist(const std::string& image_path, const std::string& label_path) {
    MNISTData data;

    // 读取标签文件
    std::ifstream label_file(label_path, std::ios::binary);
    if (!label_file.is_open()) {
        std::cerr << "无法打开标签文件：" << label_path << std::endl;
        exit(EXIT_FAILURE);
    }

    // 读取标签文件头部
    uint32_t magic_number = 0;
    uint32_t num_labels = 0;
    label_file.read(reinterpret_cast<char*>(&magic_number), 4);
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    magic_number = __builtin_bswap32(magic_number);
    num_labels = __builtin_bswap32(num_labels);

    if(magic_number != 2049) {
        std::cerr << "错误的标签文件格式！" << std::endl;
        exit(EXIT_FAILURE);
    }

    data.num_samples = num_labels;

    // 读取标签
    data.labels.resize(num_labels);
    label_file.read(reinterpret_cast<char*>(data.labels.data()), num_labels);
    label_file.close();

    // 读取图像文件
    std::ifstream image_file(image_path, std::ios::binary);
    if (!image_file.is_open()) {
        std::cerr << "无法打开图像文件：" << image_path << std::endl;
        exit(EXIT_FAILURE);
    }

    // 读取图像文件头部
    uint32_t image_magic = 0;
    uint32_t num_images = 0;
    uint32_t num_rows = 0;
    uint32_t num_cols = 0;
    image_file.read(reinterpret_cast<char*>(&image_magic), 4);
    image_file.read(reinterpret_cast<char*>(&num_images), 4);
    image_file.read(reinterpret_cast<char*>(&num_rows), 4);
    image_file.read(reinterpret_cast<char*>(&num_cols), 4);
    image_magic = __builtin_bswap32(image_magic);
    num_images = __builtin_bswap32(num_images);
    num_rows = __builtin_bswap32(num_rows);
    num_cols = __builtin_bswap32(num_cols);

    if(image_magic != 2051) {
        std::cerr << "错误的图像文件格式！" << std::endl;
        exit(EXIT_FAILURE);
    }

    if(num_images != num_labels) {
        std::cerr << "图像数量与标签数量不匹配！" << std::endl;
        exit(EXIT_FAILURE);
    }

    data.image_size = num_rows * num_cols;

    // 读取图像
    data.images.resize(num_images * data.image_size);
    std::vector<uint8_t> temp_images(num_images * data.image_size);
    image_file.read(reinterpret_cast<char*>(temp_images.data()), num_images * data.image_size);
    image_file.close();

    // 标准化图像数据到0.0 - 1.0
    for(int i = 0; i < num_images * data.image_size; ++i) {
        data.images[i] = static_cast<float>(temp_images[i]) / 255.0f;
    }

    return data;
}

// 简单的矩阵乘法核函数
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

// 向量加法核函数
__global__ void vector_addition(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        C[idx] = A[idx] + B[idx];
    }
} 