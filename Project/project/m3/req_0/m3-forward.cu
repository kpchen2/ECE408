#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "matmul.h"

#define TILE_WIDTH 16
#define PERMUTE_BLOCK_SIZE 256

__global__ void matrix_unrolling_kernel(const float *input, float *output, int Channel, int Height, int Width, int K, int Height_out, int Width_out, size_t batch_id, size_t Batch) {
    size_t s = blockIdx.x * blockDim.x + threadIdx.x;
    size_t c = blockIdx.y;

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * Width + i0]

    if (s < Height_out * Width_out) {
        size_t h_out = s / Width_out;
        size_t w_out = s % Width_out;

        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                size_t row = c * K * K + p * K + q;
                size_t col = batch_id * Height_out * Width_out + h_out * Width_out + w_out;
                
                output[row * (Batch * Height_out * Width_out) + col] = in_4d(batch_id, c, h_out + p, w_out + q);
            }
        }
    }

    #undef in_4d
}

__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out, int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] = input[m * Batch * image_size + b * image_size + x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Height_unrolled = Channel * K * K;
    const size_t Width_unrolled = (size_t)Batch * Height_out * Width_out;

    float *device_input, *device_output, *device_mask, *unrolled_input, *matmul_output;

    cudaMalloc(&device_input, sizeof(float) * Batch * Channel * Height * Width);
    cudaMalloc(&device_output, sizeof(float) * Batch * Map_out * Height_out * Width_out);
    cudaMalloc(&device_mask, sizeof(float) * Map_out * Channel * K * K);
    cudaMalloc(&unrolled_input, sizeof(float) * Height_unrolled * Width_unrolled);
    cudaMalloc(&matmul_output, sizeof(float) * Map_out * Width_unrolled);

    cudaMemcpyAsync(device_input, host_input, sizeof(float) * Batch * Channel * Height * Width, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(device_mask, host_mask, sizeof(float) * Map_out * Channel * K * K, cudaMemcpyHostToDevice);

    const int num_streams = 4;
    cudaStream_t streams[num_streams];

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 unroll_block_dim(TILE_WIDTH, 1, 1);
    dim3 unroll_grid_dim((Height_out * Width_out + TILE_WIDTH - 1) / TILE_WIDTH, Channel, 1);

    for (size_t b = 0; b < (size_t)Batch; b++) {
        int stream_id = b % num_streams;
        matrix_unrolling_kernel<<<unroll_grid_dim, unroll_block_dim, 0, streams[stream_id]>>>(device_input, unrolled_input, Channel, Height, Width, K, Height_out, Width_out, b, Batch);
    }

    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    dim3 matmul_grid_dim((Width_unrolled - 1) / MATMUL_TILE_WIDTH + 1, (Map_out - 1) / MATMUL_TILE_WIDTH + 1, 1);
    dim3 matmul_block_dim(MATMUL_TILE_WIDTH, MATMUL_TILE_WIDTH, 1);

    matrixMultiplyShared<<<matmul_grid_dim, matmul_block_dim>>>(device_mask, unrolled_input, matmul_output, Map_out, Height_unrolled, Height_unrolled, Width_unrolled, Map_out, Width_unrolled);

    dim3 permute_grid_dim((Height_out * Width_out + PERMUTE_BLOCK_SIZE - 1) / PERMUTE_BLOCK_SIZE, Batch, 1);
    matrix_permute_kernel<<<permute_grid_dim, PERMUTE_BLOCK_SIZE>>>(matmul_output, device_output, Map_out, Batch, Height_out * Width_out);

    cudaFree(unrolled_input);
    cudaFree(matmul_output);

    *device_input_ptr = device_input;
    *device_output_ptr = device_output;
    *device_mask_ptr = device_mask;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
    // do nothing
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    cudaMemcpy(host_output, device_output, sizeof(float) * Batch * Map_out * Height_out * Width_out, cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
}

__host__ void GPUInterface::get_device_properties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
