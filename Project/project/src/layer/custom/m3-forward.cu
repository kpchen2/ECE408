#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "matmul.h"

#define PERMUTE_BLOCK_SIZE 256
#define TILE_WIDTH 16

__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Width_unrolled = Batch * Height_out * Width_out;

    size_t s = blockIdx.x * blockDim.x + threadIdx.x;
    size_t c = blockIdx.y;
    size_t b = blockIdx.z;

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    if (s < Height_out * Width_out) {
        size_t h_out = s / Width_out;
        size_t w_out = s % Width_out;

        for (size_t p = 0; p < K; p++) {
            for (size_t q = 0; q < K; q++) {
                output[(c*K*K + p*K + q)*Width_unrolled + b*(Height_out*Width_out) + h_out*Width_out + w_out] =
                    in_4d(b, c, h_out+p, w_out+q);
            }
        }
    }

    #undef in_4d
}

__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask,
                                                    float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr,
                                                    const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) {
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Height_unrolled = Channel * K * K;
    const int Width_unrolled = Batch * Height_out * Width_out;

    float* matmul_output;
    float* device_output;
    float* device_mask;

    cudaMalloc((void**) &matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(float));
    cudaMalloc((void**) &device_output, ((Height-K+1) * (Width-K+1) * Batch * Map_out) * sizeof(float));
    cudaMalloc((void**) &device_mask , (K * K * Map_out * Channel) * sizeof(float));
    cudaMemcpy(device_mask, host_mask, (K * K * Map_out * Channel) * sizeof(float), cudaMemcpyHostToDevice);

    const int numStreams = 1;
    float* pinned_input[numStreams];
    float* device_input[numStreams];
    float* unrolled_matrix[numStreams];
    cudaStream_t streams[numStreams];

    const int input_bytes = ((Batch-1)/numStreams + 1) * Channel * Width * Height * sizeof(float);
    const int unrolled_bytes = ((Batch-1)/numStreams + 1) * Channel * K * K * Height_out * Width_out * sizeof(float);

    for (int i = 0; i < numStreams; i++) {
        cudaMallocHost((void**)&pinned_input[i], input_bytes);
        memcpy(pinned_input[i], host_input + i * (input_bytes / sizeof(float)), input_bytes);
    
        cudaMalloc((void**)&device_input[i], input_bytes);
        cudaMalloc((void**)&unrolled_matrix[i], unrolled_bytes);
        cudaStreamCreate(&streams[i]);
    }

    int temp = (Height_out*Width_out - 1) / TILE_WIDTH + 1;
    dim3 dimGrid(temp, Channel, Batch);
    dim3 dimBlock(TILE_WIDTH, 1, 1);

    for (int i = 0; i < numStreams; i++) {
        cudaMemcpyAsync(device_input[i], pinned_input[i], input_bytes, cudaMemcpyHostToDevice, streams[i]);
        matrix_unrolling_kernel<<<dimGrid, dimBlock, 0, streams[i]>>>(
            device_input[i], unrolled_matrix[i], Batch, Channel, Height, Width, K);
    }

    for (int i = 0; i < numStreams; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    dim3 matmul_grid_dim((Width_unrolled - 1) / MATMUL_TILE_WIDTH + 1, (Map_out - 1) / MATMUL_TILE_WIDTH + 1, 1);
    dim3 matmul_block_dim(MATMUL_TILE_WIDTH, MATMUL_TILE_WIDTH, 1);

    matrixMultiplyShared<<<matmul_grid_dim, matmul_block_dim>>>(
        device_mask, unrolled_matrix[0], matmul_output, Map_out, Height_unrolled,
        Height_unrolled, Width_unrolled, Map_out, Width_unrolled);

    const int out_image_size = Height_out * Width_out;
    dim3 permute_kernel_grid_dim((out_image_size - 1) / PERMUTE_BLOCK_SIZE + 1, Batch, 1);

    matrix_permute_kernel<<<permute_kernel_grid_dim, PERMUTE_BLOCK_SIZE>>>(
        matmul_output, device_output, Map_out, Batch, out_image_size);

    cudaFree(matmul_output);

    *device_output_ptr = device_output;
    *device_input_ptr = device_input[0];
    *device_mask_ptr = device_mask;

    for (int i = 0; i < numStreams; i++) {
        cudaFree(unrolled_matrix[i]);
        cudaFree(device_input[i]);
        cudaFreeHost(pinned_input[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // do nothing
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // do nothing
}


__host__ void GPUInterface::get_device_properties()
{
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