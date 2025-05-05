#include <cmath>
#include <iostream>
#include <cublas_v2.h>
#include "gpu-new-forward.h"
#include "matmul.h"

#define PERMUTE_BLOCK_SIZE 256
#define TILE_WIDTH 16

__global__ void matrix_unrolling_kernel(const float *input, float *output,
    const int Batch, const int Channel,
    const int Height, const int Width,
    const int K) {
    /*
    Modify this function to implement the input matrix unrolling kernel.

    Function paramter definitions:
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Width_unrolled = Batch * Height_out * Width_out;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    // TODO: Insert your input matrix unrolling kernel code here
    size_t s = blockIdx.x * blockDim.x + threadIdx.x;
    size_t c = blockIdx.y;
    size_t b = blockIdx.z;

    if (s < Height_out * Width_out) {
        size_t h_out = s / Width_out;
        size_t w_out = s % Width_out;

        for (size_t p = 0; p < K; p++) {
            for (size_t q = 0; q < K; q++) {
                output[(c*K*K + p*K + q)*Width_unrolled + b*(Height_out*Width_out) + h_out*Width_out + w_out] = in_4d(b, c, h_out+p, w_out+q);
            }
        }
    }

    #undef in_4d
}


// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
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

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int h_out = Height - K + 1;
    const int w_out = Width - K + 1;

    cudaMalloc((void**)device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void**)device_output_ptr, Batch * Map_out * h_out * w_out * sizeof(float));
    cudaMalloc((void**)device_mask_ptr, Map_out * Channel * K * K * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_output_ptr, host_output, Batch * Map_out * h_out * w_out * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int H_out = Height - K + 1;
    int W_out = Width - K + 1;
    int Width_unroll = Batch * H_out * W_out;
    int Height_unroll = Channel * K * K;
    int M = Map_out;

    float *unrolled_matrix;
    float *matmul_output;

    cudaMalloc(&unrolled_matrix, (size_t)Height_unroll * Width_unroll * sizeof(float));
    cudaMalloc(&matmul_output, (size_t)M * Width_unroll * sizeof(float));

    dim3 unroll_grid((H_out * W_out + TILE_WIDTH - 1) / TILE_WIDTH, Channel, Batch);
    dim3 unroll_block(TILE_WIDTH);
    matrix_unrolling_kernel<<<unroll_grid,unroll_block>>>(device_input, unrolled_matrix, Batch, Channel, Height, Width, K);
    
    cudaDeviceSynchronize();



    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f, beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Width_unroll, M, Height_unroll, &alpha, unrolled_matrix, Width_unroll, device_mask, Height_unroll, &beta, matmul_output,   Width_unroll);
    cublasDestroy(handle);

    cudaDeviceSynchronize();


    
    int image_size = H_out * W_out;
    dim3 perm_grid((image_size + PERMUTE_BLOCK_SIZE - 1)/PERMUTE_BLOCK_SIZE, Batch);
    matrix_permute_kernel<<<perm_grid,PERMUTE_BLOCK_SIZE>>>(matmul_output, device_output, Map_out, Batch, image_size);

    cudaDeviceSynchronize();

    cudaFree(unrolled_matrix);
    cudaFree(matmul_output);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    cudaMemcpy(host_output, device_output, ((Height-K+1) * (Width-K+1) * Batch * Map_out) * sizeof(float), cudaMemcpyDeviceToHost);

    // TODO: Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
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