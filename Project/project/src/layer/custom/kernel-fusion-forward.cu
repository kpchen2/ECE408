#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

__global__ void matmul_conv_fused(const float *mask, const float *input, float *output,
    int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    /*
    TODO: Modify this function to implement the fused unroll-matmul-permute kernel.
    
    Function parameter definitions:
    mask - convolution kernel
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    int H_out = Height - K + 1;
    int W_out = Width - K + 1;

    int unroll_rows = Channel * K * K;
    int unroll_cols = Batch * H_out * W_out;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    float val = 0.0f;

    if (col < unroll_cols) {
        int b = col / (H_out * W_out);
        int temp = col % (H_out * W_out);
        int h_out = temp / W_out;
        int w_out = temp % W_out;

        for (int tileId = 0; tileId < (unroll_rows - 1) / TILE_WIDTH + 1; tileId++) {

            if (row < Map_out && tileId * TILE_WIDTH + tx < unroll_rows) {
                tileA[ty][tx] = mask[row * unroll_rows + tileId * TILE_WIDTH + tx];
            } else {
                tileA[ty][tx] = 0.0f;
            }

            if (tileId * TILE_WIDTH + ty < unroll_rows && col < unroll_cols) {
                int input_row = tileId * TILE_WIDTH + ty;

                int c = input_row / (K * K);
                int p = (input_row / K) % K;
                int q = input_row % K;

                int h = h_out + p;
                int w = w_out + q;

                float input_val = 0.0f;
                if (h < Height && w < Width) {
                    input_val = input[b*Channel*Height*Width + c*Height*Width + h*Width + w];
                }
                tileB[ty][tx] = input_val;
            } else {
                tileB[ty][tx] = 0.0f;
            }

            __syncthreads();

            if (row < Map_out && col < unroll_cols) {
                for (int i = 0; i < TILE_WIDTH; ++i) {
                    val += tileA[ty][i] * tileB[i][tx];
                }
            }

            __syncthreads();
        }

        if (row < Map_out && col < unroll_cols) {
            int b = col / (H_out * W_out);
            int temp = col % (H_out * W_out);
            int h_out = temp / W_out;
            int w_out = temp % W_out;

            output[b*Map_out*H_out*W_out + row*H_out*W_out + h_out*W_out + w_out] = val;
        }
    }
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    const int h_out = Height - K + 1;
    const int w_out = Width - K + 1;
    
    cudaMalloc((void**)device_input_ptr, Batch* Channel * Height * Width * sizeof(float));
    cudaMalloc((void**)device_output_ptr, Batch * Map_out * h_out * w_out * sizeof(float));
    cudaMalloc((void**)device_mask_ptr, Map_out * Channel * K * K * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_output_ptr, host_output, Batch * Map_out * h_out * w_out * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Set the kernel dimensions and call the fused kernel

    const int h_out = Height - K + 1;
    const int w_out = Width - K + 1;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim((Batch*h_out*w_out + TILE_WIDTH - 1) / TILE_WIDTH, (Map_out + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    matmul_conv_fused<<<gridDim, blockDim>>>(device_mask, device_input, device_output, Batch, Map_out, Channel, Height, Width, K);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host

    // TODO: Free device memory

    const int h_out = Height - K + 1;
    const int w_out = Width - K + 1;
    
    cudaMemcpy(host_output, device_output, Batch * Map_out * h_out * w_out * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
        std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, " << deviceProp.maxThreadsDim[1] << " y, " << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, " << deviceProp.maxGridSize[1] << " y, " << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}