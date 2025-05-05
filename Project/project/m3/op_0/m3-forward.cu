#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH       16
#define MAX_CONST_MASK  16384

__constant__ float const_mask[MAX_CONST_MASK];

__global__ void matmul_conv_fused(const float *input, float *output,
    int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    int H_out = Height - K + 1;
    int W_out = Width  - K + 1;

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

    int b = col / (H_out * W_out);
    int temp = col % (H_out * W_out);
    int h_out = temp / W_out;
    int w_out = temp % W_out;

    int numTiles = (unroll_rows + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int tileId = 0; tileId < numTiles; tileId++) {
        int tiledA = tileId * TILE_WIDTH + tx;
        int tiledB = tileId * TILE_WIDTH + ty;

        tileA[ty][tx] = (row < Map_out && tiledA < unroll_rows) ? const_mask[row * unroll_rows + tiledA] : 0.0f;

        if (tiledB < unroll_rows && col < unroll_cols) {
            int c = tiledB / (K * K);
            int p = (tiledB / K) % K;
            int q = tiledB % K;
            int h = h_out + p;
            int w = w_out + q;

            tileB[ty][tx] = (b < Batch && c < Channel && h < Height && w < Width) ? input[b*Channel*Height*Width + c*Height*Width + h*Width + w] : 0.0f;
        } else {
            tileB[ty][tx] = 0.0f;
        }

        __syncthreads();

        if (row < Map_out && col < unroll_cols) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val += tileA[ty][i] * tileB[i][tx];
            }
        }

        __syncthreads();
    }

    if (row < Map_out && col < unroll_cols && b < Batch && h_out < H_out && w_out < W_out) {
        output[b*Map_out*H_out*W_out + row*H_out*W_out + h_out*W_out + w_out] = val;
    }
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask,
    float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr,
    const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int h_out = Height - K + 1;
    int w_out = Width  - K + 1;

    cudaMalloc((void**)device_input_ptr, Batch*Channel*Height*Width*sizeof(float));
    cudaMalloc((void**)device_output_ptr, Batch*Map_out*h_out*w_out   *sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, Batch*Channel*Height*Width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_output_ptr, host_output, Batch*Map_out*h_out*w_out   *sizeof(float), cudaMemcpyHostToDevice);

    int mask_elems = Map_out * Channel * K * K;
    size_t bytes = mask_elems * sizeof(float);
    
    cudaMemcpyToSymbol(const_mask, host_mask, bytes);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask,
    const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int h_out = Height - K + 1;
    int w_out = Width  - K + 1;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((Batch*h_out*w_out + TILE_WIDTH - 1)/TILE_WIDTH, (Map_out + TILE_WIDTH - 1)/TILE_WIDTH, 1);

    matmul_conv_fused<<<gridDim,blockDim>>>(device_input, device_output, Batch, Map_out, Channel, Height, Width,K);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask,
    const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int h_out = Height - K + 1;
    int w_out = Width  - K + 1;

    cudaMemcpy(host_output, device_output, Batch*Map_out*h_out*w_out*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        std::cout << "Device " << dev << " name: " << prop.name << "\n";
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "Global memory: " << prop.totalGlobalMem << "\n";
        std::cout << "Constant memory: " << prop.totalConstMem << "\n";
        std::cout << "Shared per block: " << prop.sharedMemPerBlock << "\n";
        std::cout << "Max threads/block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "Warp size: " << prop.warpSize << "\n";
    }
}
