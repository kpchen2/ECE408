#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <mma.h>

#define TILE_WIDTH 16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8
#define WARP_SIZE 32

using namespace nvcuda::wmma;

__global__ void matmul_conv_fused(const float *mask, const float *input, float *output,
    int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    int H_out = Height - K + 1;
    int W_out = Width - K + 1;
    int image_size = H_out * W_out;
    int K_dim = Channel * K * K;
    int M = Map_out;

    int b = blockIdx.z;
    int tileN = blockIdx.x;
    int tileM = blockIdx.y;

    int row0 = tileM * TILE_WIDTH;
    int col0 = tileN * TILE_WIDTH;

    int tx = threadIdx.x;

    if (b >= Batch || row0 >= M || col0 >= image_size || tx >= WARP_SIZE) {
        return;
    }

    __shared__ float sA[TILE_WIDTH][WMMA_K];
    __shared__ float sB[WMMA_K][TILE_WIDTH];
    __shared__ float sC[TILE_WIDTH][TILE_WIDTH];

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, precision::tf32, row_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    fill_fragment(c_frag, 0.0f);

    for (int sub = 0; sub < K_dim; sub += WMMA_K) {
        for (int f = 0; f < 4; f++) {
            int idx = tx*4 + f;
            int r = idx / WMMA_K;
            int c = idx % WMMA_K;
            int m = row0 + r;
            int k = sub  + c;
            float val = 0.0f;

            if (m < M && k < K_dim) {
                val = mask[m * K_dim + k];
            }
            sA[r][c] = val;
        }

        for (int f = 0; f < 4; f++) {
            int idx = tx*4 + f;
            int r = idx / TILE_WIDTH;
            int c2 = idx % TILE_WIDTH;
            int k_unr = sub + r;
            int spatIndex = col0 + c2;
            float val = 0.0f;

            if (k_unr < K_dim && spatIndex < image_size) {
                int fmap = k_unr / (K*K);
                int off = k_unr % (K*K);
                int p = off / K;
                int q = off % K;
                int ho = spatIndex / W_out;
                int wo = spatIndex % W_out;
                int hi = ho + p;
                int wi = wo + q;

                if (fmap < Channel && hi < Height && wi < Width) {
                    val = input[b*Channel*Height*Width + fmap*Height*Width + hi*Width + wi];
                }
            }
            sB[r][c2] = val;
        }

        __syncthreads();

        load_matrix_sync(a_frag, &sA[0][0], WMMA_K);

        for (int i = 0; i < a_frag.num_elements; i++)
            a_frag.x[i] = __float_to_tf32(a_frag.x[i]);

        load_matrix_sync(b_frag, &sB[0][0], TILE_WIDTH);

        for (int i = 0; i < b_frag.num_elements; i++)
            b_frag.x[i] = __float_to_tf32(b_frag.x[i]);

        mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    store_matrix_sync(&sC[0][0], c_frag, TILE_WIDTH, mem_row_major);
    __syncthreads();

    const int TOTAL = WMMA_M * WMMA_N;
    for (int idx = tx; idx < TOTAL; idx += WARP_SIZE) {
        int rr = idx / WMMA_N;
        int cc = idx % WMMA_N;
        int m  = row0 + rr;
        int s  = col0 + cc;

        if (m < M && s < image_size) {
            int ho = s / W_out;
            int wo = s % W_out;
            output[b*Map_out*H_out*W_out + m*H_out*W_out + ho*W_out + wo] = sC[rr][cc];
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

    dim3 blockDim(32, 1, 1);
    dim3 gridDim((h_out*w_out + 15) / 16, (Map_out + 15) / 16, Batch);

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