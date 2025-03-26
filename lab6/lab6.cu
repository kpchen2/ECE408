// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len, int s) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host

  __shared__ float T[2*BLOCK_SIZE];

  unsigned int t = threadIdx.x;
  unsigned int start = 2*blockIdx.x*blockDim.x;
  T[2*t] = input[start + 2*t];
  T[2*t + 1] = input[start + 2*t + 1];

  int stride = 1;
  while (stride < 2*len) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if (index < 2*len && (index-stride) >= 0) {
      T[index] += T[index-stride];
    }
    stride = stride*2;
  }

  stride = len/2;
  while (stride > 0) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if ((index+stride) < 2*len) {
      T[index+stride] += T[index];
    }
    stride = stride/2;
  }

  __syncthreads();

  output[start + 2*t] = T[2*t];
  output[start + 2*t + 1] = T[2*t + 1];
}

__global__ void add_together(float *input, float *output, float *aux_sum, int len) {

  __shared__ float T[2*BLOCK_SIZE];

  unsigned int t = threadIdx.x;
  unsigned int start = 2*blockIdx.x*blockDim.x;
  T[2*t] = input[start + 2*t];
  T[2*t + 1] = input[start + 2*t + 1];

  int stride = 1;
  while (stride < 2*len) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if (index < 2*len && (index-stride) >= 0) {
      T[index] += T[index-stride];
    }
    stride = stride*2;
  }

  stride = len/2;
  while (stride > 0) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if ((index+stride) < 2*len) {
      T[index+stride] += T[index];
    }
    stride = stride/2;
  }

  __syncthreads();

  if (blockIdx.x == 0) {
    output[start + 2*t] = T[2*t];
    output[start + 2*t + 1] = T[2*t + 1];
  } else {
    int temp = aux_sum[blockIdx.x - 1];
    output[start + 2*t] = T[2*t] + temp;
    output[start + 2*t + 1] = T[2*t + 1] + temp;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  float* aux;
  float* aux_sum;
  float* aux_temp;
  int aux_length;

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));

  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));

  aux_length = (numElements-1)/(BLOCK_SIZE*2) + 1;
  aux = (float *)malloc(BLOCK_SIZE * sizeof(float));
  wbCheck(cudaMalloc((void **)&aux_temp, BLOCK_SIZE * sizeof(float)));
  wbCheck(cudaMalloc((void **)&aux_sum, BLOCK_SIZE * sizeof(float)));

  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid1(aux_length, 1, 1);
  dim3 dimBlock1(BLOCK_SIZE, 1, 1);

  dim3 dimGrid2(1, 1, 1);
  dim3 dimBlock2(BLOCK_SIZE, 1, 1);

  dim3 dimGrid3(aux_length, 1, 1);
  dim3 dimBlock3(BLOCK_SIZE, 1, 1);

  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid1, dimBlock1>>>(deviceInput, deviceOutput, BLOCK_SIZE, 0);
  cudaDeviceSynchronize();

  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));

  for (int i = 0; i < BLOCK_SIZE; i++) {
    aux[i] = 0;
  }

  for (int i = 0; i < aux_length; i++) {
    if ((i+1) * BLOCK_SIZE*2 - 1 > numElements) {
      aux[i] = hostOutput[numElements-1];
    } else {
      aux[i] = hostOutput[2*BLOCK_SIZE*(i+1) - 1];
    }
  }

  // cudaMemset(aux_temp, 0, aux_length * sizeof(float));
  wbCheck(cudaMemcpy(aux_temp, aux, BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));

  scan<<<dimGrid2, dimBlock2>>>(aux_temp, deviceOutput, BLOCK_SIZE, 1);
  cudaDeviceSynchronize();

  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));

  for (int i = 0; i < BLOCK_SIZE; i++) {
    aux[i] = 0;
  }

  for (int i = 0; i < aux_length; i++) {
    aux[i] = hostOutput[i];
  }

  wbCheck(cudaMemcpy(aux_temp, aux, BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice));

  add_together<<<dimGrid3, dimBlock3>>>(deviceInput, deviceOutput, aux_temp, BLOCK_SIZE);
  cudaDeviceSynchronize();

  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));

  //@@  Free GPU Memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(aux_temp);
  cudaFree(aux_sum);

  wbSolution(args, hostOutput, numElements);

  free(aux);
  free(hostInput);
  free(hostOutput);

  return 0;
}

