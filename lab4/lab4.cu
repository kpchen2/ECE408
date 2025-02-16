#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here

//@@ Define constant memory for device kernel here
__constant__ float Mc[27];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here

  // true coordinates
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  // thread/arr coordinates
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  // constants
  int tileSize = 4;
  int maskWidth = 3;

  __shared__ float arr[64];
  
  if (x < x_size && y < y_size && z < z_size) {
    arr[tx + ty*(tileSize) + tz*(tileSize*tileSize)] = input[x + y*(x_size) + z*(x_size*y_size)];
  } else {
    arr[tx + ty*(tileSize) + tz*(tileSize*tileSize)] = 0;
  }

  __syncthreads();

  float pVal = 0;
  for (int c = 0; c < maskWidth; c++) {
    for (int b = 0; b < maskWidth; b++) {
      for (int a = 0; a < maskWidth; a++) {
        if (x+a-1 < 0 || x+a-1 >= x_size || y+b-1 < 0 || y+b-1 >= y_size || z+c-1 < 0 || z+c-1 >= z_size) {
          pVal += 0;
        } else if (tx+a-1 >= tileSize || tx+a-1 < 0 || ty+b-1 >= tileSize || ty+b-1 < 0 || tz+c-1 >= tileSize || tz+c-1 < 0) {
          pVal += Mc[a + b*(maskWidth) + c*(maskWidth*maskWidth)] * input[(x+a-1) + (y+b-1)*(x_size) + (z+c-1)*(x_size*y_size)];
          // pVal += 1;
        } else {
          pVal += Mc[a + b*(maskWidth) + c*(maskWidth*maskWidth)] * arr[(tx+a-1) + (ty+b-1)*(tileSize) + (tz+c-1)*(tileSize*tileSize)];
          // pVal += 1;
        }
      }
    }
  }

  if (x < x_size && y < y_size && z < z_size) {
    output[x + y*(x_size) + z*(x_size*y_size)] = pVal;
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  float *tempInput;
  float *tempOutput;

  cudaMalloc((void**) &tempInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void**) &tempOutput, (inputLength - 3) * sizeof(float));

  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(tempInput, &hostInput[3], (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Mc, hostKernel, kernelLength * sizeof(float));

  //@@ Initialize grid and block dimensions here
  int tileSize = 4;
  dim3 dimGrid((x_size-1)/tileSize + 1, (y_size-1)/tileSize + 1, (z_size-1)/tileSize + 1);
  dim3 dimBlock(tileSize, tileSize, tileSize);

  // printf("Sizes: %d %d %d Grid: %d %d %d\n", x_size, y_size, z_size, (x_size-1)/tileSize + 1, (y_size-1)/tileSize + 1, (z_size-1)/tileSize + 1);

  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(tempInput, tempOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();

  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(&hostOutput[3], tempOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // for (int i = 0; i < inputLength + 3; i++) {
  //   printf("%f\n", hostOutput[i]);
  // }
  // printf("\n\n");

  //@@ Free device memory
  cudaFree(tempInput);
  cudaFree(tempOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

