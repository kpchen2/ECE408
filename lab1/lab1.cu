// LAB 1
#include <wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) out[i] = in1[i] + in2[i];
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;

  args = wbArg_read(argc, argv);
  //@@ Importing data and creating memory on host
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  wbLog(TRACE, "The input length is ", inputLength);

  // printf("1: %f, 2: %f, 3: %d\n", *hostInput1, *hostInput2, inputLength);

  //@@ Allocate GPU memory here
  float* A;
  float* B;
  float* C;

  cudaMalloc((void**) &A, inputLength * sizeof(float));
  cudaMalloc((void**) &B, inputLength * sizeof(float));
  cudaMalloc((void**) &C, inputLength * sizeof(float));

  //@@ Copy memory to the GPU here
  cudaMemcpy(A, hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B, hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  vecAdd<<<ceil(inputLength/256.0), 256>>>(A, B, C, inputLength);

  //@@ Launch the GPU Kernel here to perform CUDA computation
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, C, inputLength * sizeof(float), cudaMemcpyDeviceToHost);
  // printf("A: %f, B: %f, C: %f\n", A, B, C);

  //@@ Free the GPU memory here
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
