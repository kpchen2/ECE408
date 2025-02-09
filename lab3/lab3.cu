#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  int tileWidth = 16;

  __shared__ float subTileM[16][16];
  __shared__ float subTileN[16][16];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * (tileWidth) + ty;
  int col = bx * (tileWidth) + tx;
  float pVal = 0;

  for (int q = 0; q < ((numAColumns-1)/tileWidth + 1); q++) {
    if (row < numARows && q*tileWidth + tx < numAColumns) {
      subTileM[ty][tx] = A[row*(numAColumns) + q*(tileWidth) + tx];
    } else {
      subTileM[ty][tx] = 0;
    }
    if (q*tileWidth + ty < numBRows && col < numBColumns) {
      subTileN[ty][tx] = B[(q*(tileWidth) + ty)*(numBColumns) + col];
    } else {
      subTileN[ty][tx] = 0;
    }
    __syncthreads();

    for (int k = 0; k < (tileWidth); k++) {
      pVal += subTileM[ty][k] * subTileN[k][tx];
    }
    __syncthreads();
  }

  if (row < numCRows && col < numCColumns) {
    C[row*(numCColumns) + col] = pVal;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix

  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;

  //@@ Allocate the hostC matrix
  hostC = (float*)malloc((numCRows * numCColumns) * sizeof(float));

  //@@ Allocate GPU memory here
  float *tempA;
  float *tempB;
  float *tempC;

  cudaMalloc((void**) &tempA, (numARows * numAColumns) * sizeof(float));
  cudaMalloc((void**) &tempB, (numBRows * numBColumns) * sizeof(float));
  cudaMalloc((void**) &tempC, (numCRows * numCColumns) * sizeof(float));

  //@@ Copy memory to the GPU here
  cudaMemcpy(tempA, hostA, (numARows * numAColumns) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(tempB, hostB, (numBRows * numBColumns) * sizeof(float), cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  float tileSize = 16.0;
  dim3 dimGrid(ceil((1.0*numCColumns)/tileSize), ceil((1.0*numCRows)/tileSize), 1);
  dim3 dimBlock(tileSize, tileSize, 1);

  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<dimGrid, dimBlock>>>(tempA, tempB, tempC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, tempC, (numCRows * numCColumns) * sizeof(float), cudaMemcpyDeviceToHost);
  // printf("first: %f, last: %f", hostC[0], hostC[numCRows*numCColumns - 1]);

  //@@ Free the GPU memory here
  cudaFree(tempA);
  cudaFree(tempB);
  cudaFree(tempC);

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);

  //@@ Free the hostC matrix
  free(hostC);

  return 0;
}
