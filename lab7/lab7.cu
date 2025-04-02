// Histogram Equalization

#include <wb.h>
#include <stdio.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 32

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// #define min(a,b) ((a) > (b) ? (b) : (a));
// #define max(a,b) ((a) > (b) ? (a) : (b));
// #define clamp(x, start, end) (min(max((x), (start)), (end)));

__device__ float clamp(float x, float start, float end) {
  if (x > start) {
    if (x < end) {
      return x;
    } else {
      return end;
    }
  } else {
    if (start < end) {
      return start;
    } else {
      return end;
    }
  }
}

//@@ insert code here
__global__ void toUnsigned(float* input, unsigned char* output, int width, int height) {
  int idx = blockDim.x*blockIdx.x + threadIdx.x;

  if (idx < 3*width*height) {
    output[idx] = (unsigned char)((255 * input[idx]));
  }
}

__global__ void toGrayScale(unsigned char* input, unsigned char* output, int width, int height) {
  int idx = blockDim.x*blockIdx.x + threadIdx.x;

  if (idx < width*height) {
    int r = input[3*idx];
    int g = input[3*idx + 1];
    int b = input[3*idx + 2];
    output[idx] = (unsigned char)(0.21*r + 0.71*g + 0.07*b);
  }
}

__global__ void toHistogram(unsigned char* input, int* histogram, int width, int height) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = blockDim.x*gridDim.x;

  while (i < width*height) {
    atomicAdd( &(histogram[input[i]]), 1);
    i += stride;
  }
}

__global__ void toEqualize(unsigned char* input, float* output, float* cdf, int width, int height) {
  int idx = blockDim.x*blockIdx.x + threadIdx.x;

  if (idx < 3*width*height) {
    unsigned char val = input[idx];
    float normalized = (cdf[val] - cdf[0]) / (1.0f - cdf[0]);
    float scaled = 255.0f * normalized;
    float clamped = clamp(scaled, 0.0f, 255.0f);
    // printf("%f, %f, %f\n", normalized, scaled, clamped);
    output[idx] = clamped / 255.0f;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  // float *hostInputImageData;
  // float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  //@@ insert code here
  int numInputElements = imageWidth * imageHeight * imageChannels;
  unsigned char *unsignedOutput;
  unsignedOutput = (unsigned char*)malloc(numInputElements * sizeof(unsigned char));

  float *tempInput;
  unsigned char *tempOutput;
  cudaMalloc((void**) &tempInput, numInputElements * sizeof(float));
  cudaMalloc((void**) &tempOutput, numInputElements * sizeof(unsigned char));
  cudaMemcpy(tempInput, wbImage_getData(inputImage), numInputElements * sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimGrid1((imageWidth*imageHeight*imageChannels-1)/(BLOCK_SIZE*BLOCK_SIZE) + 1, 1, 1);
  dim3 dimBlock1(BLOCK_SIZE*BLOCK_SIZE, 1, 1);
  toUnsigned<<<dimGrid1, dimBlock1>>>(tempInput, tempOutput, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  cudaMemcpy(unsignedOutput, tempOutput, numInputElements * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  // ********************************************
  // DONE: cast image from float to unsigned char
  // ********************************************

  unsigned char *grayScaleOutput;
  grayScaleOutput = (unsigned char*)malloc((numInputElements/3) * sizeof(unsigned char));
  unsigned char *kernel_grayScaleOutput;
  cudaMalloc((void**) &kernel_grayScaleOutput, (numInputElements/3) * sizeof(unsigned char));

  dim3 dimGrid2((imageWidth*imageHeight-1)/(BLOCK_SIZE*BLOCK_SIZE) + 1, 1, 1);
  dim3 dimBlock2(BLOCK_SIZE*BLOCK_SIZE, 1, 1);
  toGrayScale<<<dimGrid2, dimBlock2>>>(tempOutput, kernel_grayScaleOutput, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  cudaMemcpy(grayScaleOutput, kernel_grayScaleOutput, (numInputElements/3) * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  // for (int i = 0; i < numInputElements; i++) {
  //   if (i < 300) {
  //     printf("%u\n", grayScaleOutput[i]);
  //   }
  // }

  // *****************************************
  // DONE: convert image from RGB to grayscale
  // *****************************************

  int *histogram;
  histogram = (int*)malloc(256 * sizeof(int));
  int *kernel_histogram;
  cudaMalloc((void**) &kernel_histogram, 256 * sizeof(int));

  dim3 dimGrid3((imageWidth*imageHeight-1)/(BLOCK_SIZE*BLOCK_SIZE) + 1, 1, 1);
  dim3 dimBlock3(BLOCK_SIZE*BLOCK_SIZE, 1, 1);
  toHistogram<<<dimGrid3, dimBlock3>>>(kernel_grayScaleOutput, kernel_histogram, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  cudaMemcpy(histogram, kernel_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);

  // ****************************************
  // DONE: compute the histogram of grayImage
  // ****************************************

  float *cdf;
  cdf = (float*)malloc(256 * sizeof(float));

  cdf[0] = histogram[0] / (float)(imageWidth*imageHeight);
  for (int i = 1; i < 256; i++) {
    cdf[i] = cdf[i-1] + histogram[i]/(float)(imageWidth*imageHeight);
  }

  // **********************************
  // DONE: compute the CDF of histogram
  // **********************************

  // cudaMemcpy(tempOutput, unsignedOutput, numInputElements * sizeof(unsigned char), cudaMemcpyHostToDevice);

  float *kernel_equalizedOutput;
  cudaMalloc((void**) &kernel_equalizedOutput, numInputElements * sizeof(float));

  float *kernel_cdf;
  cudaMalloc((void**) &kernel_cdf, 256 * sizeof(float));
  cudaMemcpy(kernel_cdf, cdf, 256 * sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimGrid4((imageWidth*imageHeight*imageChannels-1)/(BLOCK_SIZE*BLOCK_SIZE) + 1, 1, 1);
  dim3 dimBlock4(BLOCK_SIZE*BLOCK_SIZE, 1, 1);

  // for (int i = 0; i < 256; i++) {
  //   printf("%f\n", cdf[i]);
  // }

  toEqualize<<<dimGrid4, dimBlock4>>>(tempOutput, kernel_equalizedOutput, kernel_cdf, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  cudaMemcpy(wbImage_getData(outputImage), kernel_equalizedOutput, numInputElements * sizeof(float), cudaMemcpyDeviceToHost);

  

  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(tempInput);
  cudaFree(tempOutput);
  cudaFree(kernel_grayScaleOutput);
  cudaFree(kernel_histogram);
  cudaFree(kernel_equalizedOutput);
  cudaFree(kernel_cdf);

  free(unsignedOutput);
  free(grayScaleOutput);
  free(histogram);
  free(cdf);

  return 0;
}

