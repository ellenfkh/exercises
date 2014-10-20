// -*- C++ -*-
#include <cstdio>

#include <cuda_runtime.h>

#include "ex1_histogram_cuda.cuh"

__global__
void
cudaDoHistogramPopulation_kernel(unsigned int * d_input, unsigned int * d_output,
                                unsigned int numElements,
                                unsigned int bucketSize) {
  unsigned int myID = (blockIdx.x * blockDim.x) + threadIdx.x;

  if(myID < numElements) {
    const unsigned int value = d_input[myID];
    const unsigned int bucketNumber = value / bucketSize;
    d_output[bucketNumber] += 1;
  }
}

void
cudaDoHistogramPopulation(const unsigned int threadsPerBlock,
                          unsigned int * h_outputHistogram,
                          unsigned int * h_cudaInput,
                          unsigned int * d_input,
                          unsigned int * d_output,
                          unsigned int numElements,
                          unsigned int numBuckets) {

    unsigned int * d_cudaInput;
    unsigned int * d_cudaOutput;
    cudaMalloc(&d_cudaInput, sizeof(unsigned int) * numElements);
    cudaMalloc(&d_cudaOutput, sizeof(unsigned int) * numBuckets);
    cudaMemset(d_cudaOutput, 0, sizeof(unsigned int) * numBuckets);

    cudaMemcpy(d_cudaInput, h_cudaInput,
            sizeof(unsigned int) * numElements, cudaMemcpyHostToDevice);

    dim3 blockSize(threadsPerBlock);
    dim3 gridSize((numElements / threadsPerBlock) + 1);
    const unsigned int bucketSize = numElements/numBuckets;

    cudaDoHistogramPopulation_kernel<<<gridSize, blockSize>>>(d_input, d_output,
                                                              numElements,
                                                              bucketSize);

    cudaMemcpy(h_outputHistogram, d_output, sizeof(unsigned int) * numBuckets,
            cudaMemcpyDeviceToHost);

}
