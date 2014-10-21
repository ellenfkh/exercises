// -*- C++ -*-
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#include "ex0_scalarIntegrator_cuda.cuh"

__global__
void
cudaDoScalarIntegration_kernel(double bounds, unsigned long
                              numberOfIntervals, double dx, double *partial) {
  // block-wide reduction storage, size is determined by third kernel
  // launch argument (thing between <<< and >>>)
  extern __shared__ double contributions[];

  unsigned int myID = (blockIdx.x * blockDim.x) + threadIdx.x;

  if(myID < numberOfIntervals) {
    const double evaluationPoint =
      bounds + (double(myID) + 0.5) * dx;
    contributions[threadIdx.x] = std::sin(evaluationPoint);
  }

  __syncthreads();

  if(threadIdx.x == 0) {
    for(int i = 1; i < blockDim.x; ++i) {
      contributions[0] += contributions[i];
    }
    partial[blockIdx.x] = contributions[0];
  }

}

void
cudaDoScalarIntegration(const unsigned int numberOfThreadsPerBlock,
                        double * const output, double bounds,
                        unsigned long numberOfIntervals, double dx) {

  dim3 blockSize(numberOfThreadsPerBlock);
  dim3 gridSize((numberOfIntervals / numberOfThreadsPerBlock) + 1);

  // run the kernel

  double *partialResults;
  cudaMalloc((void **) &partialResults, gridSize.x * sizeof(double));
  cudaMemset(partialResults, 0, sizeof(double));

  cudaDoScalarIntegration_kernel<<<gridSize,
    blockSize,
    numberOfThreadsPerBlock*sizeof(double)>>>(bounds, numberOfIntervals, dx,
                                              partialResults);

  // make sure that everything in flight has been completed
  cudaDeviceSynchronize();

  double *h_partialResults[gridSize.x];
  cudaMemcpy(h_partialResults, partialResults, sizeof(double) * gridSize.x,
            cudaMemcpyDeviceToHost);

  double finalSum = 0;
  for(int i = 0; i < gridSize.x; ++i) {
    finalSum += h_partialResults[i];
  }

  *output = finalSum * dx;



  // TODO: you have to do stuff in here, the junk below is just to show syntax
  /*
  double *d_partial;
  double cudaIntegral = 0;
  unsigned long chunkPerThread = numberOfIntervals/numberOfThreads + 1;
  cudaMalloc( (void**)&d_partial, sizeof(double)*numberOfThreadsPerBlock);
  sumSection<<<1,numberOfThreadsPerBlock>>>(bounds[0], chunkPerThread, d_partial,
                                            numberOfIntervals, dx);
  cudaDeviceSynchronize();

  sum<<<1,1>>>(d_partial, numberOfThreadsPerBlock);

  cudaMemcpy(&cudaIntegral, d_partial, sizeof(double), cudaMemcpyDeviceToHost);



  // this is how to use constant memory:
  // make some stuff that we'll copy into constant memory
  double * iLikePuppies = new double[MAX_NUMBER_OF_PUPPIES];
  iLikePuppies[0] = 0.;
  iLikePuppies[1] = 1.;
  iLikePuppies[2] = 2.;
  // copy some junk into constant memory
  cudaMemcpyToSymbol(constantPuppies, iLikePuppies,
                     sizeof(double) * MAX_NUMBER_OF_PUPPIES);
  delete[] iLikePuppies;


  // using global memory
  unsigned int amountOfJunk = 10;
  // make some array from which we'll copy to the device
  double * junk = new double[amountOfJunk];
  // fill it with junk
  for (int i = 0; i < amountOfJunk; ++i) {
    junk[i] = -i;
  }
  // this is going to be a pointer to memory *on the device*
  double * dev_junk;
  // allocate room on the device
  cudaMalloc((void **) &dev_junk, amountOfJunk*sizeof(double));
  // copy junk from host to device
  cudaMemcpy(dev_junk, junk, amountOfJunk*sizeof(double),
             cudaMemcpyHostToDevice);
  delete[] junk;

  // TODO: calculate the number of blocks
  const unsigned int numberOfBlocks = 1;

  // allocate somewhere to put our result
  double *dev_output;
  cudaMalloc((void **) &dev_output, 1*sizeof(double));

  // run the kernel
  cudaDoScalarIntegration_kernel<<<numberOfBlocks,
    numberOfThreadsPerBlock,
    numberOfThreadsPerBlock*sizeof(double)>>>(dev_output);

  // copy over the output
  cudaMemcpy(output, dev_output, 1*sizeof(double), cudaMemcpyDeviceToHost);
  // make sure that everything in flight has been completed
  cudaDeviceSynchronize();

  // clean up
  cudaFree(dev_junk);
  cudaFree(dev_output);
  */
}
