// -*- C++ -*-
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#include "ex0_scalarIntegrator_cuda.cuh"

__global__ void sum(double *toSum, unsigned int n)
{
  for(unsigned int i = 1; i < n; ++i)
  {
    toSum[0] += toSum[i];
  }
}

__global__ void sumSection(double firstBound, unsigned long long chunkSize,
                    double* partial, unsigned long long numberOfIntervals, double dx)
{
      int id = threadIdx.x;
      unsigned long threadMax = min(numberOfIntervals, (id+1)*chunkSize);
      for(unsigned long i = id*chunkSize; i < threadMax; ++i) {
          const double evaluationPoint = firstBound + (double(i) + 0.5) * dx;
          partial[id] += std::sin(evaluationPoint);
      }
      partial[id] *= dx;
}

__global__
void
cudaDoScalarIntegration_kernel(double* output, double * bounds, unsigned long
                              numberOfIntervals, double dx) {
  // block-wide reduction storage, size is determined by third kernel
  // launch argument (thing between <<< and >>>)
  extern __shared__ double contributions[];

  unsigned int myID = (blockIdx.x * blockDim.x) + threadIdx.x;

  if(myID < numberOfIntervals) {
    contributions[threadIdx.x] = std::sin(bounds[0] +
                                    (double(intervalIndex) + 0.5) * dx);
  }

  __syncthreads();

  if(threadIdx.x == 0) {
    for(int i = 1; i < blockDim.x; ++i) {
      cotributions[0] += contributions[i];
    }
    atomicAdd(output, contributions[0]);
  }

}

void
cudaDoScalarIntegration(const unsigned int numberOfThreadsPerBlock,
                        double * const output, double * bounds,
                        unsigned long numberOfIntervals, double dx) {

  dim3 blockSize(numberOfThreadsPerBlock);
  dim3 gridSize((numberOfIntervals / numberOfThreadsPerBlock) + 1)

  // allocate somewhere to put our result
  double *dev_output;
  cudaMalloc((void **) &dev_output, 1*sizeof(double));
  cudaMemset(dev_output, 0, sizeof(double))
  // run the kernel
  cudaDoScalarIntegration_kernel<<<gridSize,
    blockSize,
    numberOfThreadsPerBlock*sizeof(double)>>>(dev_output, bounds,
                                              numberOfIntervals, dx);

  // make sure that everything in flight has been completed
  cudaDeviceSynchronize();
  // copy over the output
  cudaMemcpy(output, dev_output, 1*sizeof(double), cudaMemcpyDeviceToHost);


  // clean up
  cudaFree(dev_output);


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
