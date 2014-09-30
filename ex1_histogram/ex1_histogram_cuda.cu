// -*- C++ -*-
#include <cstdio>

#include <cuda_runtime.h>

#include "ex1_histogram_cuda.cuh"

__global__
void
cudaDoHistogramPopulation_kernel(unsigned int * outputHistogram) {
  // TODO
}

void
cudaDoHistogramPopulation(const unsigned int numberOfThreadsPerBlock,
                          unsigned int * outputHistogram) {
  // TODO
}
