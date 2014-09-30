// -*- C++ -*-
#include <cstdio>

#include <cuda_runtime.h>

#include "ex3_naiveMatrixMultiplication_cuda.cuh"

__global__
void
cudaDoNaiveMatrixMultiplication_kernel(const unsigned int matrixSize,
                                       double * result) {
  // TODO: implement the kernel
}

void
cudaDoNaiveMatrixMultiplication(const unsigned int numberOfThreadsPerBlock,
                                const unsigned int matrixSize,
                                double * output) {

  // TODO: prepare and launch the kernel

}
