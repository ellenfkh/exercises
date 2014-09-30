// -*- C++ -*-
#ifndef EX3_NAIVEMATRIXMULTIPLICATION_CUDA_CUH
#define EX3_NAIVEMATRIXMULTIPLICATION_CUDA_CUH

void
cudaDoNaiveMatrixMultiplication(const unsigned int numberOfThreadsPerBlock,
                                const unsigned int matrixSize,
                                double * output);

#endif // EX3_NAIVEMATRIXMULTIPLICATION_CUDA_CUH
