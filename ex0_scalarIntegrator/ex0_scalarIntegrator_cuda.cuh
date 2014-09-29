// -*- C++ -*-
#ifndef EX0_SCALARINTEGRATOR_CUDA_CUH
#define EX0_SCALARINTEGRATOR_CUDA_CUH

// lots of consts for defensive programming
void
cudaDoScalarIntegration(const unsigned int numberOfThreadsPerBlock,
                        double * const output);

#endif // EX0_SCALARINTEGRATOR_CUDA_CUH
