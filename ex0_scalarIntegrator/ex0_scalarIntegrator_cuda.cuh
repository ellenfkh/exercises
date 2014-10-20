// -*- C++ -*-
#ifndef EX0_SCALARINTEGRATOR_CUDA_CUH
#define EX0_SCALARINTEGRATOR_CUDA_CUH

void
cudaDoScalarIntegration(const unsigned int numberOfThreadsPerBlock,
                        float * const output, double * bounds, 
                        unsigned long numberOfIntervals, double dx);
#endif // EX0_SCALARINTEGRATOR_CUDA_CUH
