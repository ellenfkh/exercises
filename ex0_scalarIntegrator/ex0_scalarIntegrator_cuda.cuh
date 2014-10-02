// -*- C++ -*-
#ifndef EX0_SCALARINTEGRATOR_CUDA_CUH
#define EX0_SCALARINTEGRATOR_CUDA_CUH

__global__ void sum(double *toSum, unsigned int n)
{
  for(unsigned int i = 1; i < n; ++i)
  {
    toSum[0] += toSum[i];
  }
}

__global__ void sumSection(double firstBound, unsigned long chunkSize,
                    double* partial, unsigned long numberOfIntervals, double dx)
{
      int id = threadId.x;
      unsigned long threadMax = std::min(numberOfIntervals, (id+1)*chunkPerThread);
      for(unsigned long i = id*chunkPerThread; i < threadMax; ++i) {
          const double evaluationPoint = bounds[0] + (double(i) + 0.5) * dx;
          partialResults[id] += std::sin(evaluationPoint);
      }
      partialResults[id] *= dx;
}
#endif // EX0_SCALARINTEGRATOR_CUDA_CUH
