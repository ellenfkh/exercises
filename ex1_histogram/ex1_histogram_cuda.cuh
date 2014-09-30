// -*- C++ -*-
#ifndef EX1_HISTOGRAM_CUDA_CUH
#define EX1_HISTOGRAM_CUDA_CUH

void
cudaDoHistogramPopulation(const unsigned int maxNumberOfBlocks,
                          unsigned int * outputHistogram);

#endif // EX1_HISTOGRAM_CUDA_CUH
