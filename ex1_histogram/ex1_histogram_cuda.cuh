// -*- C++ -*-
#ifndef EX1_HISTOGRAM_CUDA_CUH
#define EX1_HISTOGRAM_CUDA_CUH

void
cudaDoHistogramPopulation(const unsigned int threadsPerBlock,
                          unsigned int * h_outputHistogram,
                          unsigned int * d_input,
                          unsigned int * d_output,
                          unsigned int numElements,
                          unsigned int numBuckets);

#endif // EX1_HISTOGRAM_CUDA_CUH
