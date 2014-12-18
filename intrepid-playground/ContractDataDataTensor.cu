/*
 *
 * Contract two arrays of 3d tensors (c, p, t1, t2) to get array of scalars (c)
 *
 * cached: (c, p, t1, t2) by (c, p, t1, t2)
 * to parallelize, probably need shared memory
 */


#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>

// yucky, but for asking the system how many cores we have
#include <unistd.h>
#include <assert.h>

// header file for openmp
#include <omp.h>

// header files for kokkos
#include <Kokkos_Core.hpp>
#include <cuda_runtime.h>

//Pre-C++11 timing (thanks jeff)
double getElapsedTime(const timespec start, const timespec end) {
  timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return double(temp.tv_sec) + double(temp.tv_nsec) / 1e9;
}

#if 0
for (int cl = 0; cl < numCells; cl++) {
  Scalar tmpVal(0);
  for (int qp = 0; qp < numPoints; qp++) {
    for (int iTens1 = 0; iTens1 < dim1Tensor; iTens1++) {
      for (int iTens2 = 0; iTens2 < dim2Tensor; iTens2++) {
        tmpVal += inputDataLeft(cl, qp, iTens1, iTens2)*inputDataRight(cl, qp, iTens1, iTens2);
      } // D2-loop
    } // D1-loop
  } // P-loop
  outputData(cl) += tmpVal;
} // C-loop

#endif


void serial(double* leftInput, double* rightInput, double* output,
            int c, int p, int t1, int t2) {

  for (int cl=0; cl < c; cl++) {
    double tmp = 0;
    for (int qp=0; qp < p; qp++) {
      for (int iTens1=0; iTens1 < t1; iTens1++) {
        for (int iTens2=0; iTens2 < t2; iTens2++) {
          tmp += leftInput[cl * p * t1 * t2 + qp * t1 * t2 + iTens1 * t2 + iTens2] *
                 rightInput[cl * p * t1 * t2 + qp * t1 * t2 + iTens1 * t2 + iTens2];
        }
      }
    }
    output[cl] = tmp;
  }

}




int main(int argc, char* argv[]) {
  const int c=100000, p=10, t1=10, t2=10;
  const int repeats = 10;

  timespec tic;
  timespec toc;

  double* leftInput = new double[c * p * t1 * t2];
  double* rightInput = new double[c * p * t1 * t2];
  double* serialOutput = new double[c];

  for (int cl=0; cl < c; cl++) {
    for (int qp=0; qp < p; qp++) {
      for (int iTens1=0; iTens1 < t1; iTens1++) {
        for (int iTens2=0; iTens2 < t2; iTens2++) {
          double tmp1 = (double)std::rand();
          double tmp2 = (double)std::rand();
          leftInput[cl * p * t1 * t2 + qp * t1 * t2 + iTens1 * t2 + iTens2] = tmp1;
          rightInput[cl * p * t1 * t2 + qp * t1 * t2 + iTens1 * t2 + iTens2] = tmp2;
        }
      }
    }
  }

  for (int cl=0; cl < c; cl++) {
    serialOutput[cl] = 0;
  }


  clock_gettime(CLOCK_MONOTONIC, &tic);
  for (int i = 0; i < repeats; i++) {
    serial(leftInput, rightInput, serialOutput, c, p, t1, t2);
  }
  clock_gettime(CLOCK_MONOTONIC, &toc);
  const double elapsedTime_serial = getElapsedTime(tic, toc);

  std::cout << "cache friendly serial time: " << elapsedTime_serial << std::endl;

}
