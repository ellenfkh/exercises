// -*- C++ -*-
// matrixMultiplication.cc
// a huge comparison of doing naive and tiled matrix multiplication using many
//  different methods and technologies

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
#include "Teuchos_Array.hpp"
#include "Intrepid_ArrayTools.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_RealSpaceTools.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_GlobalMPISession.hpp"

#include <cuda_runtime.h>

using std::string;
using std::vector;
using Intrepid::FieldContainer;

typedef Intrepid::RealSpaceTools<double> rst;

#define BLOCK_SIZE 64;

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


__global__
void
cudaDocontractFieldFieldScalar_kernel(const double * const __restrict__ d_left, const double * const __restrict__ d_right,
double * d_out,
int numCells,
int numLeftFields,
int numRightFields,
int numPoints,
int dim1Tensor,
int dim2Tensor) {

  int myID = (blockIdx.x * blockDim.x) + threadIdx.x;

  if(myID < (numCells * numLeftFields * numRightFields)) {
    int myCell = myID / (numLeftFields * numRightFields);
    int matrixIndex = myID % (numLeftFields * numRightFields);

    int lbf = matrixIndex / numRightFields;
    int rbf = matrixIndex % numRightFields;
    int sub = dim1Tensor * dim2Tensor;
    int left = myCell * numLeftFields * numPoints * sub + lbf * numPoints * sub;
    int right = myCell * numPoints * sub * numRightFields;
    int rsub = sub * numRightFields;

    double temp = 0;
    for (int qp = 0; qp < numPoints; qp++) {
      for (int iTens1 = 0; iTens1 < dim1Tensor; iTens1++) {
        for (int iTens2 = 0; iTens2 < dim2Tensor; iTens2++) {
          temp += d_left[left + qp * sub +
                         iTens1 * dim2Tensor +
                         iTens2] *
                  d_right[right + qp * rsub +
                          iTens1 * dim2Tensor * numRightFields +
                          iTens2 * numRightFields +
                          rbf];
        }
      }
    }
    d_out[myID]= temp;
  }
}

void
cudaDocontractFieldFieldScalar(double * h_out,
    double * h_inLeft,
    double * h_inRight,
    int numCells,
    int numLeftFields,
    int numRightFields,
    int numPoints,
    int dim1Tensor,
    int dim2Tensor,
    timespec * tic,
    timespec * toc) {

  double * d_right;
  double * d_left;
  double * d_out;

  cudaMalloc(&d_right, sizeof(double) * numCells  * numPoints * numRightFields * dim1Tensor * dim2Tensor);

  cudaMalloc(&d_left, sizeof(double) * numCells * numPoints * numLeftFields * dim1Tensor * dim2Tensor);

  cudaMalloc(&d_out, sizeof(double) * numCells * numRightFields * numLeftFields);

  cudaMemset(d_out, 0, sizeof(double) * numCells * numRightFields * numLeftFields);

  cudaMemcpy(d_right, h_inRight,
      sizeof(double) * numCells * numPoints * numRightFields * dim1Tensor * dim2Tensor, cudaMemcpyHostToDevice);

  cudaMemcpy(d_left, h_inLeft,
      sizeof(double) * numCells * numPoints * numLeftFields * dim1Tensor * dim2Tensor, cudaMemcpyHostToDevice);


  dim3 blockSize(1024);
  dim3 gridSize((numCells * numLeftFields * numRightFields * dim1Tensor * dim2Tensor / 1024) + 1);

  clock_gettime(CLOCK_MONOTONIC, tic);
  cudaDocontractFieldFieldScalar_kernel<<<gridSize, blockSize>>>(d_left,
      d_right, d_out, numCells, numLeftFields, numRightFields, numPoints, dim1Tensor, dim2Tensor);

  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, toc);
  cudaMemcpy(h_out, d_out, sizeof(double) * numCells * numLeftFields * numRightFields, cudaMemcpyDeviceToHost);

  cudaFree(d_right);
  cudaFree(d_left);
  cudaFree(d_out);

}


// Serial contractFieldFieldScalar.  Contracts FieldContainers of doubles.
void contractFieldFieldScalarSerial(FieldContainer<double> &  outputFields,
    const FieldContainer<double> &              leftFields,
    const FieldContainer<double> &              rightFields,
    double *                                    time = 0) {

  // TODO(ellen): Might later want to template this so that both the container
  //              and the scalars inside the container are template arguments,
  //              so we can hand it kokkos views or custom structs.
  int numCells        = leftFields.dimension(0);
  int numLeftFields   = leftFields.dimension(1);
  int numRightFields  = rightFields.dimension(1);
  int numPoints       = leftFields.dimension(2);
  int dim1Tensor      = leftFields.dimension(3);
  int dim2Tensor      = leftFields.dimension(4);

  for (int cl = 0; cl < numCells; cl++) {
    for (int lbf = 0; lbf < numLeftFields; lbf++) {
      for (int rbf = 0; rbf < numRightFields; rbf++) {
        double tmpVal = 0;
        for (int qp = 0; qp < numPoints; qp++) {
          for (int iTens1 = 0; iTens1 < dim1Tensor; iTens1++) {
            for (int iTens2 = 0; iTens2 < dim2Tensor; iTens2++) {
              tmpVal += leftFields(cl, lbf, qp, iTens1, iTens2)*rightFields(cl, rbf, qp, iTens1, iTens2);
            } // D2-loop
          } // D1-loop
        } // P-loop
        outputFields(cl, lbf, rbf) = tmpVal;
      } // R-loop
    } // L-loop
  } // C-loop
}



int main(int argc, char* argv[]) {
  int c=10000, p=10, l=10, r=10, t1=10, t2=10;

  FieldContainer<double> in_c_l_p_t1_t2(c, l, p, t1, t2);
  FieldContainer<double> in_c_r_p_t1_t2(c, r, p, t1, t2);
  FieldContainer<double> out1_c_l_r(c, l, r);
  FieldContainer<double> out2_c_l_r(c, l, r);
  double zero = Intrepid::INTREPID_TOL*100000.0;
  zero *= 10;

  // fill with random numbers
  for (int i=0; i<in_c_l_p_t1_t2.size(); i++) {
    in_c_l_p_t1_t2[i] = Teuchos::ScalarTraits<double>::random();
  }
  for (int i=0; i<in_c_r_p_t1_t2.size(); i++) {
    in_c_r_p_t1_t2[i] = Teuchos::ScalarTraits<double>::random();
  }
  std::cout << "Created vectors" << std::endl;

  // ===============================================================
  // ********************** < Kokkos setup> ************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  // Doing all of this here might throw off the timing -- we're not counting the
  // cost of the copy into Kokkos or the deep copy from Kokkos host to Kokkos
  // device.



  //Cuda arrays

  double * cudaRight = new double[c * r * p * t1 * t2];
  double * cudaLeft = new double[c * l * p * t1 * t2];
  double * cudaOut = new double[c * l * r];


  for (int cl = 0; cl < c; ++cl) {
    for (int qp = 0; qp < p; ++qp) {
      for (int iTens1 = 0; iTens1 < t1; ++iTens1) {
        for (int iTens2 = 0; iTens2 < t2; ++iTens2) {
          for(int rbf = 0; rbf < r; ++rbf) {
            cudaRight[cl * p * t1 * t2 * r +
                      qp * t1 * t2 * r +
                      iTens1 * t2 * r +
                      iTens2 * r +
                      rbf] = in_c_r_p_t1_t2(cl, rbf,qp, iTens1, iTens2);
          }
          for(int lbf = 0; lbf < l; ++lbf) {
            cudaLeft[cl * l * p * t1 * t2 +
                     lbf * p * t1 * t2 +
                     qp * t1 * t2 +
                     iTens1 * t2 +
                     iTens2] = in_c_l_p_t1_t2(cl,lbf,qp, iTens1, iTens2);
          }
        }
      }
    }
  }



  std::cout << "trying serial" << std::endl;

  //Warmup
  contractFieldFieldScalarSerial(out2_c_l_r, in_c_l_p_t1_t2, in_c_r_p_t1_t2);

  timespec tic;
  clock_gettime(CLOCK_MONOTONIC, &tic);

  //repeat the calculation 5 times so we can average out some randomness
  for(int i = 0; i < 5; ++i){
    contractFieldFieldScalarSerial(out2_c_l_r, in_c_l_p_t1_t2, in_c_r_p_t1_t2);
  }

  timespec toc;
  clock_gettime(CLOCK_MONOTONIC, &toc);
  const double elapsedTime_serial = getElapsedTime(tic, toc);

  std::cout << "serial took " << elapsedTime_serial << " second" << std::endl;

  std::cout << "trying cuda" << std::endl;
  //Now try the cuda version, start with warmup
  cudaDocontractFieldFieldScalar(cudaOut,cudaLeft,cudaRight, c, l, r, p, t1, t2, &tic, &toc);
  double elapsedTime_cuda = 0;
  for(int i = 0; i < 5; ++i){
    cudaDocontractFieldFieldScalar(cudaOut,cudaLeft,cudaRight, c, l, r, p, t1, t2, &tic, &toc);
    elapsedTime_cuda += getElapsedTime(tic,toc);
  }

  for (int cl = 0; cl < c; ++cl) {
    for(int lbf = 0; lbf < l; ++lbf) {
      for(int rbf = 0; rbf < r; ++rbf) {
        out1_c_l_r(cl,lbf,rbf) = cudaOut[cl * l * r + lbf * r + rbf];
      }
    }
  }

  rst::subtract(&out1_c_l_r[0], &out2_c_l_r[0], out2_c_l_r.size());
  if (rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) > zero) {
    std::cout << "\n\nINCORRECT contractFieldFieldTensor (1): check cuda; "
    << " diff-1norm = " << rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) << "\n\n";
  }

  std::cout << "cuda speedup of " << elapsedTime_serial/elapsedTime_cuda << std::endl;

  return 0;
}
