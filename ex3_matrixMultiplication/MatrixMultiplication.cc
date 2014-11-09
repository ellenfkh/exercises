// -*- C++ -*-
// matrixMultiplication.cc
// a huge comparison of doing naive and tiled matrix multiplication using many
//  different methods and technologies

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>
#include <string>
#include <chrono>
#include <algorithm>

// yucky, but for asking the system how many cores we have
#include <unistd.h>

// header file for openmp
#include <omp.h>

// header files for tbb
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>

// header files for cuda implementation
#include "MatrixMultiplication_cuda.cuh"

// header files for eigen
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#include <Eigen/Core>
#pragma GCC diagnostic pop

// header files for kokkos
#include <Kokkos_Core.hpp>

// Include for fieldContainer *fingers crossed*
#include "Teuchos_Array.hpp"
#include "Intrepid_ArrayTools.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_RealSpaceTools.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_GlobalMPISession.hpp"

typedef Kokkos::View<double *> matrixView_type;
typedef matrixView_type::HostMirror host_matrix;

using std::string;
using std::vector;
using std::array;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

struct KokkosFunctor {

  const unsigned int _matrixSize;
  matrixView_type _leftMatrix;
  matrixView_type _rightMatrix;
  matrixView_type  _resultMatrix;

  KokkosFunctor(const unsigned int matrixSize, matrixView_type leftMatrix,
              matrixView_type rightMatrix, matrixView_type  resultMatrix):
              _matrixSize(matrixSize), _leftMatrix(leftMatrix), _rightMatrix(rightMatrix),
              _resultMatrix(resultMatrix) {

              }


  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned int elementIndex) const {

    unsigned int row = elementIndex / _matrixSize;
    unsigned int col = elementIndex % _matrixSize;

    for(unsigned int dummy = 0; dummy < _matrixSize; ++dummy) {
      _resultMatrix(elementIndex) += _leftMatrix(dummy + row * _matrixSize) *
      _rightMatrix(col + dummy * _matrixSize);
    }
  }

private:
  KokkosFunctor();

};

int main(int argc, char* argv[]) {

  // a couple of inputs.  change the numberOfIntervals to control the amount
  //  of work done
  
  // ===============================================================
  // ********************** < do naive openmp> *********************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  /*
  // for each number of threads
  for (const unsigned int numberOfThreads :
         numberOfThreadsArray) {

    // set the number of threads for openmp
    omp_set_num_threads(numberOfThreads);

    // start timing
    tic = high_resolution_clock::now();

    for (unsigned int repeatIndex = 0;
         repeatIndex < numberOfRepeats; ++repeatIndex) {
      resultMatrix.fill(0);

      #pragma omp parallel for
      for(unsigned int i = 0; i < matrixSize*matrixSize; ++i) {
        unsigned int row = i / matrixSize;
        unsigned int col = i % matrixSize;

        for(unsigned int dummy = 0; dummy < matrixSize; ++dummy) {
          resultMatrix(i) += leftMatrix(row, dummy) *
          rightMatrixRow(dummy, col);
        }
      }
    }

    // stop timing
    toc = high_resolution_clock::now();
    const double ompElapsedTime =
      duration_cast<duration<double> >(toc - tic).count();

    // check the answer
    double ompCheckSum = 0;
    for (unsigned int row = 0; row < matrixSize; ++row) {
      for (unsigned int col = 0; col < matrixSize; ++col) {
        ompCheckSum += resultMatrix(row, col);
      }
    }
    sprintf(methodName, "naive omp, %3u threads", numberOfThreads);
    if (std::abs(cacheUnfriendlyCheckSum - ompCheckSum) / cacheUnfriendlyCheckSum < 1e-3) {
      printf("%-38s : time %6.2f speedup w.r.t. unfriendly %6.2f, w.r.t. friendly %6.2f (%%%5.1f of ideal)\n",
             methodName,
             ompElapsedTime,
             cacheUnfriendlyElapsedTime / ompElapsedTime,
             cacheFriendlyElapsedTime / ompElapsedTime,
             100. * cacheFriendlyElapsedTime / ompElapsedTime / numberOfThreads);
    } else {
      printf("%-38s : incorrect checksum %lf instead of %lf\n",
             methodName, ompCheckSum, cacheUnfriendlyCheckSum);
    }
  }
  */
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do naive openmp> *********************
  // ===============================================================

  // ===============================================================
  // ********************** < do cuda> *****************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  /*
  // we will repeat the computation for each of the numbers of threads
  const vector<unsigned int> threadsPerBlockArray = {256};
  const vector<unsigned int> maxNumberOfBlocksArray = {10000};

  // warm up cuda
  {
    const unsigned int warmUpMaxNumberOfBlocks = 1e4;
    const unsigned int warmUpThreadsPerBlock   = 256;
    cudaDoMatrixMultiplication(warmUpMaxNumberOfBlocks,
                               warmUpThreadsPerBlock,
                               matrixSize);
  }

  // for each max number of blocks
  for (const unsigned int maxNumberOfBlocks :
         maxNumberOfBlocksArray) {
    // for each number of threads per block
    for (const unsigned int numberOfThreadsPerBlock :
           threadsPerBlockArray) {

      // start timing
      tic = high_resolution_clock::now();

      // do calculation with cuda for this number of threads per block
      for (unsigned int repeatIndex = 0;
           repeatIndex < numberOfRepeats; ++repeatIndex) {
        cudaDoMatrixMultiplication(maxNumberOfBlocks,
                                   numberOfThreadsPerBlock,
                                   matrixSize);
      }

      // stop timing
      toc = high_resolution_clock::now();
      const double cudaElapsedTime =
        duration_cast<duration<double> >(toc - tic).count();

      // check the answer
      double cudaCheckSum = 0;
      for (unsigned int row = 0; row < matrixSize; ++row) {
        for (unsigned int col = 0; col < matrixSize; ++col) {
          cudaCheckSum += resultMatrix(row, col);
        }
      }
      sprintf(methodName, "naive cuda %8.2e blocks %3u threads", double(maxNumberOfBlocks), numberOfThreadsPerBlock);
      if (std::abs(cacheUnfriendlyCheckSum - cudaCheckSum) / cacheUnfriendlyCheckSum < 1e-3) {
        printf("%-38s : time %6.2f speedup w.r.t. unfriendly %6.2f, w.r.t. friendly %6.2f\n",
               methodName,
               cudaElapsedTime,
               cacheUnfriendlyElapsedTime / cudaElapsedTime,
               cacheFriendlyElapsedTime / cudaElapsedTime);
      } else {
        printf("%-38s : incorrect checksum %lf instead of %lf\n",
               methodName, cudaCheckSum, cacheUnfriendlyCheckSum);
      }
    }
  }
  */
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do cuda> *****************************
  // ===============================================================

  // ===============================================================
  // ********************** < do kokkos> ***************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  /*
  Kokkos::initialize();

  //printf("kokkos is running on %s\n", typeid(Kokkos::DefaultExecutionSpace).name());
  matrixView_type left("left", matrixSize*matrixSize);
  matrixView_type right("right", matrixSize*matrixSize);
  matrixView_type result("result", matrixSize*matrixSize);

  host_matrix h_left = Kokkos::create_mirror_view(left);
  host_matrix h_right = Kokkos::create_mirror_view(right);
  host_matrix h_result = Kokkos::create_mirror_view(result);

  for(unsigned index = 0; index < matrixSize*matrixSize; ++index) {
    h_left(index) = leftMatrix(index);
    h_right(index) = rightMatrixRow(index);
    h_result(index) = 0;
  }

  Kokkos::deep_copy(left, h_left);
  Kokkos::deep_copy(right, h_right);
  Kokkos::deep_copy(result, h_result);

  KokkosFunctor kokkosFunctor(matrixSize, left, right, result);
  // start timing
  tic = high_resolution_clock::now();
  matrixView_type finalResults;
  for (unsigned int repeatIndex = 0;
       repeatIndex < numberOfRepeats; ++repeatIndex) {

    Kokkos::parallel_for(matrixSize*matrixSize, kokkosFunctor);
    Kokkos::fence();

  }
  // stop timing
  toc = high_resolution_clock::now();
  const double kokkosElapsedTime =
    duration_cast<duration<double> >(toc - tic).count();

  Kokkos::deep_copy(h_result, result);

  for(unsigned index = 0; index < matrixSize * matrixSize; ++index){
    resultMatrix(index) = h_result(index);
  }
  // check the answer
  double kokkosCheckSum = 0;
  for (unsigned int row = 0; row < matrixSize; ++row) {
    for (unsigned int col = 0; col < matrixSize; ++col) {
      kokkosCheckSum += resultMatrix(row*matrixSize + col);
    }
  }
  sprintf(methodName, "naive kokkos");
  if (std::abs(cacheUnfriendlyCheckSum - kokkosCheckSum) / cacheUnfriendlyCheckSum < 1e-3) {
    printf("%-38s : time %6.2f speedup w.r.t. unfriendly %6.2f, w.r.t. friendly %6.2f\n",
           methodName,
           kokkosElapsedTime,
           cacheUnfriendlyElapsedTime / kokkosElapsedTime,
           cacheFriendlyElapsedTime / kokkosElapsedTime);
  } else {
    printf("%-38s : incorrect checksum %lf instead of %lf\n",
           methodName, kokkosCheckSum, cacheUnfriendlyCheckSum);
  }

  Kokkos::finalize();
  */
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do kokkos> ***************************
  // ===============================================================

  return 0;
}
