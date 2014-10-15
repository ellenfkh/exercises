// -*- C++ -*-
// ex3_naiveMatrixMultiplication.cc
// an exercise for the sandia 2014 clinic team.
// here we do matrix multiplication using the naive triple-nested for loop
//  technique.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>
#include <string>
#include <chrono>
#include <algorithm>

// header file for openmp
#include <omp.h>

// header files for tbb
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>

// header files for cuda implementation
#include "ex3_naiveMatrixMultiplication_cuda.cuh"

// header file for kokkos
#include <Kokkos_Core.hpp>

using std::string;
using std::vector;
using std::array;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

struct ColMajorMatrix {
  const unsigned int _matrixSize;
  vector<double> _data;

  ColMajorMatrix(const unsigned int matrixSize) :
    _matrixSize(matrixSize), _data(_matrixSize*_matrixSize) {
  }

  inline
  double &
  operator()(const unsigned int index) {
    return _data[index];
  }

  inline
  double
  operator()(const unsigned int index) const {
    return _data[index];
  }

  inline
  double &
  operator()(const unsigned int row, const unsigned int col) {
    return _data[row + col * _matrixSize];
  }

  inline
  double
  operator()(const unsigned int row, const unsigned int col) const {
    return _data[row + col * _matrixSize];
  }

  void
  fill(const double d) {
    std::fill(_data.begin(), _data.end(), 0);
  }
};

struct RowMajorMatrix {
  const unsigned int _matrixSize;
  vector<double> _data;

  RowMajorMatrix(const unsigned int matrixSize) :
    _matrixSize(matrixSize), _data(_matrixSize*_matrixSize) {
  }

  inline
  double &
  operator()(const unsigned int index) {
    return _data[index];
  }

  inline
  double
  operator()(const unsigned int index) const {
    return _data[index];
  }

  inline
  double &
  operator()(const unsigned int row, const unsigned int col) {
    return _data[row * _matrixSize + col];
  }

  inline
  double
  operator()(const unsigned int row, const unsigned int col) const {
    return _data[row * _matrixSize + col];
  }

  void
  fill(const double d) {
    std::fill(_data.begin(), _data.end(), 0);
  }
};

class TbbFunctor {
public:

  const unsigned int _matrixSize;
  RowMajorMatrix * _leftMatrix;
  ColMajorMatrix * _rightMatrix;
  RowMajorMatrix * _resultMatrix;

  TbbFunctor(const unsigned int matrixSize, RowMajorMatrix * leftMatrix,
              ColMajorMatrix * rightMatrix, RowMajorMatrix * resultMatrix) :
    _matrixSize(matrixSize), _leftMatrix(leftMatrix), _rightMatrix(rightMatrix),
    _resultMatrix(resultMatrix) {
  }

  void operator()(const tbb::blocked_range<size_t> & range) const {
    for(unsigned int i = range.begin(); i != range.end(); ++i) {
      unsigned int row = i / _matrixSize;
      unsigned int col = i % _matrixSize;

      for(unsigned int dummy = 0; dummy < _matrixSize; ++dummy) {
        _resultMatrix->operator()(i) += _leftMatrix->operator()(row, dummy) *
        _rightMatrix->operator()(dummy, col);
      }
    }

  }

private:
  TbbFunctor();

};

struct KokkosFunctor {

  const unsigned int _matrixSize;

  KokkosFunctor(const unsigned int matrixSize) :
    _matrixSize(matrixSize) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned int elementIndex) const {
  }

private:
  KokkosFunctor();

};

int main(int argc, char* argv[]) {

  // a couple of inputs.  change the numberOfIntervals to control the amount
  //  of work done
  const unsigned int matrixSize = (unsigned int)(5e2 / 8) * 8;
  const unsigned int numberOfRepeats = 3;

  // these are c++ timers...for timing
  high_resolution_clock::time_point tic;
  high_resolution_clock::time_point toc;

  // create a c++11 random number generator
  std::mt19937 randomNumberEngine;
  std::uniform_real_distribution<double> randomNumberGenerator(0, 1);

  printf("Creating the operands\n");
  RowMajorMatrix leftMatrix(matrixSize);
  RowMajorMatrix rightMatrix(matrixSize);
  RowMajorMatrix resultMatrix(matrixSize);
  for (unsigned int row = 0; row < matrixSize; ++row) {
    for (unsigned int col = 0; col < matrixSize; ++col) {
      leftMatrix(row, col) = randomNumberGenerator(randomNumberEngine);
      rightMatrix(row, col) = randomNumberGenerator(randomNumberEngine);
    }
  }

  // ===============================================================
  // ********************** < do slow serial> **********************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  printf("performing slow serial\n");

  tic = high_resolution_clock::now();
  for (unsigned int repeatIndex = 0;
       repeatIndex < numberOfRepeats; ++repeatIndex) {
    for (unsigned int row = 0; row < matrixSize; ++row) {
      for (unsigned int col = 0; col < matrixSize; ++col) {
        resultMatrix(row, col) = 0;
        for (unsigned int dummy = 0; dummy < matrixSize; ++dummy) {
          resultMatrix(row, col) +=
            leftMatrix(row, dummy) * rightMatrix(dummy, col);
        }
      }
    }
  }
  toc = high_resolution_clock::now();
  const double slowSerialElapsedTime =
    duration_cast<duration<double> >(toc - tic).count();

  double slowSerialCheckSum = 0;
  for (unsigned int row = 0; row < matrixSize; ++row) {
    for (unsigned int col = 0; col < matrixSize; ++col) {
      slowSerialCheckSum += resultMatrix(row, col);
    }
  }

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do slow serial> **********************
  // ===============================================================

  resultMatrix.fill(0);

  // ===============================================================
  // ********************** < do fast serial> **********************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  printf("performing fast serial\n");

  ColMajorMatrix fastRightMatrix(matrixSize);

  for (unsigned int row = 0; row < matrixSize; ++row) {
    for (unsigned int col = 0; col < matrixSize; ++col) {
      fastRightMatrix(row,col) = rightMatrix(row,col);
    }
  }

  tic = high_resolution_clock::now();

  for (unsigned int repeatIndex = 0;
       repeatIndex < numberOfRepeats; ++repeatIndex) {
    resultMatrix.fill(0);
    for (unsigned int row = 0; row < matrixSize; ++row) {
      for (unsigned int col = 0; col < matrixSize; ++col) {

        for (unsigned int dummy = 0; dummy < matrixSize; ++dummy) {
          resultMatrix(row, col) +=
            leftMatrix(row, dummy) * fastRightMatrix(dummy, col);
        }
      }
    }

  }

  toc = high_resolution_clock::now();
  const double fastSerialElapsedTime =
    duration_cast<duration<double> >(toc - tic).count();

  double fastSerialCheckSum = 0;
  for (unsigned int row = 0; row < matrixSize; ++row) {
    for (unsigned int col = 0; col < matrixSize; ++col) {
      fastSerialCheckSum += resultMatrix(row, col);
    }
  }
  if (std::abs(slowSerialCheckSum - fastSerialCheckSum) / slowSerialCheckSum > 1e-3) {
    fprintf(stderr, "incorrect checksum = %lf, correct is %lf\n",
            fastSerialCheckSum, slowSerialCheckSum);
    exit(1);
  }


  // output speedup
  printf("fast: time %8.2e speedup w.r.t. slow serial %8.2e\n",
         fastSerialElapsedTime,
         slowSerialElapsedTime / fastSerialElapsedTime);

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do fast serial> **********************
  // ===============================================================

  resultMatrix.fill(0);

  // we will repeat the computation for each of the numbers of threads
  vector<unsigned int> numberOfThreadsArray;
  numberOfThreadsArray.push_back(1);
  numberOfThreadsArray.push_back(2);
  numberOfThreadsArray.push_back(4);
  numberOfThreadsArray.push_back(8);
  numberOfThreadsArray.push_back(16);
  numberOfThreadsArray.push_back(24);

  const size_t grainSize =
    std::max(unsigned(1e1), matrixSize / 48);

  // ===============================================================
  // ********************** < do tbb> ******************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  printf("performing calculations with tbb\n");
  // for each number of threads
  for (const unsigned int numberOfThreads :
         numberOfThreadsArray) {

    // initialize tbb's threading system for this number of threads
    tbb::task_scheduler_init init(numberOfThreads);

    // prepare the tbb functor.
    const TbbFunctor tbbFunctor(matrixSize, &leftMatrix, &fastRightMatrix,
                  &resultMatrix);

    // start timing
    tic = high_resolution_clock::now();
    for (unsigned int repeatIndex = 0;
         repeatIndex < numberOfRepeats; ++repeatIndex) {

      resultMatrix.fill(0);
      // dispatch threads
      parallel_for(tbb::blocked_range<size_t>(0, matrixSize*matrixSize, grainSize),
                   tbbFunctor);

    }
    // stop timing
    toc = high_resolution_clock::now();
    const double tbbElapsedTime =
      duration_cast<duration<double> >(toc - tic).count();

    // check the answer
    double tbbCheckSum = 0;
    for (unsigned int row = 0; row < matrixSize; ++row) {
      for (unsigned int col = 0; col < matrixSize; ++col) {
        tbbCheckSum += resultMatrix(row, col);
      }
    }
    if (std::abs(slowSerialCheckSum - tbbCheckSum) / slowSerialCheckSum > 1e-3) {
      fprintf(stderr, "incorrect checksum = %lf, correct is %lf\n",
              tbbCheckSum, slowSerialCheckSum);
      exit(1);
    }

    // output speedup
    printf("%3u : time %8.2e speedup w.r.t fast serial %8.2e (%%%5.1f of ideal)\n",
           numberOfThreads,
           tbbElapsedTime,
           fastSerialElapsedTime / tbbElapsedTime,
           100. * fastSerialElapsedTime / tbbElapsedTime / numberOfThreads);
  }

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do tbb> ******************************
  // ===============================================================

  resultMatrix.fill(0);

  // ===============================================================
  // ********************** < do openmp> ***************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  printf("performing calculations with openmp\n");
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
          resultMatrix->operator()(i) += leftMatrix->operator()(row, dummy) *
          fastRightMatrix->operator()(dummy, col);
        }
      }
    }

    // stop timing
    toc = high_resolution_clock::now();
    const double threadedElapsedTime =
      duration_cast<duration<double> >(toc - tic).count();

    // check the answer
    double ompCheckSum = 0;
    for (unsigned int row = 0; row < matrixSize; ++row) {
      for (unsigned int col = 0; col < matrixSize; ++col) {
        ompCheckSum += resultMatrix(row, col);
      }
    }
    if (std::abs(slowSerialCheckSum - ompCheckSum) / slowSerialCheckSum > 1e-3) {
      fprintf(stderr, "incorrect checksum = %lf, correct is %lf\n",
              ompCheckSum, slowSerialCheckSum);
      exit(1);
    }

    // output speedup
    printf("%3u : time %8.2e speedup w.r.t fast serial %8.2e (%%%5.1f of ideal)\n",
           numberOfThreads,
           threadedElapsedTime,
           fastSerialElapsedTime / threadedElapsedTime,
           100. * fastSerialElapsedTime / threadedElapsedTime / numberOfThreads);
  }

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do openmp> ***************************
  // ===============================================================

  resultMatrix.fill(0);

  // ===============================================================
  // ********************** < do cuda> *****************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  // we will repeat the computation for each of the numbers of threads
  vector<unsigned int> threadsPerBlockArray;
  threadsPerBlockArray.push_back(32);
  threadsPerBlockArray.push_back(64);
  threadsPerBlockArray.push_back(128);
  threadsPerBlockArray.push_back(256);
  threadsPerBlockArray.push_back(512);

  printf("performing calculations with cuda\n");
  // for each number of threads per block
  for (const unsigned int numberOfThreadsPerBlock :
         threadsPerBlockArray) {

    // start timing
    tic = high_resolution_clock::now();

    // TODO: do cuda stuff

    // do calculation with cuda for this number of threads per block
    cudaDoNaiveMatrixMultiplication(numberOfThreadsPerBlock,
                                    matrixSize,
                                    &resultMatrix(0, 0));

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
    if (std::abs(slowSerialCheckSum - cudaCheckSum) / slowSerialCheckSum > 1e-3) {
      fprintf(stderr, "incorrect checksum = %lf, correct is %lf\n",
              cudaCheckSum, slowSerialCheckSum);
      exit(1);
    }

    // output speedup
    printf("%3u : time %8.2e speedup w.r.t fast serial %8.2e\n",
           numberOfThreadsPerBlock,
           cudaElapsedTime,
           fastSerialElapsedTime / cudaElapsedTime);
  }

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do cuda> *****************************
  // ===============================================================

  resultMatrix.fill(0);

  // ===============================================================
  // ********************** < do kokkos> *****************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  printf("performing calculations with kokkos running on %s\n",
         typeid(Kokkos::DefaultExecutionSpace).name());

  Kokkos::initialize();

  KokkosFunctor kokkosFunctor(matrixSize);
  // TODO: do kokkos stuff

  // start timing
  tic = high_resolution_clock::now();

  for (unsigned int repeatIndex = 0;
       repeatIndex < numberOfRepeats; ++repeatIndex) {
    // TODO: multiply with kokkos
  }

  // stop timing
  toc = high_resolution_clock::now();
  const double kokkosElapsedTime =
    duration_cast<duration<double> >(toc - tic).count();

  // check the answer
  double kokkosCheckSum = 0;
  for (unsigned int row = 0; row < matrixSize; ++row) {
    for (unsigned int col = 0; col < matrixSize; ++col) {
      kokkosCheckSum += resultMatrix(row, col);
    }
  }
  if (std::abs(slowSerialCheckSum - kokkosCheckSum) / slowSerialCheckSum > 1e-3) {
    fprintf(stderr, "incorrect checksum = %lf, correct is %lf\n",
            kokkosCheckSum, slowSerialCheckSum);
    exit(1);
  }

  // output speedup
  printf("kokkos : time %8.2e speedup w.r.t fast serial %8.2e\n",
         kokkosElapsedTime,
         fastSerialElapsedTime / kokkosElapsedTime);


  Kokkos::finalize();
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do kokkos> ***************************
  // ===============================================================

  return 0;
}
