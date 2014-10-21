// -*- C++ -*-
// ex0_scalarIntegrator.cc
// an exercise for the sandia 2014 clinic team.
// here we integrate sin over some interval using various methods

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>
#include <string>
#include <chrono>

// header file for openmp
#include <omp.h>

// header files for tbb
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>

// header files for cuda implementation
#include "ex0_scalarIntegrator_cuda.cuh"

// header files for kokkos
#include <Kokkos_Core.hpp>

using std::string;
using std::vector;
using std::array;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

class myFunctor {
public:
  myFunctor() {

  }

  double operator()(double functionInput) {
    return std::sin(functionInput);
  }
};

class TbbOutputter {
public:

  const double startingPoint_;

  const double dx_;

  myFunctor function_;

  double sum_;

  TbbOutputter(const double startingPoint, const double dx, myFunctor
    function) :
    startingPoint_(startingPoint), dx_(dx), function_(function), sum_(0) {
  }

  TbbOutputter(const TbbOutputter & other,
               tbb::split) :
    startingPoint_(other.startingPoint_), dx_(other.dx_),
    function_(other.function_), sum_(0) {

  }

  void operator()(const tbb::blocked_range<size_t> & range) {

    double sum = sum_;

    for(unsigned int i=range.begin(); i!= range.end(); ++i )
            sum += function_(startingPoint_ + (double(i) + .5) * dx_);

    sum_ = sum;
  }

  void join(const TbbOutputter & other) {
    sum_ += other.sum_;
  }

private:
  TbbOutputter();

};

struct KokkosFunctor {
  typedef double value_type;
  const double dx_;

  KokkosFunctor(double dx): dx_(dx) {}


  KOKKOS_INLINE_FUNCTION
  void operator()(int intervalIndex, double &sum) const {
    sum += std::sin((double(intervalIndex) + .5)*dx_);
  }

private:
  KokkosFunctor();

};

int main(int argc, char* argv[]) {

  // a couple of inputs.  change the numberOfIntervals to control the amount
  //  of work done
  const unsigned long numberOfIntervals = 1e8;
  // the integration bounds
  const array<double, 2> bounds = {{0, 1.314}};

  // these are c++ timers...for timing
  high_resolution_clock::time_point tic;
  high_resolution_clock::time_point toc;

  const double dx = (bounds[1] - bounds[0]) / numberOfIntervals;

  // calculate analytic solution
  const double libraryAnswer =
    std::cos(bounds[0]) - std::cos(bounds[1]);

  // ===============================================================
  // ********************** < do slow serial> **********************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  double slowSerialIntegral = 0;
  tic = high_resolution_clock::now();
  for (unsigned int intervalIndex = 0;
       intervalIndex < numberOfIntervals; ++intervalIndex) {
    if (intervalIndex % (numberOfIntervals / 10) == 0) {
      printf("serial calculation on interval %8.2e / %8.2e (%%%5.1f)\n",
             double(intervalIndex),
             double(numberOfIntervals),
             100. * intervalIndex / double(numberOfIntervals));
    }
    const double evaluationPoint =
      bounds[0] + (double(intervalIndex) + 0.5) * dx;
    slowSerialIntegral += std::sin(evaluationPoint);
  }
  slowSerialIntegral *= dx;
  toc = high_resolution_clock::now();
  const double slowSerialElapsedTime =
    duration_cast<duration<double> >(toc - tic).count();

  // check our serial answer
  const double serialRelativeError =
    std::abs(libraryAnswer - slowSerialIntegral) / std::abs(libraryAnswer);
  if (serialRelativeError > 1e-3) {
    fprintf(stderr, "our answer is too far off: %15.8e instead of %15.8e\n",
            slowSerialIntegral, libraryAnswer);
    exit(1);
  }

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do slow serial> **********************
  // ===============================================================

  // ===============================================================
  // ********************** < do fast serial> **********************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  const double fastSerialIntegral = slowSerialIntegral;
  const double fastSerialElapsedTime = slowSerialElapsedTime;

  // check our serial answer
  const double fastSerialRelativeError =
    std::abs(libraryAnswer - fastSerialIntegral) / std::abs(libraryAnswer);
  if (fastSerialRelativeError > 1e-3) {
    fprintf(stderr, "our answer is too far off: %15.8e instead of %15.8e\n",
            fastSerialIntegral, libraryAnswer);
    exit(1);
  }

  // output speedup
  printf("fast: time %8.2e speedup %8.2e\n",
         fastSerialElapsedTime,
         slowSerialElapsedTime / fastSerialElapsedTime);

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do fast serial> **********************
  // ===============================================================

  // we will repeat the computation for each of the numbers of threads
  vector<unsigned int> numberOfThreadsArray;
  numberOfThreadsArray.push_back(1);
  numberOfThreadsArray.push_back(2);
  numberOfThreadsArray.push_back(4);
  numberOfThreadsArray.push_back(8);
  numberOfThreadsArray.push_back(16);
  numberOfThreadsArray.push_back(24);

  // ===============================================================
  // ********************** < do tbb> ******************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  myFunctor function;
  printf("performing calculations with tbb\n");
  // for each number of threads
  for (const unsigned int numberOfThreads :
         numberOfThreadsArray) {

    // initialize tbb's threading system for this number of threads
    tbb::task_scheduler_init init(numberOfThreads);

    TbbOutputter tbbOutputter(bounds[0], dx, function);

    // start timing
    tic = high_resolution_clock::now();
    // dispatch threads
    parallel_reduce(tbb::blocked_range<size_t>(0, numberOfIntervals),
                    tbbOutputter);
    // stop timing
    toc = high_resolution_clock::now();
    const double threadedElapsedTime =
      duration_cast<duration<double> >(toc - tic).count();

    // somehow get the threaded integral answer
    const double threadedIntegral = tbbOutputter.sum_ * dx;
    // check the answer
    const double threadedRelativeError =
      std::abs(libraryAnswer - threadedIntegral) / std::abs(libraryAnswer);

    if (threadedRelativeError > 1e-3) {
      fprintf(stderr, "our answer is too far off: %15.8e instead of %15.8e\n",
              threadedIntegral, libraryAnswer);
      exit(1);
    }

    // output speedup
    printf("%3u : time %8.2e speedup %8.2e (%%%5.1f of ideal)\n",
           numberOfThreads,
           threadedElapsedTime,
           fastSerialElapsedTime / threadedElapsedTime,
           100. * fastSerialElapsedTime / threadedElapsedTime / numberOfThreads);
  }

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do tbb> ******************************
  // ===============================================================


  // ===============================================================
  // ********************** < do openmp> ***************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  printf("performing calculations with openmp\n");
  // for each number of threads
  for (const unsigned int numberOfThreads :
         numberOfThreadsArray) {

    double threadedIntegral = 0;
    // start timing
    tic = high_resolution_clock::now();

    omp_set_num_threads(numberOfThreads);

    #pragma omp parallel for reduction(+:threadedIntegral)
      for(unsigned int i = 0; i < numberOfIntervals; i += 1) {
        threadedIntegral += std::sin((double(i)+.5)*dx + bounds[0]);
      }

    threadedIntegral *= dx;
    // stop timing
    toc = high_resolution_clock::now();
    const double threadedElapsedTime =
      duration_cast<duration<double> >(toc - tic).count();

    // check the answer
    const double threadedRelativeError =
      std::abs(libraryAnswer - threadedIntegral) / std::abs(libraryAnswer);
    if (threadedRelativeError > 1) {
      fprintf(stderr, "our answer is too far off: %15.8e instead of %15.8e\n",
              threadedIntegral, libraryAnswer);
      //exit(1);
    }

    // output speedup
    printf("%3u : time %8.2e speedup %8.2e (%%%5.1f of ideal)\n",
           numberOfThreads,
           threadedElapsedTime,
           fastSerialElapsedTime / threadedElapsedTime,
           100. * fastSerialElapsedTime / threadedElapsedTime / numberOfThreads);
  }

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do openmp> ***************************
  // ===============================================================

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

    float cudaIntegral = 0;
    // start timing
    tic = high_resolution_clock::now();
    cudaDoScalarIntegration(numberOfThreadsPerBlock,
                            &cudaIntegral, bounds[0],
                            numberOfIntervals, dx);
    // stop timing
    toc = high_resolution_clock::now();
    const double cudaElapsedTime =
      duration_cast<duration<double> >(toc - tic).count();

    // check the answer
    const double cudaRelativeError =
      std::abs(float(libraryAnswer) - cudaIntegral) / std::abs(libraryAnswer);
    if (cudaRelativeError > 1e-3) {
      fprintf(stderr, "our answer is too far off: %15.8e instead of %15.8e\n",
              cudaIntegral, libraryAnswer);
      //exit(1);
    }

    // output speedup
    printf("%3u : time %8.2e speedup %8.2e\n",
           numberOfThreadsPerBlock,
           cudaElapsedTime,
           fastSerialElapsedTime / cudaElapsedTime);
  }

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do cuda> *****************************
  // ===============================================================


  // ===============================================================
  // ********************** < do kokkos> *****************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  printf("performing calculations with kokkos on %s\n",
         typeid(Kokkos::DefaultExecutionSpace).name());

  Kokkos::initialize();

  //const funct KokkosFunctor();

  // start timing
  tic = high_resolution_clock::now();

  double kokkosIntegral = 0;

  Kokkos::parallel_reduce(numberOfIntervals, KokkosFunctor(dx),kokkosIntegral);

  kokkosIntegral *= dx;
  // stop timing
  toc = high_resolution_clock::now();
  const double kokkosElapsedTime =
    duration_cast<duration<double> >(toc - tic).count();

  // check the answer
  const double kokkosRelativeError =
    std::abs(libraryAnswer - kokkosIntegral) / std::abs(libraryAnswer);
  if (kokkosRelativeError > 1e-3) {
    fprintf(stderr, "our answer is too far off: %15.8e instead of %15.8e\n",
            kokkosIntegral, libraryAnswer);
    exit(1);
  }

  // output speedup
  printf("kokkos : time %8.2e speedup %8.2e\n",
         kokkosElapsedTime,
         fastSerialElapsedTime / kokkosElapsedTime);

  Kokkos::finalize();

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do kokkos> ***************************
  // ===============================================================

  return 0;
}
