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

using std::string;
using std::vector;
using Intrepid::FieldContainer;

typedef Intrepid::RealSpaceTools<double> rst;

//Pre-C++11 timing (thanks jeff)
double                                                                                                                                      getElapsedTime(const timespec start, const timespec end) {                                                                                                   
  timespec temp;                                                                                                                                             
  if ((end.tv_nsec-start.tv_nsec)<0) {                                                                                                                       
    temp.tv_sec = end.tv_sec-start.tv_sec-1;                                                                                                                 
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;                                                                                                     
  } else {                                                                                                                                      temp.tv_sec = end.tv_sec-start.tv_sec;                                                                                                      temp.tv_nsec = end.tv_nsec-start.tv_nsec;                                                                                                 }                                                                                                                                       
  return double(temp.tv_sec) + double(temp.tv_nsec) / 1e9;                                                                                  }       

template<class DeviceType, class LeftViewType, class RightViewType, class OutputViewType>
struct ContractFieldFieldTensorFunctorUnrolled1 {
  typedef DeviceType device_type;
  LeftViewType _leftFields;
  RightViewType _rightFields;
  OutputViewType _outputFields;
  int _numLeftFields;
  int _numRightFields;
  int _numPoints;
  int _dim1Tensor;
  int _dim2Tensor;

  ContractFieldFieldTensorFunctorUnrolled1(LeftViewType leftFields,
				  RightViewType rightFields,
				  OutputViewType outputFields,
				  int numLeftFields,
				  int numRightFields,
				  int numPoints,
				  int dim1Tensor,
				  int dim2Tensor) :
    _leftFields(leftFields),
    _rightFields(rightFields),
    _outputFields(outputFields),
    _numLeftFields(numLeftFields),
    _numRightFields(numRightFields),
    _numPoints(numPoints),
    _dim1Tensor(dim1Tensor),
    _dim2Tensor(dim2Tensor)
  {
    // Nothing to do
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned int elementIndex) const {
    int pointIndex = elementIndex / _numLeftFields;
    int leftField = elementIndex % _numLeftFields;
    
    for (int rbf = 0; rbf < _numRightFields; rbf++) {
      double tmpVal = 0;
      for (int qp = 0; qp < _numPoints; qp++) {
        for (int iTens1 = 0; iTens1 < _dim1Tensor; iTens1++) {
          for (int iTens2 = 0; iTens2 < _dim2Tensor; iTens2++) {
            tmpVal += _leftFields(pointIndex, leftField, qp, iTens1, iTens2)*_rightFields(pointIndex, rbf, qp, iTens1, iTens2);
          } // D2-loo
        } // D1-loop
      } // P-loop
      _outputFields(pointIndex, leftField, rbf) = tmpVal;
    } // R-loop
  }
};


template<class DeviceType, class LeftViewType, class RightViewType, class OutputViewType>
struct ContractFieldFieldTensorFunctor {
  typedef DeviceType device_type;
  LeftViewType _leftFields;
  RightViewType _rightFields;
  OutputViewType _outputFields;
  int _numLeftFields;
  int _numRightFields;
  int _numPoints;
  int _dim1Tensor;
  int _dim2Tensor;

  ContractFieldFieldTensorFunctor(LeftViewType leftFields,
				  RightViewType rightFields,
				  OutputViewType outputFields,
				  int numLeftFields,
				  int numRightFields,
				  int numPoints,
				  int dim1Tensor,
				  int dim2Tensor) :
    _leftFields(leftFields),
    _rightFields(rightFields),
    _outputFields(outputFields),
    _numLeftFields(numLeftFields),
    _numRightFields(numRightFields),
    _numPoints(numPoints),
    _dim1Tensor(dim1Tensor),
    _dim2Tensor(dim2Tensor)
  {
    // Nothing to do
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned int elementIndex) const {

    for (int lbf = 0; lbf < _numLeftFields; lbf++) {
      for (int rbf = 0; rbf < _numRightFields; rbf++) {
      double tmpVal = 0;
        for (int qp = 0; qp < _numPoints; qp++) {
          for (int iTens1 = 0; iTens1 < _dim1Tensor; iTens1++) {
            for (int iTens2 = 0; iTens2 < _dim2Tensor; iTens2++) {
              tmpVal += _leftFields(elementIndex, lbf, qp, iTens1, iTens2)*_rightFields(elementIndex, rbf, qp, iTens1, iTens2);
            } // D2-loop
          } // D1-loop
        } // P-loop
        _outputFields(elementIndex, lbf, rbf) = tmpVal;
      } // R-loop
    } // L-loop
  }
};


template <class DeviceType>
void contractFieldFieldTensor(FieldContainer<double> & outputFields,
			      const FieldContainer<double> &   leftFields,
			      const FieldContainer<double> &  rightFields,
                              const int compEngine,
			      double* time = 0) {

  typedef Kokkos::View<double *****, DeviceType> input_view_t;
  typedef Kokkos::View<double ***, DeviceType> output_view_t;

  typedef typename input_view_t::HostMirror input_host_t;
  typedef typename output_view_t::HostMirror output_host_t;


  // get sizes
  int numCells        = leftFields.dimension(0);
  int numLeftFields   = leftFields.dimension(1);
  int numRightFields  = rightFields.dimension(1);
  int numPoints       = leftFields.dimension(2);
  int dim1Tensor      = leftFields.dimension(3);
  int dim2Tensor      = leftFields.dimension(4);

  switch(compEngine) {
    case 0: {
        for (int cl = 0; cl < numCells; cl++) {
	        for (int lbf = 0; lbf < numLeftFields; lbf++) {
	          for (int rbf = 0; rbf < numRightFields; rbf++) {
	          double tmpVal = 0;
	            for (int qp = 0; qp < numPoints; qp++) {
	              for (int iTens1 = 0; iTens1 < dim1Tensor; iTens1++) {
		              for (int iTens2 = 0; iTens2 < dim2Tensor; iTens2++) {

                    tmpVal += leftFields(cl, lbf, qp, iTens1, iTens2) *
                              rightFields(cl, rbf, qp, iTens1, iTens2);
		              } // D2-loop
	              } // D1-loop
	            } // P-loop
	            outputFields(cl, lbf, rbf) = tmpVal;
	          } // R-loop
	        } // L-loop
        } // C-loop
      }
    break;

  case 1: {
    /*
       GEMM parameters and their values.
       (Note: It is assumed that the result needs to be transposed into row-major format.
              Think of left and right input matrices as A(p x l) and B(p x r), respectively,
              even though the indexing is ((C),L,P) and ((C),R,P). Due to BLAS formatting
              assumptions, we are computing (A^T*B)^T = B^T*A.)
       TRANSA   TRANS
       TRANSB   NO_TRANS
       M        #rows(B^T)                            = number of right fields
       N        #cols(A)                              = number of left fields
       K        #cols(B^T)                            = number of integration points * size of tensor
       ALPHA    1.0
       A        right data for cell cl                = &rightFields[cl*skipR]
       LDA      #rows(B)                              = number of integration points * size of tensor
       B        left data for cell cl                 = &leftFields[cl*skipL]
       LDB      #rows(A)                              = number of integration points * size of tensor
       BETA     0.0
       C        result for cell cl                    = &outputFields[cl*skipOp]
       LDC      #rows(C)                              = number of right fields
      */

    //TODO this breaks the compiler right now
    /*
      int numData  = numPoints*dim1Tensor*dim2Tensor;
      int skipL    = numLeftFields*numData;         // size of the left data chunk per cell
      int skipR    = numRightFields*numData;        // size of the right data chunk per cell
      int skipOp   = numLeftFields*numRightFields;  // size of the output data chunk per cell
      double alpha = 1.0;                            // these are left unchanged by GEMM
      double  beta = 0.0;

      for (int cl=0; cl < numCells; cl++) {
        // Use this if data is used in row-major format
        Teuchos::BLAS<int, double> myblas;
        myblas.GEMM(Teuchos::TRANS, Teuchos::NO_TRANS,
                    numRightFields, numLeftFields, numData,
                    alpha, &rightFields[cl*skipR], numData,
                    &leftFields[cl*skipL], numData,
                    beta, &outputFields[cl*skipOp], numRightFields);
    */
	// Use this if data is used in column-major format
        /*
        myblas.GEMM(Teuchos::TRANS, Teuchos::NO_TRANS,
                    numLeftFields, numRightFields, numData,
                    alpha, &leftFields[cl*skipL], numData,
                    &rightFields[cl*skipR], numData,
                    beta, &outputFields[cl*skipOp], numLeftFields);
        */
    //}
    }
    break;
  case 2: {


    Kokkos::initialize();

    input_view_t kokkosLeft("left_input", numCells, numLeftFields, numPoints, dim1Tensor, dim2Tensor);

    input_view_t kokkosRight("right_input", numCells, numRightFields, numPoints, dim1Tensor, dim2Tensor);
    output_view_t kokkosOut("output", numCells, numLeftFields, numRightFields);

    input_host_t hostLeft = Kokkos::create_mirror_view(kokkosLeft);
    input_host_t hostRight = Kokkos::create_mirror_view(kokkosRight);
    output_host_t hostOut = Kokkos::create_mirror_view(kokkosOut);

    // Copy everything
    for (int cl = 0; cl < numCells; ++cl) {
      for (int lbf = 0; lbf < numLeftFields; ++lbf) {
	for (int qp = 0; qp < numPoints; ++qp) {
	  for (int iTens1 = 0; iTens1 < dim1Tensor; ++iTens1) {
	    for (int iTens2 = 0; iTens2 < dim2Tensor; ++iTens2) {
	      hostLeft(cl, lbf, qp, iTens1, iTens2) = leftFields(cl, lbf, qp, iTens1, iTens2);
	    }
          }
        }
      }

      for (int rbf = 0; rbf < numRightFields; ++rbf) {
	for (int qp = 0; qp < numPoints; ++qp) {
	  for (int iTens1 = 0; iTens1 < dim1Tensor; ++iTens1) {
	    for (int iTens2 = 0; iTens2 < dim2Tensor; ++iTens2) {
	      hostRight(cl, rbf, qp, iTens1, iTens2) = rightFields(cl, rbf, qp, iTens1, iTens2);
	    }
          }
        }
      }

      for (int lbf = 0; lbf < numLeftFields; ++lbf) {
	for (int rbf = 0; rbf < numRightFields; ++rbf) {
	  hostOut(cl, lbf, rbf) = outputFields(cl, lbf, rbf);
        }
      }
    }
    
    Kokkos::deep_copy(kokkosLeft, hostLeft);
    Kokkos::deep_copy(kokkosRight, hostRight);
    Kokkos::deep_copy(kokkosOut, hostOut);
    
    timespec tic;
    if(time != 0)
      clock_gettime(CLOCK_MONOTONIC, &tic);
 
    ContractFieldFieldTensorFunctor<DeviceType, input_view_t, input_view_t, output_view_t>
      kokkosFunctor(kokkosLeft, kokkosRight, kokkosOut,
		    numLeftFields, numRightFields, numPoints,
		    dim1Tensor, dim2Tensor);
    Kokkos::parallel_for(numCells, kokkosFunctor);

    Kokkos::fence();
    
    timespec toc;
    if(time !=0){
      clock_gettime(CLOCK_MONOTONIC, &toc);
      *time += getElapsedTime(tic, toc);
    }
    Kokkos::deep_copy(hostOut, kokkosOut);

    for (int cl = 0; cl < numCells; ++cl) {
      for (int lbf = 0; lbf < numLeftFields; ++lbf) {
        for (int rbf = 0; rbf < numRightFields; ++rbf) {
          outputFields(cl, lbf, rbf) = hostOut(cl, lbf, rbf);
        }
      }
    }
    
    Kokkos::finalize();

  }
  break;


  }
}

int main(int argc, char* argv[]) {

  // ===============================================================
  // ********************** < do kokkos> ***************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  int c=5000, p=20, l=3, r=7, d1=13, d2=5;

  FieldContainer<double> in_c_l_p_d_d(c, l, p, d1, d2);
  FieldContainer<double> in_c_r_p_d_d(c, r, p, d1, d2);
  FieldContainer<double> out1_c_l_r(c, l, r);
  FieldContainer<double> out2_c_l_r(c, l, r);
  double zero = Intrepid::INTREPID_TOL*10000.0;

  /* I got rid of the typedefs - for now
   * 0 -> manual computation
   * 1 -> blas
   * 2 -> kokkos
   */

  // fill with random numbers
  for (int i=0; i<in_c_l_p_d_d.size(); i++) {
    in_c_l_p_d_d[i] = Teuchos::ScalarTraits<double>::random();
  }
  for (int i=0; i<in_c_r_p_d_d.size(); i++) {
    in_c_r_p_d_d[i] = Teuchos::ScalarTraits<double>::random();
  }

  printf("trying kokkos openmp 0, (manual)\n");
  
  //Warmup
  contractFieldFieldTensor<Kokkos::OpenMP>(out1_c_l_r, in_c_l_p_d_d, in_c_r_p_d_d,0);
  
  timespec tic;
  clock_gettime(CLOCK_MONOTONIC, &tic);
    
  //repeat the calculation 5 times so we can average out some randomness
  for(int i = 0; i < 5; ++i){
    contractFieldFieldTensor<Kokkos::OpenMP>(out1_c_l_r, in_c_l_p_d_d, in_c_r_p_d_d, 0);
  }

  timespec toc;
  clock_gettime(CLOCK_MONOTONIC, &toc);
  const double elapsedTime_manual = getElapsedTime(tic, toc);
  
  printf("trying kokkos openmp 2\n");
  
  //Warmpup
  contractFieldFieldTensor<Kokkos::OpenMP>(out2_c_l_r, in_c_l_p_d_d, in_c_r_p_d_d, 2);  
  clock_gettime(CLOCK_MONOTONIC, &tic);

  //repeat the calculation 5 times so we can average out some randomness                                                                                     
  for(int i = 0; i < 5; ++i){
    contractFieldFieldTensor<Kokkos::OpenMP>(out1_c_l_r, in_c_l_p_d_d, in_c_r_p_d_d, 2);
  }

  clock_gettime(CLOCK_MONOTONIC, &toc);
  const double elapsedTime_kokkos = getElapsedTime(tic, toc);
  
  rst::subtract(&out1_c_l_r[0], &out2_c_l_r[0], out2_c_l_r.size());
  if (rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) > zero) {
    std::cout << "\n\nINCORRECT contractFieldFieldTensor (1): check COMP_CPP vs. COMP_KOKKOS; "
	      << " diff-1norm = " << rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) << "\n\n";
  }
  else {
    std::cout << "Cpp and Kokkos get same result, kokkos speedup of " << elapsedTime_manual/elapsedTime_kokkos << std::endl;
  }
  
  //Now try the kokkos version without the copying of things in/out
  double elapsedTime_kokkos_noCopy = 0;
  
  for(int i = 0; i < 5; ++i){
    contractFieldFieldTensor<Kokkos::OpenMP>(out1_c_l_r, in_c_l_p_d_d, in_c_r_p_d_d, 2, &elapsedTime_kokkos_noCopy);
  }
  
  std::cout << "not counting copying yields a speedup of " << elapsedTime_manual/elapsedTime_kokkos_noCopy << std::endl;

  printf("trying cuda cuda 0, (manual)\n");
  
  //Warmup
  contractFieldFieldTensor<Kokkos::Cuda>(out1_c_l_r, in_c_l_p_d_d, in_c_r_p_d_d,0);
  
  clock_gettime(CLOCK_MONOTONIC, &tic);
    
  //repeat the calculation 5 times so we can average out some randomness
  for(int i = 0; i < 5; ++i){
    contractFieldFieldTensor<Kokkos::Cuda>(out1_c_l_r, in_c_l_p_d_d, in_c_r_p_d_d, 0);
  }
  clock_gettime(CLOCK_MONOTONIC, &toc);
  const double elapsedTime_manual_cuda = getElapsedTime(tic, toc);
  
  printf("trying kokkos cuda 2\n");
  
  //Warmpup
  contractFieldFieldTensor<Kokkos::Cuda>(out2_c_l_r, in_c_l_p_d_d, in_c_r_p_d_d, 2);  
  clock_gettime(CLOCK_MONOTONIC, &tic);

  //repeat the calculation 5 times so we can average out some randomness                                                                                     
  for(int i = 0; i < 5; ++i){
    contractFieldFieldTensor<Kokkos::Cuda>(out1_c_l_r, in_c_l_p_d_d, in_c_r_p_d_d, 2);
  }

  clock_gettime(CLOCK_MONOTONIC, &toc);
  const double elapsedTime_kokkos_cuda = getElapsedTime(tic, toc);
  
  std::cout << "kokkos speedup of " << elapsedTime_manual_cuda/elapsedTime_kokkos_cuda << std::endl;
  
  //Now try the kokkos version without the copying of things in/out
  double elapsedTime_kokkos_noCopyCuda = 0;
  
  for(int i = 0; i < 5; ++i){
    contractFieldFieldTensor<Kokkos::Cuda>(out1_c_l_r, in_c_l_p_d_d, in_c_r_p_d_d, 2, &elapsedTime_kokkos_noCopyCuda);
  }
  
  std::cout << "not counting copying yields a speedup of " << elapsedTime_manual_cuda/elapsedTime_kokkos_noCopyCuda << std::endl;


  return 0;
}
