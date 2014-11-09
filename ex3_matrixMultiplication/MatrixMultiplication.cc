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
#include "Teuchos_Array.hpp"
#include "Intrepid_ArrayTools.hpp"
#include "Intrepid_FieldContainer.hpp"

using std::string;
using std::vector;
using std::array;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;
/*
template<class Scalar, class LeftViewType, class RightViewType, class OutputViewType>
struct ContractFieldFieldTensorFunctor {
  LeftViewType _leftFields;
  RightViewType _rightFields;
  OutputViewType _outputFields;
  int _numLeftFields;
  int _numRightFields;
  int _numPoints;
  int _dim1Tensor;
  int _dim2Tensor;
  bool _sumInto;

  ContractFieldFieldTensorFunctor(
				  LeftViewType leftFields,
				  RightViewType rightFields,
				  OutputViewType outputFields,
				  int numLeftFields,
				  int numRightFields,
				  int numPoints,
				  int dim1Tensor,
				  int dim2Tensor,
				  bool sumInto) :
    _leftFields(leftFields),
    _rightFields(rightFields),
    _outputFields(outputFields),
    _numLeftFields(numLeftFields),
    _numRightFields(numRightFields),
    _numPoints(numPoints),
    _dim1Tensor(dim1Tensor),
    _dim2Tensor(dim2Tensor),
    _sumInto(sumInto)
  {
    // Nothing to do                                                                                                                                          
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned int elementIndex) const {
    if (_sumInto) {
      for (int lbf = 0; lbf < _numLeftFields; lbf++) {
        for (int rbf = 0; rbf < _numRightFields; rbf++) {
          Scalar tmpVal(0);
          for (int qp = 0; qp < _numPoints; qp++) {
            for (int iTens1 = 0; iTens1 < _dim1Tensor; iTens1++) {
              for (int iTens2 = 0; iTens2 < _dim2Tensor; iTens2++) {
                tmpVal += _leftFields(elementIndex, lbf, qp, iTens1, iTens2)*_rightFields(elementIndex, rbf, qp, iTens1, iTens2);
	      } // D2-loop                                                                                                                                   
            } // D1-loop                                                                                                                                     
          } // P-loop                                                                                                                                        
          _outputFields(elementIndex, lbf, rbf) += tmpVal;
        } // R-loop                                                                                                                                          
      } // L-loop                                                                                                                                            
    }
    else {
      for (int lbf = 0; lbf < _numLeftFields; lbf++) {
        for (int rbf = 0; rbf < _numRightFields; rbf++) {
          Scalar tmpVal(0);
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
  }
};

void ArrayTools::contractFieldFieldTensor(ArrayOutFields &            outputFields,
                                          const ArrayInFieldsLeft &   leftFields,
                                          const ArrayInFieldsRight &  rightFields,
                                          const ECompEngine           compEngine,
                                          const bool                  sumInto) {

  // get sizes                                                                                                                                                
  int numCells        = leftFields.dimension(0);
  int numLeftFields   = leftFields.dimension(1);
  int numRightFields  = rightFields.dimension(1);
  int numPoints       = leftFields.dimension(2);
  int dim1Tensor      = leftFields.dimension(3);
  int dim2Tensor      = leftFields.dimension(4);

  switch(compEngine) {
  case COMP_CPP: {
    if (sumInto) {
      for (int cl = 0; cl < numCells; cl++) {
	for (int lbf = 0; lbf < numLeftFields; lbf++) {
	  for (int rbf = 0; rbf < numRightFields; rbf++) {
	    Scalar tmpVal(0);
	    for (int qp = 0; qp < numPoints; qp++) {
	      for (int iTens1 = 0; iTens1 < dim1Tensor; iTens1++) {
		for (int iTens2 = 0; iTens2 < dim2Tensor; iTens2++) {
		  tmpVal += leftFields(cl, lbf, qp, iTens1, iTens2)*rightFields(cl, rbf, qp, iTens1, iTens2);
		} // D2-loop                                                                                                                                
	      } // D1-loop                                                                                                                                  
	    } // P-loop                                                                                                                                     
	    outputFields(cl, lbf, rbf) += tmpVal;
	  } // R-loop                                                                                                                                       
	} // L-loop                                                                                                                                         
      } // C-loop                                                                                                                                           
    }
    else {
      for (int cl = 0; cl < numCells; cl++) {
	for (int lbf = 0; lbf < numLeftFields; lbf++) {
	  for (int rbf = 0; rbf < numRightFields; rbf++) {
	    Scalar tmpVal(0);
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
  }
    break;

  case COMP_KOKKOS: {

    typedef Kokkos::View<Scalar*****> input_view_t;
    typedef Kokkos::View<Scalar***> output_view_t;

    typedef typename input_view_t::HostMirror input_host_t;
    typedef typename output_view_t::HostMirror output_host_t;

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

    ContractFieldFieldTensorFunctor<Scalar, input_view_t, input_view_t, output_view_t>
      kokkosFunctor(kokkosLeft, kokkosRight, kokkosOut,
		    numLeftFields, numRightFields, numPoints,
		    dim1Tensor, dim2Tensor, sumInto);
    Kokkos::parallel_for(numCells, kokkosFunctor);

    Kokkos::fence();

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
} // end contractFieldFieldTensor
*/    
int main(int argc, char* argv[]) {

  // ===============================================================
  // ********************** < do kokkos> ***************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  int c=5, p=9, l=3, r=7, d1=13, d2=5;                                                                                                                    

  Intrepid::FieldContainer<double> in_c_l_p_d_d(c, l, p, d1, d2);                                                                                           
  Intrepid::FieldContainer<double> in_c_r_p_d_d(c, r, p, d1, d2);                                                                                            
  Intrepid::FieldContainer<double> out1_c_l_r(c, l, r);                                                                                                     
  Intrepid::FieldContainer<double> out2_c_l_r(c, l, r);                                                                                                      
                                                                                                                      
  
  // fill with random numbers                                                                                                                             
  for (int i=0; i<in_c_l_p_d_d.size(); i++) {                                                                                                             
    in_c_l_p_d_d[i] = Teuchos::ScalarTraits<double>::random();                                                                                            
  }                                                                                                                                                       
  for (int i=0; i<in_c_r_p_d_d.size(); i++) {                                                                                                             
    in_c_r_p_d_d[i] = Teuchos::ScalarTraits<double>::random();                                                                                            
  }                                                                                                                                                       
  /*                                                                                                                                                          
  art::contractFieldFieldTensor<double>(out1_c_l_r, in_c_l_p_d_d, in_c_r_p_d_d, COMP_CPP);                                                                
  art::contractFieldFieldTensor<double>(out2_c_l_r, in_c_l_p_d_d, in_c_r_p_d_d, COMP_BLAS);

  
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
