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

typedef Kokkos::View<double *****> input_view_t;
typedef Kokkos::View<double ***> output_view_t;

typedef input_view_t::HostMirror input_host_t;
typedef output_view_t::HostMirror output_host_t;

template<class LeftViewType, class RightViewType, class OutputViewType>
struct ContractFieldFieldTensorFunctor {
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


void contractFieldFieldTensor(FieldContainer<double> & outputFields,
			      const FieldContainer<double> &   leftFields,
			      const FieldContainer<double> &  rightFields,
                              const int compEngine) {

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
    
    hostRight(1,1,1,1,1) = 5;
    
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
    /*
    Kokkos::deep_copy(kokkosLeft, hostLeft);
    Kokkos::deep_copy(kokkosRight, hostRight);
    Kokkos::deep_copy(kokkosOut, hostOut);
    
    ContractFieldFieldTensorFunctor<input_view_t, input_view_t, output_view_t>
      kokkosFunctor(kokkosLeft, kokkosRight, kokkosOut,
		    numLeftFields, numRightFields, numPoints,
		    dim1Tensor, dim2Tensor);
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
    */
    Kokkos::finalize();
    
  }
  break;


  }
}

int main(int argc, char* argv[]) {

  // ===============================================================
  // ********************** < do kokkos> ***************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
  int c=5, p=9, l=3, r=7, d1=13, d2=5;

  FieldContainer<double> in_c_l_p_d_d(c, l, p, d1, d2);
  FieldContainer<double> in_c_r_p_d_d(c, r, p, d1, d2);
  FieldContainer<double> out1_c_l_r(c, l, r);
  FieldContainer<double> out2_c_l_r(c, l, r);
  double zero = Intrepid::INTREPID_TOL*10000.0;

  // fill with random numbers
  for (int i=0; i<in_c_l_p_d_d.size(); i++) {
    in_c_l_p_d_d[i] = Teuchos::ScalarTraits<double>::random();
  }
  for (int i=0; i<in_c_r_p_d_d.size(); i++) {
    in_c_r_p_d_d[i] = Teuchos::ScalarTraits<double>::random();
  }



 contractFieldFieldTensor(out1_c_l_r, in_c_l_p_d_d, in_c_r_p_d_d, 0);
 contractFieldFieldTensor(out2_c_l_r, in_c_l_p_d_d, in_c_r_p_d_d, 2);
 

  /* I got rid of the typedefs - for now
   * 0 -> manual computation
   * 1 -> blas
   * 2 -> kokkos
   */
   rst::subtract(&out1_c_l_r[0], &out2_c_l_r[0], out2_c_l_r.size());
   if (rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) > zero) {
      std::cout << "\n\nINCORRECT contractFieldFieldTensor (1): check COMP_CPP vs. COMP_KOKKOS; "
		<< " diff-1norm = " << rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) << "\n\n";
   }
   else {
     std::cout << "Cpp and Kokkos get same result" << std::endl;
   }

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
