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

template<class Scalar, class ArrayOutFields, class ArrayInData, class ArrayInFields>
void scalarMultiplyDataField(ArrayOutFields &     outputFields,
                                         const ArrayInData &  inputData,
                                         ArrayInFields &      inputFields,
                                         const bool           reciprocal) {

#ifdef HAVE_INTREPID_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION( (inputData.rank() != 2), std::invalid_argument,
                      ">>> ERROR (ArrayTools::scalarMultiplyDataField): Input data container must have rank 2.");
  if (outputFields.rank() <= inputFields.rank()) {
    TEUCHOS_TEST_FOR_EXCEPTION( ( (inputFields.rank() < 3) || (inputFields.rank() > 5) ), std::invalid_argument,
                        ">>> ERROR (ArrayTools::scalarMultiplyDataField): Input fields container must have rank 3, 4, or 5.");
    TEUCHOS_TEST_FOR_EXCEPTION( (outputFields.rank() != inputFields.rank()), std::invalid_argument,
                        ">>> ERROR (ArrayTools::scalarMultiplyDataField): Input and output fields containers must have the same rank.");
    TEUCHOS_TEST_FOR_EXCEPTION( (inputFields.dimension(0) != inputData.dimension(0) ), std::invalid_argument,
                        ">>> ERROR (ArrayTools::scalarMultiplyDataField): Zeroth dimensions (number of integration domains) of the fields and data input containers must agree!");
    TEUCHOS_TEST_FOR_EXCEPTION( ( (inputFields.dimension(2) != inputData.dimension(1)) && (inputData.dimension(1) != 1) ), std::invalid_argument,
                        ">>> ERROR (ArrayTools::scalarMultiplyDataField): Second dimension of the fields input container and first dimension of data input container (number of integration points) must agree or first data dimension must be 1!");
    for (int i=0; i<inputFields.rank(); i++) {
      std::string errmsg  = ">>> ERROR (ArrayTools::scalarMultiplyDataField): Dimension ";
      errmsg += (char)(48+i);
      errmsg += " of the input and output fields containers must agree!";
      TEUCHOS_TEST_FOR_EXCEPTION( (inputFields.dimension(i) != outputFields.dimension(i)), std::invalid_argument, errmsg );
    }
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION( ( (inputFields.rank() < 2) || (inputFields.rank() > 4) ), std::invalid_argument,
                        ">>> ERROR (ArrayTools::scalarMultiplyDataField): Input fields container must have rank 2, 3, or 4.");
    TEUCHOS_TEST_FOR_EXCEPTION( (outputFields.rank() != inputFields.rank()+1), std::invalid_argument,
                        ">>> ERROR (ArrayTools::scalarMultiplyDataField): The rank of the input fields container must be one less than the rank of the output fields container.");
    TEUCHOS_TEST_FOR_EXCEPTION( ( (inputFields.dimension(1) != inputData.dimension(1)) && (inputData.dimension(1) != 1) ), std::invalid_argument,
                        ">>> ERROR (ArrayTools::scalarMultiplyDataField): First dimensions of fields input container and data input container (number of integration points) must agree or first data dimension must be 1!");
    TEUCHOS_TEST_FOR_EXCEPTION( ( inputData.dimension(0) != outputFields.dimension(0) ), std::invalid_argument,
                        ">>> ERROR (ArrayTools::scalarMultiplyDataField): Zeroth dimensions of fields output container and data input containers (number of integration domains) must agree!");
    for (int i=0; i<inputFields.rank(); i++) {
      std::string errmsg  = ">>> ERROR (ArrayTools::scalarMultiplyDataField): Dimensions ";
      errmsg += (char)(48+i);
      errmsg += " and ";
      errmsg += (char)(48+i+1);
      errmsg += " of the input and output fields containers must agree!";
      TEUCHOS_TEST_FOR_EXCEPTION( (inputFields.dimension(i) != outputFields.dimension(i+1)), std::invalid_argument, errmsg );
    }
  }
#endif

  // get sizes
  int invalRank      = inputFields.rank();
  int outvalRank     = outputFields.rank();
  int numCells       = outputFields.dimension(0);
  int numFields      = outputFields.dimension(1);
  int numPoints      = outputFields.dimension(2);
  int numDataPoints  = inputData.dimension(1);
  int dim1Tens       = 0;
  int dim2Tens       = 0;
  if (outvalRank > 3) {
    dim1Tens = outputFields.dimension(3);
    if (outvalRank > 4) {
      dim2Tens = outputFields.dimension(4);
    }
  }

  if (outvalRank == invalRank) {

    if (numDataPoints != 1) { // nonconstant data

      switch(invalRank) {
        case 3: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  outputFields(cl, bf, pt) = inputFields(cl, bf, pt)/inputData(cl, pt);
                } // P-loop
              } // F-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  outputFields(cl, bf, pt) = inputFields(cl, bf, pt)*inputData(cl, pt);
                } // P-loop
              } // F-loop
            } // C-loop
          }
        }// case 3
        break;

        case 4: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iVec = 0; iVec < dim1Tens; iVec++) {
                    outputFields(cl, bf, pt, iVec) = inputFields(cl, bf, pt, iVec)/inputData(cl, pt);
                  } // D1-loop
                } // P-loop
              } // F-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iVec = 0; iVec < dim1Tens; iVec++) {
                    outputFields(cl, bf, pt, iVec) = inputFields(cl, bf, pt, iVec)*inputData(cl, pt);
                  } // D1-loop
                } // P-loop
              } // F-loop
            } // C-loop
          }
        }// case 4
        break;

        case 5: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iTens1 = 0; iTens1 < dim1Tens; iTens1++) {
                    for( int iTens2 = 0; iTens2 < dim2Tens; iTens2++) {
                      outputFields(cl, bf, pt, iTens1, iTens2) = inputFields(cl, bf, pt, iTens1, iTens2)/inputData(cl, pt);
                    } // D2-loop
                  } // D1-loop
                } // F-loop
              } // P-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iTens1 = 0; iTens1 < dim1Tens; iTens1++) {
                    for( int iTens2 = 0; iTens2 < dim2Tens; iTens2++) {
                      outputFields(cl, bf, pt, iTens1, iTens2) = inputFields(cl, bf, pt, iTens1, iTens2)*inputData(cl, pt);
                    } // D2-loop
                  } // D1-loop
                } // F-loop
              } // P-loop
            } // C-loop
          }
        }// case 5
        break;

        default:
	    ;
	}// invalRank

    }
    else { //constant data

      switch(invalRank) {
        case 3: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  outputFields(cl, bf, pt) = inputFields(cl, bf, pt)/inputData(cl, 0);
                } // P-loop
              } // F-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  outputFields(cl, bf, pt) = inputFields(cl, bf, pt)*inputData(cl, 0);
                } // P-loop
              } // F-loop
            } // C-loop
          }
        }// case 3
        break;

        case 4: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iVec = 0; iVec < dim1Tens; iVec++) {
                    outputFields(cl, bf, pt, iVec) = inputFields(cl, bf, pt, iVec)/inputData(cl, 0);
                  } // D1-loop
                } // P-loop
              } // F-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iVec = 0; iVec < dim1Tens; iVec++) {
                    outputFields(cl, bf, pt, iVec) = inputFields(cl, bf, pt, iVec)*inputData(cl, 0);
                  } // D1-loop
                } // P-loop
              } // F-loop
            } // C-loop
          }
        }// case 4
        break;

        case 5: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iTens1 = 0; iTens1 < dim1Tens; iTens1++) {
                    for( int iTens2 = 0; iTens2 < dim2Tens; iTens2++) {
                      outputFields(cl, bf, pt, iTens1, iTens2) = inputFields(cl, bf, pt, iTens1, iTens2)/inputData(cl, 0);
                    } // D2-loop
                  } // D1-loop
                } // F-loop
              } // P-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iTens1 = 0; iTens1 < dim1Tens; iTens1++) {
                    for( int iTens2 = 0; iTens2 < dim2Tens; iTens2++) {
                      outputFields(cl, bf, pt, iTens1, iTens2) = inputFields(cl, bf, pt, iTens1, iTens2)*inputData(cl, 0);
                    } // D2-loop
                  } // D1-loop
                } // F-loop
              } // P-loop
            } // C-loop
          }
        }// case 5
        break;

        default:
	    ; 
      } // invalRank
    } // numDataPoints

  }
  else {

    if (numDataPoints != 1) { // nonconstant data

      switch(invalRank) {
        case 2: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  outputFields(cl, bf, pt) = inputFields(bf, pt)/inputData(cl, pt);
                } // P-loop
              } // F-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  outputFields(cl, bf, pt) = inputFields(bf, pt)*inputData(cl, pt);
                } // P-loop
              } // F-loop
            } // C-loop
          }
        }// case 2
        break;

        case 3: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iVec = 0; iVec < dim1Tens; iVec++) {
                    outputFields(cl, bf, pt, iVec) = inputFields(bf, pt, iVec)/inputData(cl, pt);
                  } // D1-loop
                } // P-loop
              } // F-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iVec = 0; iVec < dim1Tens; iVec++) {
                    outputFields(cl, bf, pt, iVec) = inputFields(bf, pt, iVec)*inputData(cl, pt);
                  } // D1-loop
                } // P-loop
              } // F-loop
            } // C-loop
          }
        }// case 3
        break;

        case 4: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iTens1 = 0; iTens1 < dim1Tens; iTens1++) {
                    for( int iTens2 = 0; iTens2 < dim2Tens; iTens2++) {
                      outputFields(cl, bf, pt, iTens1, iTens2) = inputFields(bf, pt, iTens1, iTens2)/inputData(cl, pt);
                    } // D2-loop
                  } // D1-loop
                } // F-loop
              } // P-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iTens1 = 0; iTens1 < dim1Tens; iTens1++) {
                    for( int iTens2 = 0; iTens2 < dim2Tens; iTens2++) {
                      outputFields(cl, bf, pt, iTens1, iTens2) = inputFields(bf, pt, iTens1, iTens2)*inputData(cl, pt);
                    } // D2-loop
                  } // D1-loop
                } // F-loop
              } // P-loop
            } // C-loop
          }
        }// case 4
        break;

        default:
	    ;
	}// invalRank

    }
    else { //constant data

      switch(invalRank) {
        case 2: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  outputFields(cl, bf, pt) = inputFields(bf, pt)/inputData(cl, 0);
                } // P-loop
              } // F-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  outputFields(cl, bf, pt) = inputFields(bf, pt)*inputData(cl, 0);
                } // P-loop
              } // F-loop
            } // C-loop
          }
        }// case 2
        break;

        case 3: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iVec = 0; iVec < dim1Tens; iVec++) {
                    outputFields(cl, bf, pt, iVec) = inputFields(bf, pt, iVec)/inputData(cl, 0);
                  } // D1-loop
                } // P-loop
              } // F-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iVec = 0; iVec < dim1Tens; iVec++) {
                    outputFields(cl, bf, pt, iVec) = inputFields(bf, pt, iVec)*inputData(cl, 0);
                  } // D1-loop
                } // P-loop
              } // F-loop
            } // C-loop
          }
        }// case 3
        break;

        case 4: {
          if (reciprocal) {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iTens1 = 0; iTens1 < dim1Tens; iTens1++) {
                    for( int iTens2 = 0; iTens2 < dim2Tens; iTens2++) {
                      outputFields(cl, bf, pt, iTens1, iTens2) = inputFields(bf, pt, iTens1, iTens2)/inputData(cl, 0);
                    } // D2-loop
                  } // D1-loop
                } // F-loop
              } // P-loop
            } // C-loop
          }
          else {
            for(int cl = 0; cl < numCells; cl++) {
              for(int bf = 0; bf < numFields; bf++) {
                for(int pt = 0; pt < numPoints; pt++) {
                  for( int iTens1 = 0; iTens1 < dim1Tens; iTens1++) {
                    for( int iTens2 = 0; iTens2 < dim2Tens; iTens2++) {
                      outputFields(cl, bf, pt, iTens1, iTens2) = inputFields(bf, pt, iTens1, iTens2)*inputData(cl, 0);
                    } // D2-loop
                  } // D1-loop
                } // F-loop
              } // P-loop
            } // C-loop
          }
        }// case 4
        break;

        default:
	    ;
	} // invalRank
    } // numDataPoints

  } // end if (outvalRank = invalRank)

}


int main(int argc, char* argv[]) {
    int c = 1000, p = 10, l = 10, r = 10, t1 = 10, t2 = 10;

    FieldContainer<double> in_c_l_p_t1_t2(c, l, p, t1, t2);
    FieldContainer<double> in_c_r_p_t1_t2(c, r, p, t1, t2);
    FieldContainer<double> out1_c_l_r(c, l, r);
    FieldContainer<double> out2_c_l_r(c, l, r);


    Kokkos::initialize();

    Kokkos::finalize();

    return 0;

}








