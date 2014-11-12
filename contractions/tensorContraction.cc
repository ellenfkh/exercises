#include <vector>
#include <stdio.h>
#include <assert.h>

#include <Kokkos_Core.hpp>

template <class Scalar>
struct fourDTensorArray {
  std::vector<Scalar> _data;
  const unsigned int _numCells;
  const unsigned int _numFields;
  const unsigned int _numPoints;
  const unsigned int _dimVec;

  fourDTensorArray(const unsigned int cl, const unsigned int bf, const unsigned int qp, const unsigned int iVec)
  :_numCells(cl), _numFields(bf), _numPoints(qp), _dimVec(iVec), _data(cl*bf*qp*iVec) {
  }

  inline
  Scalar &
  operator()(const unsigned int cl, const unsigned int bf, const unsigned int qp, const unsigned int iVec) {
    return _data[cl * _numFields * _numPoints * _dimVec + bf * _numPoints * _dimVec  + qp * _dimVec + iVec];
  }

  inline
  Scalar
  operator()(const unsigned int cl, const unsigned int bf, const unsigned int qp, const unsigned int iVec) const {
    return _data[cl * _numFields * _numPoints * _dimVec + bf * _numPoints * _dimVec  + qp * _dimVec + iVec];
  }

  void
  fill(const Scalar d) {
    std::fill(_data.begin(), _data.end(), d);
  }
};

template <class Scalar>
struct threeDTensorArray {
  std::vector<Scalar> _data;
  const unsigned int _numCells;
  const unsigned int _numFields;
  const unsigned int _numPoints;

  threeDTensorArray(const unsigned int cl, const unsigned int bf, const unsigned int qp)
  :_numCells(cl), _numFields(bf), _numPoints(qp), _data(cl*bf*qp) {
  }

  inline
  Scalar &
  operator()(const unsigned int cl, const unsigned int bf, const unsigned int qp) {
    return _data[cl * _numFields * _numPoints + bf * _numPoints + qp];
  }

  inline
  Scalar
  operator()(const unsigned int cl, const unsigned int bf, const unsigned int qp) const {
    return _data[cl * _numFields * _numPoints + bf * _numPoints + qp];
  }

  void
  fill(const Scalar d) {
    std::fill(_data.begin(), _data.end(), d);
  }
};

enum ECompEngine {COMP_CPP, COMP_BLAS, COMP_KOKKOS};


template <class Scalar>
void contractFieldFieldVectorSerial(threeDTensorArray<Scalar> &         outputFields,
                                    const fourDTensorArray<Scalar> &    leftFields,
                                    const fourDTensorArray<Scalar> &    rightFields,
                                    const ECompEngine           compEngine,
                                    const bool                  sumInto) {

  const unsigned int numCells        = leftFields._numCells;
  const unsigned int numLeftFields   = leftFields._numFields;
  const unsigned int numRightFields  = rightFields._numFields;
  const unsigned int numPoints       = leftFields._numPoints;
  const unsigned int dimVec          = leftFields._dimVec;

  switch(compEngine) {
    case COMP_CPP: {
      if (sumInto) {
        for (unsigned int cl = 0; cl < numCells; cl++) {
          for (unsigned int lbf = 0; lbf < numLeftFields; lbf++) {
            for (unsigned int rbf = 0; rbf < numRightFields; rbf++) {
              Scalar tmpVal(0);
              for (unsigned int qp = 0; qp < numPoints; qp++) {
                for (unsigned int iVec = 0; iVec < dimVec; iVec++) {
                  tmpVal += leftFields(cl, lbf, qp, iVec)*rightFields(cl, rbf, qp, iVec);
                } //D-loop
              } // P-loop
              outputFields(cl, lbf, rbf) += tmpVal;
            } // R-loop
          } // L-loop
        } // C-loop
      }
      else {
        for (unsigned int cl = 0; cl < numCells; cl++) {
          for (unsigned int lbf = 0; lbf < numLeftFields; lbf++) {
            for (unsigned int rbf = 0; rbf < numRightFields; rbf++) {
              Scalar tmpVal(0);
              for (unsigned int qp = 0; qp < numPoints; qp++) {
                for (unsigned int iVec = 0; iVec < dimVec; iVec++) {
                  tmpVal += leftFields(cl, lbf, qp, iVec)*rightFields(cl, rbf, qp, iVec);
                } //D-loop
              } // P-loop
              outputFields(cl, lbf, rbf) = tmpVal;
            } // R-loop
          } // L-loop
        } // C-loop
      }
    }
    break;
  }
}


int main(int argc, const char* argv[]) {
  const unsigned int dummySize = 10;


  fourDTensorArray<double> leftInput(dummySize, dummySize, dummySize, dummySize);
  fourDTensorArray<double> rightInput(dummySize, dummySize, dummySize, dummySize);
  threeDTensorArray<double> output(dummySize, dummySize, dummySize);

  leftInput.fill(1);
  rightInput.fill(1);
  output.fill(1);

  for (unsigned int cl = 0; cl < dummySize; cl++) {
    for (unsigned int lbf = 0; lbf < dummySize; lbf++) {
      for (unsigned int rbf = 0; rbf < dummySize; rbf++) {
        if (output(cl, lbf, rbf) != 1) {
          printf("Output at %d, %d, %d should be 1; instead is %f\n", cl, lbf, rbf, output(cl, lbf, rbf));
          return 1;
        }
      }
    }
  }

  contractFieldFieldVectorSerial<double>(output, leftInput, rightInput, COMP_CPP, false);

  for (unsigned int cl = 0; cl < dummySize; cl++) {
    for (unsigned int lbf = 0; lbf < dummySize; lbf++) {
      for (unsigned int rbf = 0; rbf < dummySize; rbf++) {
        if (output(cl, lbf, rbf) != 100) {
          printf("Output at %d, %d, %d should be 100; instead is %f\n", cl, lbf, rbf, output(cl, lbf, rbf));
          return 1;
        }
      }
    }
  }

  printf("passed serial, sumInto = false\n");

  output.fill(1);

  contractFieldFieldVectorSerial<double>(output, leftInput, rightInput, COMP_CPP, true);

  for (unsigned int cl = 0; cl < dummySize; cl++) {
    for (unsigned int lbf = 0; lbf < dummySize; lbf++) {
      for (unsigned int rbf = 0; rbf < dummySize; rbf++) {
        if (output(cl, lbf, rbf) != 101) {
          printf("Output at %d, %d, %d should be 101; instead is %f\n", cl, lbf, rbf, output(cl, lbf, rbf));
          return 1;
        }
      }
    }
  }
  printf("passed serial, sumInto = true\n");

  return 0;
}
