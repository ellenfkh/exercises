#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>
#include <string>
#include <chrono>
#include <algorithm>

#include <Kokkos_Core.hpp>

using std::string;
using std::vector;
using std::array;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

template <class Scalar>
struct 4DTensorArray {
  vector<Scalar> _data;
  const unsigned int _numCells;
  const unsigned int _numFields;
  const unsigned int _numPoints;
  const unsigned int _dimVec;

  4DTensorArray(const unsigned int cl, const unsigned int bf, const unsigned int qp, const unsigned int iVec)
  :_numCells(cl), _numFields(bf), _numPoints(qp), _dimVec(iVec), _data(cl*bf*qp*iVec) {
  }

  inline
  double &
  operator()(const unsigned int cl, const unsigned int bf, const unsigned int qp, const unsigned int iVec) {
    return _data[cl *_numCells + bf * _numFields + qp * numPoints + iVec];
  }

  inline
  double
  operator()(const unsigned int cl, const unsigned int bf, const unsigned int qp, const unsigned int iVec) {
    return _data[cl *_numCells + bf * _numFields + qp * numPoints + iVec];
  }

  void
  fill(const Scalar d) {
    std::fill(_data.begin(), _data.end(), 0);
  }
}

template <class Scalar>
struct 3DTensorArray {
  vector<Scalar> _data;
  const unsigned int _numCells;
  const unsigned int _numFields;
  const unsigned int _numPoints;

  3DTensorArray(const unsigned int cl, const unsigned int bf, const unsigned int qp)
  :_numCells(cl), _numFields(bf), _numPoints(qp), _data(cl*bf*qp) {
  }

  inline
  double &
  operator()(const unsigned int cl, const unsigned int bf, const unsigned int qp) {
    return _data[cl *_numCells + bf * _numFields + qp];
  }

  inline
  double
  operator()(const unsigned int cl, const unsigned int bf, const unsigned int qp) {
    return _data[cl *_numCells + bf * _numFields + qp];
  }

  void
  fill(const Scalar d) {
    std::fill(_data.begin(), _data.end(), 1);
  }
}

enum ECompEngine {COMP_CPP, COMP_BLAS, COMP_KOKKOS};


template <class Scalar>
void contractFieldFieldVectorSerial(3DTensorArray &             outputFields,
                                    const 4DTensorArray &       leftFields,
                                    const 4DTensorArray &       rightFields,
                                    const ECompEngine           compEngine,
                                    const bool                  sumInto) {

  int numCells        = leftFields._numCells;
  int numLeftFields   = leftFields._numFields;
  int numRightFields  = rightFields._numFields;
  int numPoints       = leftFields._numPoints;
  int dimVec          = leftFields._dimVec;

  switch(compEngine) {
    case COMP_CPP: {
      if (sumInto) {
        for (int cl = 0; cl < numCells; cl++) {
          for (int lbf = 0; lbf < numLeftFields; lbf++) {
            for (int rbf = 0; rbf < numRightFields; rbf++) {
              Scalar tmpVal(0);
              for (int qp = 0; qp < numPoints; qp++) {
                for (int iVec = 0; iVec < dimVec; iVec++) {
                  tmpVal += leftFields(cl, lbf, qp, iVec)*rightFields(cl, rbf, qp, iVec);
                } //D-loop
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
                for (int iVec = 0; iVec < dimVec; iVec++) {
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


int main(int argc, char* argv[]) {
  const unsigned int dummySize = 10;


  4DTensorArray<double> leftInput(dummySize, dummySize, dummySize, dummySize);
  4DTensorArray<double> rightInput(dummySize, dummySize, dummySize, dummySize);
  3DTensorArray<double> output(dummySize, dummySize, dummySize);

  leftInput.fill(1);
  rightInput.fill(1);
  output.fill(1);

  contractFieldFieldVectorSerial<double>(output, leftInput, rightInput, COMP_CPP, false);

  for (int cl = 0; cl < dummySize; cl++) {
    for (int lbf = 0; lbf < dummySize; lbf++) {
      for (int rbf = 0; rbf < dummySize; rbf++) {
        assert(output(cl, lbf, rbf) == 100);
      }
    }
  }

  output.fill(1);

  contractFieldFieldVectorSerial<double>(output, leftInput, rightInput, COMP_CPP, true);

  for (int cl = 0; cl < dummySize; cl++) {
    for (int lbf = 0; lbf < dummySize; lbf++) {
      for (int rbf = 0; rbf < dummySize; rbf++) {
        assert(output(cl, lbf, rbf) == 101);
      }
    }
  }

  return 0;
}
