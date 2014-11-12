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
  :_data(cl*bf*qp*iVec), _numCells(cl), _numFields(bf), _numPoints(qp), _dimVec(iVec) {
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
  :_data(cl*bf*qp), _numCells(cl), _numFields(bf), _numPoints(qp) {
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

template <class Scalar, class input_view_type, class output_view_type>
struct ContractFieldFieldVectorKokkosFunctor {
  input_view_type _leftFields;
  input_view_type _rightFields;

  output_view_type _outputFields;

  int _numLeftFields;
  int _numRightFields;
  int _numPoints;
  int _dimVec;
  bool _sumInto;

  ContractFieldFieldVectorKokkosFunctor(
    int numLeftFields,
    int numRightFields,
    int numPoints,
    int dimVec,
    bool sumInto,
    input_view_type leftFields,
    input_view_type rightFields,
    output_view_type outputFields
    )
    :_leftFields(leftFields), _rightFields(rightFields),
    _outputFields(outputFields),
    _numLeftFields(numLeftFields), _numRightFields(numRightFields),
    _numPoints(numPoints), _dimVec(dimVec), _sumInto(sumInto)
  { }

  KOKKOS_INLINE_FUNCTION
    void operator() (const unsigned int elementIndex) const {
      if (_sumInto) {
        for (int lbf = 0; lbf < _numLeftFields; lbf++) {
          for (int rbf = 0; rbf < _numRightFields; rbf++) {
            Scalar tmpVal(0);
            for (int qp = 0; qp < _numPoints; qp++) {
              for (int iVec = 0; iVec < _dimVec; iVec++) {
                tmpVal += _leftFields(elementIndex, lbf, qp, iVec)*_rightFields(elementIndex, rbf, qp, iVec);
              } //D-loop
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
              for (int iVec = 0; iVec < _dimVec; iVec++) {
                tmpVal += _leftFields(elementIndex, lbf, qp, iVec)*_rightFields(elementIndex, rbf, qp, iVec);
              } //D-loop
            } // P-loop
            _outputFields(elementIndex, lbf, rbf) = tmpVal;
          } // R-loop
        } // L-loop
      }
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
#if 0
    case COMP_KOKKOS: {
      typedef Kokkos::View<Scalar****> input_view_t;
      typedef Kokkos::View<Scalar***> output_view_t;

      typedef typename output_view_t::HostMirror output_host_t;
      typedef typename input_view_t::HostMirror input_host_t;

      Kokkos::initialize();

      input_view_t kokkosLeft("left_input", numCells, numLeftFields, numPoints, dimVec);
      input_view_t kokkosRight("right_input", numCells, numRightFields, numPoints, dimVec);
      output_view_t kokkosOutput("output", numCells, numLeftFields, numRightFields);

      input_host_t hostLeft = Kokkos::create_mirror_view(kokkosLeft);
      input_host_t hostRight = Kokkos::create_mirror_view(kokkosRight);
      output_host_t hostOutput = Kokkos::create_mirror_view(kokkosOutput);

      for (int cl = 0; cl < numCells; cl++) {
        for (int qp = 0; qp < numPoints; qp++) {
          for (int iVec = 0; iVec < dimVec; iVec++) {
            for (int lbf = 0; lbf < numLeftFields; lbf++) {
              hostLeft(cl, lbf, qp, iVec) = leftFields(cl, lbf, qp, iVec);
            } // L-loop
            for (int rbf = 0; rbf < numRightFields; rbf++) {
              hostRight(cl, rbf, qp, iVec) = rightFields(cl, rbf, qp, iVec);
            } // R-loop
          } // D-loop
        } // P-loop
        for (int lbf = 0; lbf < numLeftFields; lbf++) {
          for (int rbf = 0; rbf < numRightFields; rbf++) {
            hostOutput(cl, lbf, rbf) = outputFields(cl, lbf, rbf);
          } // R-loop
        } // L-loop
      } // C-loop

      Kokkos::deep_copy(kokkosLeft, hostLeft);
      Kokkos::deep_copy(kokkosRight, hostRight);
      Kokkos::deep_copy(kokkosOutput, hostOutput);

      ContractFieldFieldVectorKokkosFunctor<Scalar, input_view_t, output_view_t> kokkosFunctor(numLeftFields,
          numRightFields, numPoints, dimVec, sumInto, kokkosLeft, kokkosRight,
          kokkosOutput);

      Kokkos::parallel_for(numCells, kokkosFunctor);

      Kokkos::fence();
      Kokkos::deep_copy(hostOutput, kokkosOutput);

      for (int cl = 0; cl < numCells; cl++) {
        for (int lbf = 0; lbf < numLeftFields; lbf++) {
          for (int rbf = 0; rbf < numRightFields; rbf++) {
            outputFields(cl, lbf, rbf) = hostOutput(cl, lbf, rbf);
          } // R-loop
        } // L-loop
      } //C-loop

      Kokkos::finalize();
    }
    break;
#endif
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
