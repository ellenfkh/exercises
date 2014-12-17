#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

#include <Kokkos_Core.hpp>

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

void serial(double* inputFields, double* inputData, double* output,
            int c, int p, int f) {

  for (int cl = 0; cl < c; cl++) {
    for (int lbf = 0; lbf < f; lbf++) {
      double tmpVal = 0;
      for (int qp = 0; qp < p; qp++) {
        tmpVal += inputFields[cl * f * p + lbf * p + qp]*inputData[cl * p + qp];
      } // P-loop
      output[cl * f + lbf] += tmpVal;
    } // F-loop
  } // C-loop
}

int main(int argc, char* argv[]) {
	int c=1000000, p=10, f=10;
  int repeats = 10;

  double* inputFields = new double[c * p * f];
  double* inputData = new double[c * p];
  double* serialOutput = new double[c * f];

  timespec tic;
  timespec toc;

  for (int cl = 0; cl < c; cl++) {
    for (int lbf = 0; lbf < f; lbf++) {
      for (int qp = 0; qp < p; qp++) {
        inputFields[cl * f * p + lbf * p + qp] = (double)std::rand();
      } // P-loop
    } // F-loop
  } // C-loop

  for (int cl = 0; cl < c; cl++) {
    for (int qp = 0; qp < p; qp++) {
      inputData[cl * p + qp] = (double)std::rand();
    } // P-loop
  } // C-loop

  for (int cl = 0; cl < c; cl++) {
    for (int lbf = 0; lbf < f; lbf++) {
      serialOutput[cl * f + lbf] = 0;
    } // F-loop
  } // C-loop

  clock_gettime(CLOCK_MONOTONIC, &tic);
  for (int i = 0; i < repeats; i++) {
    serial(inputFields, inputData, serialOutput, c, p, f);
  }
  clock_gettime(CLOCK_MONOTONIC, &toc);
  const double elapsedTime_serial = getElapsedTime(tic, toc);

	std::cout << "cache friendly serial time: " << elapsedTime_serial << std::endl;

}
