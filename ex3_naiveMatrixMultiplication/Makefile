# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35

# OS-specific build flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  # compiler
  CC              := /opt/local/bin/g++-mp-4.8
  # cuda
  CUDA_PATH       ?= /usr/local/cuda
  CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib/
  LD_FLAGS        += -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart -lcufft
  # tbb
  TBB_DIR         += /opt/tbb-4.2
  # kokkos
  KOKKOS_DIR      := /opt/kokkos
else
  # compiler
  CC              := g++
  # cuda
  CUDA_PATH       ?= /usr/local/cuda-6.5
  CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib64/
  ifeq ($(OS_SIZE),32)
    LD_FLAGS      += -L$(CUDA_LIB_PATH) -lcudart -lcufft
    CC_FLAGS      += -m32
  else
    LD_FLAGS      += -L$(CUDA_LIB_PATH) -lcudart -lcufft
    CC_FLAGS      += -m64
  endif
  # tbb
  TBB_DIR         += /clinic/2014/sandia14/tbb-4.3
  # kokkos
  KOKKOS_DIR      := /clinic/2014/sandia14/kokkos
  CC_FLAGS        += -DENABLE_KOKKOS
endif

# tbb
CC_INCLUDE        += -I$(TBB_DIR)/include
LD_FLAGS          += -L$(TBB_DIR)/lib -ltbb

# kokkos
CC_INCLUDE        += -I$(KOKKOS_DIR)/core/src -I$(KOKKOS_DIR)/containers/src -I$(KOKKOS_DIR)/algorithms/src -I$(KOKKOS_DIR)/linalg/src -DKOKKOS_DONT_INCLUDE_CORE_CONFIG_H  -DKOKKOS_HAVE_PTHREAD -DKOKKOS_HAVE_OPENMP
LD_FLAGS          += -L$(KOKKOS_DIR) -lkokkoscore

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32
else
      NVCCFLAGS := -m64
endif

CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

CC_FLAGS   += -std=c++11 -fopenmp

TARGETS = ex3_naiveMatrixMultiplication

all: $(TARGETS)

ex3_naiveMatrixMultiplication: ex3_naiveMatrixMultiplication.cc ex3_naiveMatrixMultiplication_cuda.o
	$(CC) $< -o $@ ex3_naiveMatrixMultiplication_cuda.o -O3 $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS) -Wall

ex3_naiveMatrixMultiplication_cuda.o: ex3_naiveMatrixMultiplication_cuda.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)
