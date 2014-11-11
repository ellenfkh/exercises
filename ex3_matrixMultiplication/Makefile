# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35

# OS-specific build flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  # compiler
  CC              := /opt/local/bin/g++-mp-4.8
  NVCCSCC         := /opt/local/bin/g++-mp-4.7
  # cuda
  CUDA_PATH       ?= /usr/local/cuda
  CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib/
  LD_FLAGS        += -Xlinker -L$(CUDA_LIB_PATH) -lcudart -lcufft -lcublas
  # tbb
  TBB_DIR         += /opt/tbb-4.2
  LD_FLAGS        += -L$(TBB_DIR)/lib-gcc48 -ltbb
  # eigen
  EIGEN_DIR       := /opt/eigen
else
  # compiler
  CC              := g++
  NVCCSCC         := $(CC)
  # cuda
  CUDA_PATH       ?= /usr/local/cuda-6.5
  CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib64/
  ifeq ($(OS_SIZE),32)
    CC_FLAGS      += -m32
  else
    CC_FLAGS      += -m64
  endif
  LD_FLAGS      += -L$(CUDA_LIB_PATH) -lcudart -lcufft -lcublas
  # tbb
  TBB_DIR         += /clinic/2014/sandia14/tbb-4.3
  LD_FLAGS        += -L$(TBB_DIR)/lib -ltbb
  # kokkos
  KOKKOS_DIR      := /clinic/2014/sandia14/trilinos-install/
  # eigen
  EIGEN_DIR       := /clinic/2014/sandia14/eigen
  # avx vectorization support
  CC_FLAGS        += -mavx
  # enable kokkos
  NVCC_LD_FLAGS   += -L$(KOKKOS_DIR)/lib -lkokkoscore

  # kokkos
  CC_INCLUDE      += -I$(KOKKOS_DIR)/include
  NVCC_INCLUDE    += -I$(KOKKOS_DIR)/include
  LD_FLAGS        += -L$(KOKKOS_DIR)/lib -lkokkoscore

  # add hwloc
  LD_FLAGS        += -lhwloc
  NVCC_LD_FLAGS   += -lhwloc
endif

# eigen
CC_INCLUDE        += -I$(EIGEN_DIR)

# tbb
CC_INCLUDE        += -I$(TBB_DIR)/include

# I'm adding the linking to trilinos here this may be wrong
CC_INCLUDE += -I/clinic/2014/sandia14/trilinos-11.10.2-Source/packages/intrepid/src/Shared
LD_FLAGS += -L/clinic/2014/sandia14/trilinos-install/lib/ -lteuchoscore
# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCC_FLAGS += -m32
else
      NVCC_FLAGS += -m64
endif


CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

CC_FLAGS        += -std=c++11 -fopenmp
NVCC_LD_FLAGS   += -lgomp

TARGETS = MatrixMultiplication MatrixMultiplication_kokkos

all: $(TARGETS)

MatrixMultiplication: MatrixMultiplication.cu
	$(NVCC) -ccbin=$(NVCCSCC) $(NVCC_FLAGS) -O3 $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) $(NVCC_INCLUDE) -o $@ $< -L/clinic/2014/sandia14/trilinos-install/lib -lkokkoscore -lhwloc -lgomp -lcudart -lcublas -lteuchoscore


clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)
