# CUDA code generation flags
GENCODE_FLAGS   := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35

# OS-specific build flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  CUDA_PATH       ?= /usr/local/cuda
  CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib/
  LD_FLAGS        := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart -lcufft
  CC              := /opt/local/bin/g++-mp-4.8
  CC_INCLUDE      += -I/opt/tbb-4.2/include
  LD_FLAGS        += -L/opt/tbb-4.2/lib -ltbb
else
  CUDA_PATH       ?= /usr/local/cuda-6.5
  CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib64/
  CC              := g++
  CC_INCLUDE      += -I/clinic/2014/sandia14/tbb-4.3/include
  LD_FLAGS        += -L/clinic/2014/sandia14/tbb-4.3/lib64 -ltbb
  ifeq ($(OS_SIZE),32)
    LD_FLAGS      += -L$(CUDA_LIB_PATH) -lcudart -lcufft
    CC_FLAGS      += -m32
  else
    LD_FLAGS      += -L$(CUDA_LIB_PATH) -lcudart -lcufft
    CC_FLAGS      += -m64
  endif
endif

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

TARGETS = ex0_scalarIntegrator

all: $(TARGETS)

ex0_scalarIntegrator: ex0_scalarIntegrator.cc ex0_scalarIntegrator_cuda.o
	$(CC) $< -o $@ ex0_scalarIntegrator_cuda.o -O3 $(CC_INCLUDE) $(LD_FLAGS) $(CC_FLAGS) -Wall

ex0_scalarIntegrator_cuda.o: ex0_scalarIntegrator_cuda.cu
	$(NVCC) $(NVCCFLAGS) -O3 $(GENCODE_FLAGS) -I$(CUDA_INC_PATH) -o $@ -c $<

clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)
