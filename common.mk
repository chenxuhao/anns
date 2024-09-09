DEBUG ?= 0
BIN = ../../bin/
CC := gcc
CXX := g++
ICC := $(ICC_HOME)/icc
ICPC := $(ICC_HOME)/icpc
MPICC := mpicc
MPICXX := mpicxx
NVCC := nvcc

UNAME_P := $(shell uname -p)
ifndef GPU_ARCH
GPU_ARCH = 61
endif
CUDA_ARCH := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)
CXXFLAGS  := -Wall -fopenmp -std=c++20
#CXXFLAGS  += -ftree-vectorizer-verbose=6
CXXFLAGS += -march=native
ICPCFLAGS := -O3 -Wall -qopenmp -std=c++20
NVFLAGS := $(CUDA_ARCH)
NVFLAGS += -Xptxas -v -std=c++20
NVFLAGS += -DUSE_GPU
NVFLAGS += --expt-relaxed-constexpr
NVFLAGS += --extended-lambda
NVFLAGS += -DRAFT_SYSTEM_LITTLE_ENDIAN=1
NVLIBS = -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib64/stubs -lcuda -lcudart -lcurand -lcublas -lcusolver
MPI_LIBS = -L$(MPI_HOME)/lib -lmpi
NVSHMEM_LIBS = -L$(NVSHMEM_HOME)/lib -lnvshmem -lnvToolsExt -lnvidia-ml -ldl -lrt
LIBS := -lgomp
MKL_LIBS := -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -I${MKL_DIR}/include -L${MKL_DIR}/lib/intel64
OPENBLAS_LIBS := -L$(OPENBLAS_HOME)/lib -lopenblas
RAFT_LIBS := -L$(CONDA_HOME)/lib -lraft
RAFT_INCS := -I$(CONDA_HOME)/include -I$(RAFT_HOME)/include
PARLAY_INCS := ../ParlayANN/parlaylib
INCLUDES := -I./include -I$(ANN_HOME)/include 
#INCLUDES += -I$(ANN_HOME)/include/efanna2e $(RAFT_INCS)

ifeq ($(DEBUG), 1)
	CXXFLAGS += -g -O0
	NVFLAGS += -G
else
	CXXFLAGS += -O3 
	NVFLAGS += -O3 -w 
endif

# Vectorization options (from 6.106)
ifeq ($(ASSEMBLE),1)                                                                                                                                                     
  CXXFLAGS += -S
endif

ifeq ($(AVX2),1)
  CXXFLAGS += -mavx2
else
  #CXXFLAGS += -mno-avx
endif

#INCLUDES += -I$(OPENBLAS_HOME)/include

#INCLUDES += -I$(FAISS_HOME)
FAISS_LIBS = -L$(FAISS_HOME)/lib -lfaiss $(OPENBLAS_LIBS) -lgomp

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

