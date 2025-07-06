include common.mk

OBJS=tester.o
INCS=include/pqueue.hpp
CUINCS=include/pqueue.cuh
BIN=./bin/

all: brute_force_cpu quantized_search_cpu ivf_flat_cpu parlayann_cpu
gpu: brute_force_gpu ivf_flat_gpu
faiss: faiss_test faiss_ivf_cpu faiss_hnsw

faiss_test:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -I$(CONDA_PREFIX)/include -L$(CONDA_PREFIX)/lib faiss_test.cpp -o $@ -lfaiss
	mv $@ $(BIN)

faiss_ivf_flat:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -I$(CONDA_PREFIX)/include -L$(CONDA_PREFIX)/lib faiss_ivf_flat.cpp -o $@ -lfaiss
	mv $@ $(BIN)

faiss_hnsw:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -I$(CONDA_PREFIX)/include -L$(CONDA_PREFIX)/lib faiss_hnsw.cpp -o $@ -lfaiss
	mv $@ $(BIN)

brute_force_cpu: $(INCS) $(OBJS) brute_force_cpu.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) brute_force_cpu.o -o $@ $(LIBS)
	mv $@ $(BIN)

brute_force_gpu: $(CUINCS) $(OBJS) brute_force_gpu.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) brute_force_gpu.o -o $@ $(LIBS) $(NVLIBS)
	mv $@ $(BIN)

ivf_flat_cpu: $(INCS) $(OBJS) ivf_flat_cpu.o kmeans_cpu.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ivf_flat_cpu.o kmeans_cpu.o -o $@ $(LIBS)
	mv $@ $(BIN)

ivf_flat_gpu: $(INCS) $(OBJS) ivf_flat_gpu.o kmeans_gpu.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ivf_flat_gpu.o kmeans_gpu.o -o $@ $(LIBS) $(NVLIBS)
	mv $@ $(BIN)

parlayann_cpu: $(INCS) $(OBJS) beam_search.o 
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) beam_search.o -o $@ $(LIBS) -I$(PARLAY_INCS)/include
	mv $@ $(BIN)

quantized_search_cpu: $(INCS) $(OBJS) quantized_search_cpu.o kmeans_cpu.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) quantized_search_cpu.o kmeans_cpu.o -o $@ $(LIBS)
	mv $@ $(BIN)

gen_groundtruth: 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -I$(CONDA_PREFIX)/include -L$(CONDA_PREFIX)/lib gen_groundtruth.cpp -o $@ -lfaiss -fopenmp
	mv $@ $(BIN)

clean:
	rm *.o

