include common.mk

OBJS=tester.o
INCS=include/pqueue.hpp
CUINCS=include/pqueue.cuh
BIN=./bin/

all: brute_force_cpu quantized_search_cpu ivf_flat_cpu
g-ann: graph_search_cpu parlayann_cpu
ann-gpu: brute_force_gpu graph_search_gpu ivf_flat_gpu

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

graph_search_cpu: $(INCS) $(OBJS) bfs_cpu.o 
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) bfs_cpu.o -o $@ $(LIBS)
	mv $@ $(BIN)

graph_search_gpu: $(INCS) $(OBJS) bfs_gpu.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) bfs_gpu.o -o $@ $(LIBS) $(NVLIBS)
	mv $@ $(BIN)

parlayann_cpu: $(INCS) $(OBJS) beam_search.o 
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) beam_search.o -o $@ $(LIBS) -I$(PARLAY_INCS)/include
	mv $@ $(BIN)

quantized_search_cpu: $(INCS) $(OBJS) quantized_search_cpu.o 
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) quantized_search_cpu.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o

