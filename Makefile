include common.mk

OBJS=
GOBJS=#VertexSet.o
INCS=include/pqueue.hpp
CUINCS=include/pqueue.cuh
BIN=./bin/

all: brute_force_cpu quantized_search_cpu ivf_flat_cpu
g-ann: graph_search_cpu parlayann_cpu
ann-gpu: brute_force_gpu

brute_force_cpu: $(INCS) $(OBJS) brute_force_cpu.o tester.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) brute_force_cpu.o tester.o -o $@ $(LIBS)
	mv $@ $(BIN)

brute_force_gpu: $(CUINCS) $(OBJS) brute_force_gpu.o tester.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) brute_force_gpu.o tester.o -o $@ $(LIBS) $(NVLIBS)
	mv $@ $(BIN)

quantized_search_cpu: $(INCS) $(OBJS) quantized_search_cpu.o tester.o 
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) quantized_search_cpu.o tester.o -o $@ $(LIBS)
	mv $@ $(BIN)

ivf_flat_cpu: $(INCS) $(OBJS) ivf_flat_cpu.o tester.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ivf_flat_cpu.o tester.o -o $@ $(LIBS)
	mv $@ $(BIN)

ivf_flat_gpu: $(INCS) $(OBJS) ivf_flat_gpu.o kmeans_gpu.o tester.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ivf_flat_gpu.o kmeans_gpu.o tester.o -o $@ $(LIBS) $(NVLIBS)
	mv $@ $(BIN)

graph_search_cpu: $(INCS) $(OBJS) bfs_cpu.o tester.o $(GOBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) bfs_cpu.o tester.o $(GOBJS) -o $@ $(LIBS)
	mv $@ $(BIN)

graph_search_gpu: $(INCS) $(OBJS) bfs_gpu.o tester.o $(GOBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) bfs_gpu.o tester.o $(GOBJS) -o $@ $(LIBS) $(NVLIBS)
	mv $@ $(BIN)

parlayann_cpu: $(INCS) $(OBJS) beam_search.o tester.o $(GOBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) beam_search.o tester.o $(GOBJS) -o $@ $(LIBS) -I$(PARLAY_INCS)/include
	mv $@ $(BIN)

clean:
	rm *.o

