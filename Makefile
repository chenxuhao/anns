include common.mk

OBJS=data_loader.o
GOBJS=#graph.o VertexSet.o
INCS=pqueue.hpp
BIN=./bin/

all: brute_force_cpu quantized_search_cpu
g-ann: graph_search_cpu build_graph_cpu

brute_force_cpu: $(INCS) $(OBJS) brute_force_cpu.o tester.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) brute_force_cpu.o tester.o -o $@ $(LIBS)
	mv $@ $(BIN)

quantized_search_cpu: $(INCS) $(OBJS) quantized_search_cpu.o tester.o kmeans.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) quantized_search_cpu.o tester.o kmeans.o -o $@ $(LIBS)
	mv $@ $(BIN)

ivf_flat_cpu: $(INCS) $(OBJS) ivf_flat_cpu.o kmeans.o tester.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ivf_flat_cpu.o kmeans.o tester.o -o $@ $(LIBS)
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

build_graph_cpu: $(INCS) $(OBJS) $(GOBJS) build_knn_graph.o build_graph.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) $(GOBJS) build_knn_graph.o build_graph.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o

