include common.mk

OBJS=data_loader.o
INCS=pqueue.hh
BIN=../bin/

all: brute_force_cpu quantized_search_cpu

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

clean:
	rm *.o

