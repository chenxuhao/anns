#pragma once

#include <algorithm>
#include <iostream>
#include <fstream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "VertexSet.hpp"

typedef int64_t eidType;   // edge ID type
typedef std::vector<vidType> VertexList; // vertex ID list

template<typename indexType>
struct Graph {
 private:
  size_t n;
  long maxDeg;
  std::vector<indexType> graph;

 public:
  long max_degree() const {return maxDeg;}
  size_t size() const {return n;}
  size_t V() const { return n; }
  size_t E() const { return n*maxDeg; }
  vidType* get_adj() { return graph.data(); }

  Graph(){}
  Graph(long maxDeg, size_t n) : maxDeg(maxDeg), n(n) {
    //graph = parlay::sequence<indexType>(n*(maxDeg+1),0);
    graph.resize(n*maxDeg);
    std::fill(graph.begin(), graph.end(), 0);
  }
  Graph(const char* gFile) { load_graph(gFile); }
  void load_graph(std::string prefix) {
    std::ifstream f_meta((prefix + ".meta.txt").c_str());
    assert(f_meta);
    int64_t nv = 0, ne = 0;
    int vid_size, eid_size, vlabel_size, elabel_size;
    int feat_len, num_vertex_classes, num_edge_classes;
    f_meta >> nv >> ne >> vid_size >> eid_size >> vlabel_size >> elabel_size
      >> maxDeg >> feat_len >> num_vertex_classes >> num_edge_classes;
    f_meta.close();
    assert(nv > 0 && ne > 0);
    n = nv;
    std::cout << "Graph index: nv = " << n << " max_degree = " << maxDeg << "\n";

    auto d = maxDeg;
    graph.resize(n*d);
    std::fill(graph.begin(), graph.end(), 0);
    indexType *adj_list = graph.data();
    auto length = graph.size();
    auto fname = prefix + ".edge.bin";
    std::ifstream inf(fname.c_str(), std::ios::binary);
    if (!inf.good()) {
      std::cerr << "Failed to open file: " << fname << "\n";
      exit(1);
    }
    inf.read(reinterpret_cast<char*>(adj_list), sizeof(indexType) * length);
    inf.close();
  }
  void allocateFrom(indexType nv, int64_t ne) {
    n = nv;
    graph.resize(ne);
    std::fill(graph.begin(), graph.end(), 0);
    assert(ne % n == 0);
    maxDeg = ne / n;
  }
  void constructEdge(int64_t eid, indexType dst) {
    indexType *edges = graph.data();
    edges[eid] = dst;
  }
  VertexSet N(indexType v) {
    indexType *adj_list = graph.data();
    return VertexSet(adj_list+(eidType)maxDeg*v, maxDeg, v);
  }
};

