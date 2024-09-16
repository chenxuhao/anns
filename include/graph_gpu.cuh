#pragma once
#include "graph.h"
#include "cutil_subset.cuh"

using vlabel_t = uint32_t;

template <typename T>
class GraphGPU {
protected:
  bool is_directed_;                // is it a directed graph?
  bool has_reverse;                 // has reverse/incoming edges maintained
  vidType num_vertices;             // number of vertices
  eidType num_edges;                // number of edges
  int device_id, n_gpu;             // no. of GPUs
  int num_vertex_classes;           // number of unique vertex labels
  vidType max_degree;               // maximun degree
  eidType *d_rowptr, *d_in_rowptr;  // row pointers of CSR format
  vidType *d_colidx, *d_in_colidx;  // column induces of CSR format
  vidType *d_src_list, *d_dst_list; // for COO format
  vlabel_t *d_vlabels;              // vertex labels
public:
  GraphGPU(vidType nv, eidType ne, int vl=0, int el=0, 
           int n=0, int m=1, bool use_nvshmem=false) :
      GraphGPU(n, m, nv, ne, vl, el) {
    if (nv>0 && ne>0 && !use_nvshmem) allocateFrom(nv, ne, vl, el);
  }
  GraphGPU(Graph<T> &g, int n=0, int m=1) : 
      GraphGPU(n, m, g.V(), g.E()) {
    init(g);
  }
  GraphGPU(int n=0, int m=0, vidType nv=0, eidType ne=0, int vl=1, int el=1,
           bool directed=false, bool reverse=false) : 
      is_directed_(directed),
      has_reverse(reverse),
      num_vertices(nv),
      num_edges(ne),
      device_id(n), n_gpu(m),
      num_vertex_classes(vl),
      max_degree(0),
      d_rowptr(NULL),
      d_in_rowptr(NULL),
      d_colidx(NULL), 
      d_in_colidx(NULL), 
      d_src_list(NULL),
      d_dst_list(NULL),
      d_vlabels(NULL)
      { }
  void release() { clean(); clean_edgelist(); clean_labels(); }
  inline __device__ __host__ bool is_directed() { return is_directed_; }
  inline __device__ __host__ int get_num_devices() { return n_gpu; }
  inline __device__ __host__ vidType V() { return num_vertices; }
  inline __device__ __host__ vidType size() { return num_vertices; }
  inline __device__ __host__ eidType E() { return num_edges; }
  inline __device__ __host__ eidType sizeEdges() { return num_edges; }
  inline __device__ __host__ vidType get_max_degree() { return max_degree; }
  inline __device__ __host__ bool valid_vertex(vidType vertex) { return (vertex < num_vertices); }
  inline __device__ __host__ bool valid_edge(eidType edge) { return (edge < num_edges); }
  inline __device__ __host__ vidType get_src(eidType eid) const { return d_src_list[eid]; }
  inline __device__ __host__ vidType get_dst(eidType eid) const { return d_dst_list[eid]; }
  inline __device__ __host__ vidType* get_src_ptr(eidType eid) const { return d_src_list; }
  inline __device__ __host__ vidType* get_dst_ptr(eidType eid) const { return d_dst_list; }
  inline __device__ __host__ vidType* N(vidType vid) { return d_colidx + d_rowptr[vid]; }
  inline __device__ __host__ vidType N(vidType v, eidType e) { return d_colidx[d_rowptr[v] + e]; }
  inline __device__ __host__ eidType* rowptr() { return d_rowptr; }
  inline __device__ __host__ vidType* colidx() { return d_colidx; }
  inline __device__ __host__ eidType* out_rowptr() { return d_rowptr; }
  inline __device__ __host__ vidType* out_colidx() { return d_colidx; }
  inline __device__ __host__ eidType* in_rowptr() { return d_in_rowptr; }
  inline __device__ __host__ vidType* in_colidx() { return d_in_colidx; }
  inline __device__ __host__ eidType getOutDegree(vidType src) { return d_rowptr[src+1] - d_rowptr[src]; }
  inline __device__ __host__ eidType getInDegree(vidType src) { return d_in_rowptr[src+1] - d_in_rowptr[src]; }
  inline __device__ __host__ vidType get_degree(vidType src) { return vidType(d_rowptr[src+1] - d_rowptr[src]); }
  inline __device__ __host__ vidType getEdgeDst(eidType edge) { return d_colidx[edge]; }
  inline __device__ __host__ vidType getOutEdgeDst(eidType edge) { return d_colidx[edge]; }
  inline __device__ __host__ vidType getInEdgeDst(eidType edge) { return d_in_colidx[edge]; }
  inline __device__ __host__ eidType edge_begin(vidType src) { return d_rowptr[src]; }
  inline __device__ __host__ eidType edge_end(vidType src) { return d_rowptr[src+1]; }
  inline __device__ __host__ eidType out_edge_begin(vidType src) { return d_rowptr[src]; }
  inline __device__ __host__ eidType out_edge_end(vidType src) { return d_rowptr[src+1]; }
  inline __device__ __host__ eidType in_edge_begin(vidType src) { return d_in_rowptr[src]; }
  inline __device__ __host__ eidType in_edge_end(vidType src) { return d_in_rowptr[src+1]; }
  inline __device__ __host__ vlabel_t getData(vidType vid) { return d_vlabels[vid]; }
  inline __device__ __host__ vlabel_t* getVlabelPtr() { return d_vlabels; }
  inline __device__ __host__ vlabel_t* get_vlabel_ptr() { return d_vlabels; }

  inline __device__ __host__ void fixEndEdge(vidType vid, eidType row_end) { d_rowptr[vid+1] = row_end; }
  inline __device__ __host__ void constructEdge(eidType eid, vidType dst)  { d_colidx[eid] = dst; }
  void clean() {
    if (d_rowptr != NULL)
      CUDA_SAFE_CALL(cudaFree(d_rowptr));
    if (d_colidx != NULL)
      CUDA_SAFE_CALL(cudaFree(d_colidx));
  }
  void clean_edgelist() {
    if (d_src_list != NULL)
      CUDA_SAFE_CALL(cudaFree(d_src_list));
    if (d_dst_list != NULL)
      CUDA_SAFE_CALL(cudaFree(d_dst_list));
  }
  void clean_labels() {
    if (d_vlabels != NULL)
      CUDA_SAFE_CALL(cudaFree(d_vlabels));
  }
  void allocateFrom(vidType nv, eidType ne, bool has_vlabel = false, 
                    bool has_elabel = false, bool use_uva = false, bool has_reverse = false) {
    std::cout << "Allocating GPU memory for the graph ... ";
    if (use_uva) {
      CUDA_SAFE_CALL(cudaMallocManaged(&d_rowptr, (nv+1) * sizeof(eidType)));
      CUDA_SAFE_CALL(cudaMallocManaged(&d_colidx, ne * sizeof(vidType)));
      if (has_reverse) {
        CUDA_SAFE_CALL(cudaMallocManaged(&d_in_rowptr, (nv+1) * sizeof(eidType)));
        CUDA_SAFE_CALL(cudaMallocManaged(&d_in_colidx, ne * sizeof(vidType)));
      }
    } else {
      CUDA_SAFE_CALL(cudaMalloc((void **)&d_rowptr, (nv+1) * sizeof(eidType)));
      CUDA_SAFE_CALL(cudaMalloc((void **)&d_colidx, ne * sizeof(vidType)));
      if (has_reverse) {
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_rowptr, (nv+1) * sizeof(eidType)));
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_in_colidx, ne * sizeof(vidType)));
      }
    }
    if (has_vlabel)
      CUDA_SAFE_CALL(cudaMalloc((void **)&d_vlabels, nv * sizeof(vlabel_t)));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    std::cout << "Done\n";
  }
  void copyToDevice(Graph<T> &hg) {
    std::cout << "Copying graph data to GPU memory ... ";
    auto ne = hg.V() * hg.max_degree();
    auto h_colidx = hg.get_adj();
    if (h_colidx != NULL)
      CUDA_SAFE_CALL(cudaMemcpy(d_colidx, h_colidx, ne * sizeof(vidType), cudaMemcpyHostToDevice));
    std::cout << "Done\n";
  }
  void init(Graph<T> &g, int n, int m) {
    device_id = n;
    n_gpu = m;
    init(g);
  }
  void init(Graph<T> &hg) {
    auto nv = hg.V();
    auto ne = hg.E();
    max_degree = hg.max_degree();
    allocateFrom(nv, ne);
    copyToDevice(hg);
  }
  void toHost(Graph<T> &hg) {
    auto nv = num_vertices;
    auto ne = num_edges;
    hg.allocateFrom(nv, ne);
    auto h_rowptr = hg.out_rowptr();
    auto h_colidx = hg.out_colidx();
    CUDA_SAFE_CALL(cudaMemcpy(h_rowptr, d_rowptr, (nv+1) * sizeof(eidType), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(h_colidx, d_colidx, ne * sizeof(vidType), cudaMemcpyDeviceToHost));
  }
};

