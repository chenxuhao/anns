#include <set>
#include <map>
#include "graph.hpp"
#include "scan.h"

template<typename T, typename U>
T fetch_and_add(T &x, U inc) {
  //auto y = x;
  //x += inc;
  //return y;
  return __sync_fetch_and_add(&x, inc);
}

template<bool map_vertices, bool map_edges, typename elabel_t>
GraphT<map_vertices, map_edges, elabel_t>::GraphT(std::string prefix, bool use_dag, bool directed, 
             bool use_vlabel, bool use_elabel, bool need_reverse, bool bipartite, bool partitioned) :
    GraphT<map_vertices, map_edges, elabel_t>(directed, bipartite, 0, 0) {
  // parse file name
  inputfile_prefix = prefix;
  size_t i = prefix.rfind('/', prefix.length());
  if (i != string::npos) inputfile_path = prefix.substr(0, i);
  i = inputfile_path.rfind('/', inputfile_path.length());
  if (i != string::npos) name_ = inputfile_path.substr(i+1);
  //std::cout << "input file prefix: " << inputfile_prefix << ", graph name: " << name_ << "\n";
  if (bipartite) std::cout << "This is a Bipartite graph\n";
  load_graph(prefix, use_dag, use_vlabel, use_elabel, need_reverse, partitioned);
}

template<bool map_vertices, bool map_edges, typename elabel_t>
void GraphT<map_vertices, map_edges, elabel_t>::load_graph(std::string prefix,
                                                 bool use_dag, bool use_vlabel, bool use_elabel,
                                                 bool need_reverse, bool partitioned) {
  VertexSet::release_buffers();

  // read meta information
  read_meta_info(prefix);

  // load graph data
  if (partitioned) std::cout << "This graph is partitioned, not loading the full graph\n";
  else
    load_graph_data(prefix, use_dag, use_vlabel, use_elabel, need_reverse);
}

template<bool map_vertices, bool map_edges, typename elabel_t>
void GraphT<map_vertices, map_edges, elabel_t>::load_graph_data(std::string prefix, 
    bool use_dag, bool use_vlabel, bool use_elabel, bool need_reverse) {
  // read row pointers
  load_row_pointers(prefix);

  // read column indices
  if constexpr (map_edges) {
    map_file(prefix + ".edge.bin", edges, n_edges);
  } else {
    read_file(prefix + ".edge.bin", edges, n_edges);
    //if (n_vertices > 1500000000) std::cout << "Update: edge loaded\n";
  }

  if (is_directed_) {
    std::cout << "This is a directed graph\n";
    if (need_reverse) {
      build_reverse_graph();
      std::cout << "This graph maintains both incomming and outgoing edge-list\n";
      has_reverse = true;
    }
  } else {
    has_reverse = true;
    reverse_vertices = vertices;
    reverse_edges = edges;
  }

  // compute maximum degree
  if (max_degree == 0) compute_max_degree();
  //else std::cout << "max_degree: " << max_degree << "\n";
  assert(max_degree > 0 && max_degree < n_vertices);

  // read vertex labels
  if (use_vlabel) {
    assert (num_vertex_classes > 0);
    assert (num_vertex_classes < 255); // we use 8-bit vertex label dtype
    std::string vlabel_filename = prefix + ".vlabel.bin";
    ifstream f_vlabel(vlabel_filename.c_str());
    if (f_vlabel.good()) {
      if constexpr (map_vertices)
        map_file(vlabel_filename, vlabels, n_vertices);
      else read_file(vlabel_filename, vlabels, n_vertices);
      std::set<vlabel_t> labels;
      for (vidType v = 0; v < n_vertices; v++)
        labels.insert(vlabels[v]);
      //for (int i = 0; i < n_vertices; i++) std::cout << unsigned(vlabels[i]) << "\n";
      std::cout << "# distinct vertex labels: " << labels.size() << "\n";
      assert(size_t(num_vertex_classes) == labels.size());
    } else {
      std::cout << "WARNING: vertex label file not exist; generating random labels\n";
      vlabels = new vlabel_t[n_vertices];
      for (vidType v = 0; v < n_vertices; v++) {
        vlabels[v] = rand() % num_vertex_classes + 1;
      }
    }
    auto max_vlabel = unsigned(*(std::max_element(vlabels, vlabels+n_vertices)));
    std::cout << "maximum vertex label: " << max_vlabel << "\n";
  }
  // read edge labels
  if (use_elabel) {
    std::string elabel_filename = prefix + ".elabel.bin";
    ifstream f_elabel(elabel_filename.c_str());
    if (f_elabel.good()) {
      assert (num_edge_classes > 0);
      if constexpr (map_edges)
        map_file(elabel_filename, elabels, n_edges);
      else read_file(elabel_filename, elabels, n_edges);
      std::set<elabel_t> labels;
      for (eidType e = 0; e < n_edges; e++)
        labels.insert(elabels[e]);
      //for (int i = 0; i < n_edges; i++) {
      //  if (elabels[i] > 5 || elabels[i] < 1)
      //    std::cout << "elabels[" << i << "]=" << elabels[i] << "\n";
      //}
      //for (int i = 0; i < 10; i++) std::cout << elabels[i] << "\n";
      std::cout << "# distinct edge labels: " << labels.size() << "\n";
      //for (auto l : labels) std::cout << l << "\n";
      assert(size_t(num_edge_classes) >= labels.size());
    } else {
      std::cout << "WARNING: edge label file not exist; generating random labels\n";
      elabels = new elabel_t[n_edges];
      if (num_edge_classes < 1) {
        num_edge_classes = 1;
        for (eidType e = 0; e < n_edges; e++) {
          elabels[e] = 1;
        }
      } else {
        for (eidType e = 0; e < n_edges; e++) {
          elabels[e] = rand() % num_edge_classes + 1;
        }
      }
    }
    auto max_elabel = unsigned(*(std::max_element(elabels, elabels+n_edges)));
    std::cout << "maximum edge label: " << max_elabel << "\n";
  }
  // Orientation: convert the undirected graph into directed. 
  // An optimization used for k-cliques. This would likely decrease max_degree.
  if (use_dag) {
    assert(!is_directed_); // must be undirected before orientation
    this->orientation();
  }
  VertexSet::MAX_DEGREE = std::max(max_degree, VertexSet::MAX_DEGREE);
  labels_frequency_.clear();
}

template<bool map_vertices, bool map_edges, typename elabel_t>
GraphT<map_vertices, map_edges, elabel_t>::~GraphT() {
  deallocate();
}
 
template<bool map_vertices, bool map_edges, typename elabel_t>
void GraphT<map_vertices, map_edges, elabel_t>::deallocate() {
  if (dst_list != NULL && dst_list != edges) {
    delete [] dst_list;
    dst_list = NULL;
  }
  if (src_list != NULL) {
    delete [] src_list;
    src_list = NULL;
  }
  if (edges != NULL) {
    if constexpr (map_edges) munmap(edges, n_edges*sizeof(vidType));
    else custom_free(edges, n_edges);
    edges = NULL;
  }
  if (vertices != NULL) {
    if constexpr (map_vertices) munmap(vertices, (n_vertices+1)*sizeof(eidType));
    else custom_free(vertices, n_vertices+1);
    vertices = NULL;
  }
  if (vlabels != NULL) {
    delete [] vlabels;
    vlabels = NULL;
  }
  if (elabels != NULL) {
    delete [] elabels;
    elabels = NULL;
  }
  if (features != NULL) {
    delete [] features;
    features = NULL;
  }
}

template<bool map_vertices, bool map_edges, typename elabel_t>
void GraphT<map_vertices, map_edges, elabel_t>::read_meta_info(std::string prefix) {
  std::ifstream f_meta((prefix + ".meta.txt").c_str());
  assert(f_meta);
  int64_t nv = 0;
  if (is_bipartite_) {
    f_meta >> n_vert0 >> n_vert1;
    nv = int64_t(n_vert0) + int64_t(n_vert1);
  } else f_meta >> nv;
  if (is_bipartite_) std::cout << "Debug: Bipartite graph nv0=" << n_vert0 << " nv1=" << n_vert1 << "\n";
  f_meta >> n_edges >> vid_size >> eid_size >> vlabel_size >> elabel_size
         >> max_degree >> feat_len >> num_vertex_classes >> num_edge_classes;
  assert(sizeof(vidType) == vid_size);
  assert(sizeof(eidType) == eid_size);
  assert(sizeof(vlabel_t) == vlabel_size);
  //assert(sizeof(elabel_t) == elabel_size);
  f_meta.close();
  assert(nv > 0 && n_edges > 0);
  if (vid_size == 4) assert(nv < 4294967295);
  n_vertices = nv;
  //std::cout << "Reading graph: |V| " << nv << " |E| " << n_edges << "\n";
}

template<bool map_vertices, bool map_edges, typename elabel_t>
void GraphT<map_vertices, map_edges, elabel_t>::load_row_pointers(std::string prefix) {
  if constexpr (map_vertices) {
    map_file(prefix + ".vertex.bin", vertices, n_vertices+1);
  } else {
    read_file(prefix + ".vertex.bin", vertices, n_vertices+1);
    //if (n_vertices > 1500000000) std::cout << "Update: vertex loaded\n";
  }
}

template<bool map_vertices, bool map_edges, typename elabel_t>
void GraphT<map_vertices, map_edges, elabel_t>::compute_max_degree() {
  #pragma omp parallel for reduction(max:max_degree)
  for (vidType v = 0; v < n_vertices; v++) {
    auto deg = this->get_degree(v);
    if (deg > max_degree) max_degree = deg;
  }
}

template<bool map_vertices, bool map_edges, typename elabel_t>
VertexSet GraphT<map_vertices, map_edges, elabel_t>::N(vidType vid) const {
  assert(vid >= 0);
  assert(vid < n_vertices);
  eidType begin = vertices[vid], end = vertices[vid+1];
  if (begin > end || end > n_edges) {
    fprintf(stderr, "vertex %u bounds error: [%lu, %lu)\n", vid, begin, end);
    exit(1);
  }
  assert(end <= n_edges);
  return VertexSet(edges + begin, end - begin, vid);
}

template<bool map_vertices, bool map_edges, typename elabel_t>
void GraphT<map_vertices,map_edges, elabel_t>::sort_neighbors() {
  std::cout << "Sorting the neighbor lists (used for pattern mining)\n";
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++) {
    auto begin = edge_begin(v);
    auto end = edge_end(v);
    std::sort(edges+begin, edges+end);
  }
}

template<bool map_vertices, bool map_edges, typename elabel_t>
void GraphT<map_vertices,map_edges, elabel_t>::write_to_file(std::string outfilename, bool v, bool e, bool vl, bool el) {
  std::cout << "Writing graph to file\n";
  if (v) {
    std::ofstream outfile((outfilename+".vertex.bin").c_str(), std::ios::binary);
    if (!outfile) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile.write(reinterpret_cast<const char*>(vertices), (n_vertices+1)*sizeof(eidType));
    outfile.close();
  }

  if (e) {
    std::ofstream outfile1((outfilename+".edge.bin").c_str(), std::ios::binary);
    if (!outfile1) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile1.write(reinterpret_cast<const char*>(edges), n_edges*sizeof(vidType));
    outfile1.close();
  }

  if (vl && vlabels) {
    std::ofstream outfile((outfilename+".vlabel.bin").c_str(), std::ios::binary);
    if (!outfile) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile.write(reinterpret_cast<const char*>(&vlabels[0]), n_vertices*sizeof(vlabel_t));
    outfile.close();
  }

  if (el && elabels) {
    std::ofstream outfile((outfilename+".elabel.bin").c_str(), std::ios::binary);
    if (!outfile) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile.write(reinterpret_cast<const char*>(&elabels[0]), n_edges*sizeof(elabel_t));
    outfile.close();
  }
}
 
template<bool map_vertices, bool map_edges, typename elabel_t>
void GraphT<map_vertices, map_edges, elabel_t>::build_reverse_graph() {
  std::vector<VertexList> reverse_adj_lists(n_vertices);
  for (vidType v = 0; v < n_vertices; v++) {
    for (auto u : N(v)) {
      reverse_adj_lists[u].push_back(v);
    }
  }
  reverse_vertices = custom_alloc_global<eidType>(n_vertices+1);
  reverse_vertices[0] = 0;
  for (vidType i = 1; i < n_vertices+1; i++) {
    auto degree = reverse_adj_lists[i-1].size();
    reverse_vertices[i] = reverse_vertices[i-1] + degree;
  }
  reverse_edges = custom_alloc_global<vidType>(n_edges);
  //#pragma omp parallel for
  for (vidType i = 0; i < n_vertices; i++) {
    auto begin = reverse_vertices[i];
    std::copy(reverse_adj_lists[i].begin(), 
        reverse_adj_lists[i].end(), &reverse_edges[begin]);
  }
  for (auto adjlist : reverse_adj_lists) adjlist.clear();
  reverse_adj_lists.clear();
}

template<> VertexSet GraphT<>::out_neigh(vidType vid, vidType offset) const {
  assert(vid >= 0);
  assert(vid < n_vertices);
  auto begin = vertices[vid];
  auto end = vertices[vid+1];
  if (begin > end) {
    fprintf(stderr, "vertex %u bounds error: [%lu, %lu)\n", vid, begin, end);
    exit(1);
  }
  assert(end <= n_edges);
  return VertexSet(edges + begin + offset, end - begin, vid);
}

// TODO: fix for directed graph
template<> VertexSet GraphT<>::in_neigh(vidType vid) const {
  assert(vid >= 0);
  assert(vid < n_vertices);
  auto begin = reverse_vertices[vid];
  auto end = reverse_vertices[vid+1];
  if (begin > end) {
    fprintf(stderr, "vertex %u bounds error: [%lu, %lu)\n", vid, begin, end);
    exit(1);
  }
  assert(end <= n_edges);
  return VertexSet(reverse_edges + begin, end - begin, vid);
}
 
template<bool map_vertices, bool map_edges, typename elabel_t>
void GraphT<map_vertices, map_edges, elabel_t>::allocateFrom(vidType nv, eidType ne) {
  n_vertices = nv;
  n_edges    = ne;
  vertices = new eidType[nv+1];
  edges = new vidType[ne];
  vertices[0] = 0;
}

template<bool map_vertices, bool map_edges, typename elabel_t>
void GraphT<map_vertices,map_edges, elabel_t>::symmetrize() {
  degrees.resize(n_vertices);
  std::fill(degrees.begin(), degrees.end(), 0);
  std::cout << "Computing degrees\n";
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++) {
    degrees[v] = get_degree(v);
  }
  std::cout << "Computing new degrees\n";
  eidType num_new_edges = 0;
  #pragma omp parallel for reduction(+:num_new_edges)
  for (vidType v = 0; v < n_vertices; v++) {
    eidType i = 0;
    //std::sort(edges+edge_begin(v), edges+edge_end(v));
    for (auto u : N(v)) {
      assert(u < n_vertices);
      assert(u != v); // assuming self-loops are removed
      assert(i==0 || u != N(v,i-1)); // assuming redundant edges are removed
      if (binary_search(v, edge_begin(u), edge_end(u)))
        continue;
      fetch_and_add(degrees[u], 1);
      num_new_edges += 1;
    }
  }
  std::cout << "Adding " << num_new_edges << " new edges\n";
  eidType *new_vertices = custom_alloc_global<eidType>(n_vertices+1);
  parallel_prefix_sum<vidType,eidType>(degrees, new_vertices);
  degrees.clear();
  auto num_edges = new_vertices[n_vertices];
  std::cout << "|E| after symmetrization: " << num_edges << "\n";
  assert(num_edges <= 2*n_edges);
  vidType *new_edges = custom_alloc_global<vidType>(num_edges);
  std::cout << "Copying existing edges\n";
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v ++) {
    auto begin = new_vertices[v];
    std::copy(edges+edge_begin(v), edges+edge_end(v), &new_edges[begin]);
  }
  std::vector<vidType> offsets(n_vertices, 0);
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++) {
    offsets[v] = get_degree(v);
  }
  std::cout << "Computing new column indices\n";
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v ++) {
    for (auto u : N(v)) {
      if (binary_search(v, edge_begin(u), edge_end(u)))
        continue;
      auto begin = new_vertices[u];
      auto offset = fetch_and_add(offsets[u], 1);
      new_edges[begin+offset] = v;
    }
  }
  if constexpr (map_vertices) {
  } else {
    delete [] vertices;
  }
  if constexpr (map_edges) {
  } else {
    delete [] edges;
  }
  vertices = new_vertices;
  edges = new_edges;
  n_edges = num_edges;
  sort_neighbors();
}

template<bool map_vertices, bool map_edges, typename elabel_t>
void GraphT<map_vertices, map_edges, elabel_t>::orientation(std::string outfile_prefix) {
  //std::cout << "Orientation enabled, generating DAG\n";
  if (is_directed_) return;
  //Timer t;
  //t.Start();
  degrees.resize(n_vertices);
  std::fill(degrees.begin(), degrees.end(), 0);
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++) {
    degrees[v] = this->get_degree(v);
  }
  std::vector<vidType> new_degrees(n_vertices, 0);
  #pragma omp parallel for
  for (vidType src = 0; src < n_vertices; src ++) {
    for (auto dst : N(src)) {
      if (degrees[dst] > degrees[src] ||
          (degrees[dst] == degrees[src] && dst > src)) {
        new_degrees[src]++;
      }
    }
  }
  max_degree = *(std::max_element(new_degrees.begin(), new_degrees.end()));
  eidType *new_vertices = custom_alloc_global<eidType>(n_vertices+1);
  parallel_prefix_sum<vidType,eidType>(new_degrees, new_vertices);
  auto num_edges = new_vertices[n_vertices];

  //std::cout << "|E| after clean: " << num_edges << "\n";
  assert(n_edges == num_edges*2);
  std::string vertex_file_path = outfile_prefix + ".vertex.bin";
  std::string edge_file_path = outfile_prefix + ".edge.bin";

  vidType *new_edges;
  int fd = 0;
  size_t num_bytes = 0;
  void *map_ptr = NULL;
  if (outfile_prefix == "") {
    //std::cout << "Generating the new graph in memory\n";
    new_edges = custom_alloc_global<vidType>(num_edges);
  } else {
    //std::cout << "generating the new graph in disk\n";
    std::ofstream outfile(vertex_file_path.c_str(), std::ios::binary);
    if (!outfile) {
      std::cout << "File not available\n";
      throw 1;
    }
    outfile.write(reinterpret_cast<const char*>(new_vertices), (n_vertices+1)*sizeof(eidType));
    outfile.close();

    fd = open(edge_file_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
    if (fd == -1) {
      perror("Error opening file for writing");
      exit(EXIT_FAILURE);
    }
    num_bytes = num_edges * sizeof(vidType)+1;
    // Stretch the file size to the size of the (mmapped) bytes
    if (lseek(fd, num_bytes-1, SEEK_SET) == -1) {
      close(fd);
      perror("Error calling lseek() to 'stretch' the file");
      exit(EXIT_FAILURE);
    }
    // Something needs to be written at the end of the file to make the file actually have the new size.
    if (write(fd, "", 1) == -1) {
      close(fd);
      perror("Error writing last byte of the file");
      exit(EXIT_FAILURE);
    }
    // Now the file is ready to be mmapped
    map_ptr = mmap(0, num_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    new_edges = (vidType*)map_ptr;
  }

  #pragma omp parallel for
  for (vidType src = 0; src < n_vertices; src ++) {
    auto begin = new_vertices[src];
    eidType offset = 0;
    for (auto dst : N(src)) {
      if (degrees[dst] > degrees[src] ||
          (degrees[dst] == degrees[src] && dst > src)) {
        new_edges[begin+offset] = dst;
        offset ++;
      }
    }
  }
  //std::cout << "deleting old graph\n";
  if constexpr (map_vertices) {
  } else {
    delete [] vertices;
  }
  if constexpr (map_edges) {
  } else {
    delete [] edges;
  }
  n_edges = num_edges;
  vertices = new_vertices;
  edges = new_edges;
  if (outfile_prefix != "") {
    // Write it now to disk
    if (msync(map_ptr, num_bytes, MS_SYNC) == -1)
      perror("Could not sync the file to disk");
    // Don't forget to free the mmapped memory
    if (munmap(map_ptr, num_bytes) == -1) {
      close(fd);
      perror("Error un-mmapping the file");
      exit(EXIT_FAILURE);
    }
    // Un-mmaping doesn't close the file, so we still need to do that.
    close(fd);
  }
  is_directed_ = true;
  //t.Stop();
  //std::cout << "Time on generating the DAG: " << t.Seconds() << " sec\n";
}

template<bool map_vertices, bool map_edges, typename elabel_t>
bool GraphT<map_vertices,map_edges, elabel_t>::binary_search(vidType key, eidType begin, eidType end) const {
  auto l = begin;
  auto r = end-1;
  while (r >= l) { 
    auto mid = l + (r - l) / 2;
    auto value = getEdgeDst(mid);
    if (value == key) return true;
    if (value < key) l = mid + 1; 
    else r = mid - 1; 
  } 
  return false;
}

// if u is an outgoing neighbor of v
template<> bool GraphT<>::is_neighbor(vidType v, vidType u) const {
  return binary_search(u, edge_begin(v), edge_end(v));
}

template<> bool GraphT<>::is_connected(vidType v, vidType u) const {
  auto v_deg = this->get_degree(v);
  auto u_deg = this->get_degree(u);
  bool found;
  if (v_deg < u_deg) {
    found = binary_search(u, edge_begin(v), edge_end(v));
  } else {
    found = binary_search(v, edge_begin(u), edge_end(u));
  }
  return found;
}

template<> bool GraphT<>::is_connected(std::vector<vidType> sg) const {
  return false;
}

template<bool map_vertices, bool map_edges, typename elabel_t>
void GraphT<map_vertices,map_edges, elabel_t>::print_meta_data() const {
  std::cout << "|V|: " << n_vertices << ", |E|: " << n_edges << ", Max Degree: " << max_degree << "\n";
  if (num_vertex_classes > 0) {
    std::cout << "vertex-|\u03A3|: " << num_vertex_classes;
    if (!labels_frequency_.empty()) 
      std::cout << ", Max Label Frequency: " << max_label_frequency_;
    std::cout << "\n";
  } else {
    //std::cout  << "This graph does not have vertex labels\n";
  }
  if (num_edge_classes > 0) {
    std::cout << "edge-|\u03A3|: " << num_edge_classes << "\n";
  } else {
    //std::cout  << "This graph does not have edge labels\n";
  }
  if (feat_len > 0) {
    std::cout << "Vertex feature vector length: " << feat_len << "\n";
  } else {
    //std::cout  << "This graph has no input vertex features\n";
  }
}

template class GraphT<false, false, float>;
template class GraphT<false, false, int32_t>;
template class GraphT<false, true, int32_t>;
template class GraphT<true, true, int32_t>;
