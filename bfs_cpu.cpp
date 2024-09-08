#include <omp.h>
#include <set>
//#include "bfs.hpp"
#include "graph.h"
#include "utils.hpp"
#include "pqueue.hpp"
#include "common.hpp"

// expand node 'u' in graph 'g'
// unvisited neighbors whose distances are close to the query are inserted into the queue
// the number of distance computation performed is returned
template <typename Td, typename Tv>
int expand_node(Graph<vid_t> &g, Tv u, int dim, const Td* query, const Td* data_vectors,
                std::vector<uint8_t> &is_visited, pqueue_t<Tv> &queue) {
  auto dist_bound = queue.get_tail_dist();
  uint64_t num_distance_computation = 0;
  // for each neighbor w of node u
  //for (auto w : g.N(u)) {
  for (size_t i=0; i<g[u].size(); i++) {
    auto w = g[u][i]; 
    if (is_visited[w]) continue;
    is_visited[w] = 1;
    const Td* w_data = data_vectors + w * dim;
    auto dist = utils::compute_distance(dim, w_data, query);
    ++num_distance_computation;
    if (dist < dist_bound) queue.push(w, dist);
  }
  return num_distance_computation;
}

// This is not a good starting strategy
inline std::vector<vid_t> get_init_nodes(Graph<vid_t> &g, int num) {
  std::vector<vid_t> init_ids;
  vid_t u = rand() % g.size();
  std::set<vid_t> id_set;
  //std::cout << u << " [ ";

  //for (auto v : g.N(u)) {
  for (size_t i=0; i<g[u].size(); i++) {
    auto v = g[u][i];
    //std::cout << v << " ";
    if (id_set.find(v) == id_set.end()) {
      id_set.insert(v);
      init_ids.push_back(v);
    }
  }
  //std::cout << "]";
  return init_ids;
}

void kNN_search(int K, int qsize, int dim, size_t dsize,
                const float *queries, const float *data_vectors,
                result_t &results, char *index_file) {
  // load graph
  Graph<vid_t> g(index_file);

  // hyper-parameters
  const int L = K * 5; // queue 

  int num_threads = 0;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP ANN search (" << num_threads << " threads)\n";
  uint64_t total_count_dc = 0;

  Timer t;
  t.Start();
  #pragma omp parallel for schedule(dynamic,1) reduction(+:total_count_dc)
  for (int query_id = 0; query_id < qsize; query_id ++) {
    auto init_ids = get_init_nodes(g, L);
    std::vector<uint8_t> is_visited(dsize, 0);

    const float *query_data = queries + query_id * dim;
    pqueue_t<vid_t> S(L); // priority queue

    // initialize the queue with a random node and its neighbors
    for (size_t i = 0; i < init_ids.size(); i++) {
      auto v_id = init_ids[i];
      is_visited[v_id] = 1;
      auto *v_data = data_vectors + v_id * dim;
      auto dist = utils::compute_distance(dim, v_data, query_data);
      S.push(v_id, dist);
    }

    // start search until no more un-expanded nodes in the queue
    int idx = 0; // Index of first un-expanded candidate in the queue
    int iter = 0, num_hops = 0;
    uint64_t count_dc = 0;
    while (idx < L) {
      ++iter;
      if (!S.is_expanded(idx)) {
        S.set_expanded(idx);
        count_dc += expand_node(g, S[idx], dim, query_data, data_vectors, is_visited, S);
        ++ num_hops;
        auto r = S.get_next_index();
        if (r < idx) idx = r;
      } else ++idx;
    }

    // write the top-K nodes into results
    for (int i = 0; i < K; ++ i) {
      results[query_id * K + i] = S[i];
    }
    //std::fill(is_visited.begin(), is_visited.end(), 0); // reset visited status
    total_count_dc += count_dc;
  }
  t.Stop();
  double runtime = t.Seconds();
  auto throughput = double(qsize) / runtime;
  auto latency = runtime / qsize * 1000.0;
  printf("runtime: %f sec\n", runtime);
  printf("avg latency: %f ms/query, throughput: %f queries/sec\n", latency, throughput);
  std::cout << "total # distance computation: " << total_count_dc << "\n";
  std::cout << "average # distance computation: " << total_count_dc / qsize << "\n";
}

