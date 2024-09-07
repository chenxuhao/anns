#include <omp.h>
#include "bfs.hpp"

void kNN_search(int K, int qsize, int dim, size_t dsize,
                const float *queries, const float *data_vectors,
                result_t &results, char *index_file) {
  // load graph
  Graph g(index_file);

  // hyper-parameters
  const int L = K * 5; // queue 

  int num_threads = 0;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP ANN search (" << num_threads << " threads)\n";

  #pragma omp parallel for
  for (int query_id = 0; query_id < qsize; query_id ++) {
    auto init_ids = get_init_nodes(g, L);
    std::vector<uint8_t> is_visited(dsize, 0);

    //Timer t;
    //t.Start();
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
    //t.Stop();
    //double runtime = t.Seconds();
    //printf("query %d latency: %f sec\n", query_id, runtime);
    //std::fill(is_visited.begin(), is_visited.end(), 0); // reset visited status
  }
}

