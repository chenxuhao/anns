#include <omp.h>
#include <set>
#include "graph.h"
#include "utils.hpp"
#include "pqueue.hpp"
#include "common.hpp"
#include "distance.hpp"
#include "hash_filter.hpp"

template <typename T>
void ANNS<T>::search(int k, int qsize, int dim, size_t npoints,
                     const T* queries, const T* data_vectors,
                     int *results, const char *index_file) {
  // load graph
  Graph<vidType> g(index_file);

  // hyper-parameters
  const int L = K * 5; // queue 
  int D = g.max_degree();

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
    int64_t count_dc = 0;
    hash_filter<vidType> is_visited(L*L);
    auto query_data = queries + query_id * dim;
    //auto query_data = queries[query_id];
    new_pqueue_t<vidType> S(L); // priority queue

    auto distance_to = [&](int v) {
      return compute_distance_squared(dim, data_vectors + v*dim, query_data);
    };

    // initialize the queue with a random nodes
    std::vector<vidType> init_ids(K);
    for (int i = 0; i < K; i++) init_ids[i] = rand() % npoints;
    int bv = init_ids[0];
    float m = distance_to(bv);
    count_dc++;
    // start from the node closest to the query
    for (size_t i = 1; i < init_ids.size(); i++) {
      auto v_id = init_ids[i];
      auto dist = distance_to(v_id);
      count_dc++; 
      if (dist < m) {
        m = dist;
        bv = v_id;
      }
    }
    S.push(bv,m);
    int iter = 0, num_hops = 0, num_push = 0;

    Timer tt;
    tt.Start();
    // start search until no more un-expanded nodes in the queue
    is_visited.add(bv);
    std::vector<vidType> keep;
    keep.reserve(D); // a buffer for neighbor expansion
    while (S.has_unexpanded()) {
      ++iter;
      int idx = S.get_next_index();
      int u = S[idx];
      //g.fastprefetch(u,_MM_HINT_T0);
      S.set_front_expanded();
      if ((S.size() > K) && (S.get_dist(idx) > S.get_dist(K)*1.35)) continue;
      keep.clear();
      for (auto v: g.N(u)) {
        if (is_visited.add(v)) {
          keep.push_back(v);
          //PREFETCH_VECTOR(dim, data_vectors[v]);
        }
      }
      //float close = 1e99;
      //if (S.has_unexpanded()) {
        //int idx2 = S.get_next_index();
        //g.fastprefetch(S[idx2],_MM_HINT_T2);
        //close = S.get_dist(idx2);
      //}
      for (auto v: keep) {
        auto dist = distance_to(v);
        count_dc++;
        if ((S.size() > K) && (dist > S.get_dist(K)*1.35)) continue;
        if ((S.size() < L) || (dist < S.get_tail_dist())) {
          S.push(v,dist);
          num_push++;
          //if (dist < close) {
            //g.fastprefetch(v,_MM_HINT_T2);
            //close = dist;
          //}
        }
      }
      ++num_hops;
    }
/*
    // start search until no more un-expanded nodes in the queue
    int idx = 0; // Index of first un-expanded candidate in the queue
    while (idx < L) {
      ++iter;
      if (!S.is_expanded(idx)) {
        S.set_front_expanded(idx);
        auto dist_bound = S.get_tail_dist();
        auto u = S[idx];
        for (auto w: g.N(u)) {
          if (is_visited[w]) continue;
          is_visited[w] = 1;
          auto dist = distance_to(w);
          if (dist < dist_bound) S.push(w, dist);
          ++count_dc;
        }
        ++ num_hops;
        auto r = S.get_next_index();
        if (r < idx) idx = r;
      } else ++idx;
    }
*/
    tt.Stop();
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

// Explicit template instantiation
template class ANNS<float>;
