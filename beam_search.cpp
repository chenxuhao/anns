#include <set>
#include <vector>
#include "graph.h"
#include "utils.hpp"
#include "distance.hpp"

// main beam search
template<typename indexType = vidType, typename distanceType = float>
std::pair<std::pair<std::vector<std::pair<indexType, distanceType>>, std::vector<std::pair<indexType, distanceType>>>, size_t>
beam_search(Graph<indexType> &G, std::vector<indexType> starting_points, QueryParams QP,
            int K, int dim, size_t dsize,
            const float *q_data, const float *data_vectors) {
  if (starting_points.size() == 0) {
    std::cout << "beam search expects at least one start point" << std::endl;
    abort();
  }

  // compare two (node_id,distance) pairs, first by distance and then id if equal
  auto less = [&](std::pair<indexType, distanceType> a, std::pair<indexType, distanceType> b) {
    return a.second < b.second || (a.second == b.second && a.first < b.first);
  };
 
  // used as a hash filter (can give false negative -- i.e. can say not in table when it is)
  int bits = std::max<int>(10, std::ceil(std::log2(QP.beamSize * QP.beamSize)) - 2);
  std::vector<indexType> hash_filter(1 << bits, -1);
  auto has_been_seen = [&](indexType a) -> bool {
    //int loc = parlay::hash64_2(a) & ((1 << bits) - 1);
    int loc = std::hash<indexType>{}(a) & ((1 << bits) - 1);
    if (hash_filter[loc] == a) return true;
    hash_filter[loc] = a;
    return false;
  };

  std::vector<std::pair<indexType, distanceType>> frontier;
  frontier.reserve(QP.beamSize);
  for (auto p : starting_points) {
    auto *p_data = data_vectors + p * dim;
    auto dist = compute_distance_squared(dim, p_data, q_data);
    frontier.push_back(std::pair<indexType, distanceType>(p, dist));
  }
  std::sort(frontier.begin(), frontier.end(), less);

  // The subset of the frontier that has not been visited
  // Use the first of these to pick next vertex to visit.
  std::vector<std::pair<indexType, distanceType>> unvisited_frontier(QP.beamSize);
  unvisited_frontier[0] = frontier[0];

  // maintains sorted set of visited vertices (id-distance pairs)
  std::vector<std::pair<indexType, distanceType>> visited;
  visited.reserve(2 * QP.beamSize);

  // counters
  size_t dist_cmps = starting_points.size();
  int remain = 1;
  int num_visited = 0;

  // used as temporaries in the loop
  std::vector<std::pair<indexType, distanceType>> new_frontier(std::max<size_t>(QP.beamSize,starting_points.size()) + G.max_degree());
  std::vector<std::pair<indexType, distanceType>> candidates;
  candidates.reserve(G.max_degree());
  std::vector<indexType> keep;
  keep.reserve(G.max_degree());

  // The main loop.  Terminate beam search when the entire frontier
  // has been visited or have reached max_visit.
  while (remain > 0 && num_visited < QP.limit) {
    // the next node to visit is the unvisited frontier node that is closest to q
    std::pair<indexType, distanceType> current = unvisited_frontier[0];
    //G[current.first].prefetch();
    // add to visited set
    visited.insert(std::upper_bound(visited.begin(), visited.end(), current, less), current);
    num_visited++;

    // keep neighbors that have not been visited (using approximate hash).
    // Note that if a visited node is accidentally kept due to approximate
    // hash it will be removed below by the union or will not bump anyone else.
    candidates.clear();
    keep.clear();
    //long num_elts = std::min<long>(G[current.first].size(), QP.degree_limit);
    //for (indexType i=0; i<num_elts; i++) {
      //auto a = G[current.first][i];
    for (auto a : G.N(current.first)) {
      if (has_been_seen(a)) continue;  // skip if already seen
      keep.push_back(a);
      //TODO: prefetch a' vector data;
    }

    // Further filter on whether distance is greater than current
    // furthest distance in current frontier (if full).
    distanceType cutoff = ((frontier.size() < size_t(QP.beamSize))
                           ? (distanceType)std::numeric_limits<int>::max()
                           : frontier[frontier.size() - 1].second);
    for (auto a : keep) {
      auto dist = compute_distance_squared(dim, data_vectors + a * dim, q_data);
      dist_cmps++;
      // skip if frontier not full and distance too large
      if (dist >= cutoff) continue;
      candidates.push_back(std::pair{a, dist});
    }

    // sort the candidates by distance from p
    std::sort(candidates.begin(), candidates.end(), less);

    // union the frontier and candidates into new_frontier, both are sorted
    auto new_frontier_size =
        std::set_union(frontier.begin(), frontier.end(), candidates.begin(),
                       candidates.end(), new_frontier.begin(), less) - new_frontier.begin();

    // trim to at most beam size
    new_frontier_size = std::min<size_t>(QP.beamSize, new_frontier_size);

    // if a k is given (i.e. k != 0) then trim off entries that have a
    // distance greater than cut * current-kth-smallest-distance.
    // Only used during query and not during build.
    if (QP.k > 0 && new_frontier_size > QP.k)// && Points[0].is_metric())
      new_frontier_size = (std::upper_bound(new_frontier.begin(),
                            new_frontier.begin() + new_frontier_size,
                            std::pair{0, QP.cut * new_frontier[QP.k].second}, less) -
                            new_frontier.begin());

    // copy new_frontier back to the frontier
    frontier.clear();
    for (indexType i = 0; i < new_frontier_size; i++)
      frontier.push_back(new_frontier[i]);

    // get the unvisited frontier (we only care about the first one)
    remain = std::set_difference(frontier.begin(), frontier.end(), visited.begin(),
                                 visited.end(), unvisited_frontier.begin(), less) -
             unvisited_frontier.begin();
  }

  return std::make_pair(std::make_pair(frontier, visited), dist_cmps);
}

template <typename T>
void ANNS<T>::search(int k, int qsize, int dim, size_t npoints,
                     const T* queries, const T* data_vectors,
                     int *results, const char *index_file) {
  // load graph
  Graph<vidType> g(index_file);

  // hyper-parameters
  const int L = K * 5; // queue 
  QueryParams PQ(0, L, 0, g.V(), g.max_degree());

  int num_threads = 0;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP ANN search (" << num_threads << " threads)\n";
 
  int64_t total_count_dc = 0;
  Timer t;
  t.Start();

  #pragma omp parallel for schedule(dynamic,1) reduction(+:total_count_dc)
  for (int query_id = 0; query_id < qsize; query_id ++) {
    std::vector<vidType> starting_points(K);
    for (int i = 0; i < K; i++) starting_points[i] = rand() % npoints;
    const float *query_data = queries + query_id * dim;
    auto res = beam_search(g, starting_points, PQ, K, dim, npoints, query_data, data_vectors);
    // write the top-K nodes into results
    for (int i = 0; i < K; ++ i) {
      results[query_id * K + i] = res.first.first[i].first;
    }
    total_count_dc += res.second;
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

template class ANNS<float>;
