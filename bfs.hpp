#include <set>
#include "graph.hpp"
#include "utils.hpp"
#include "pqueue.hpp"
#include "common.hpp"

// expand node 'u' in graph 'g'
// unvisited neighbors whose distances are close to the query are inserted into the queue
// the number of distance computation performed is returned
template <typename Td, typename Tv>
int expand_node(Graph &g, Tv u, int dim, const Td* query, const Td* data_vectors,
                std::vector<uint8_t> &is_visited, pqueue_t<Tv> &queue) {
  auto dist_bound = queue.get_tail_dist();
  uint64_t num_distance_computation = 0;
  // for each neighbor w of node u
  for (auto w : g.N(u)) {
    if (is_visited[w]) continue;
    is_visited[w] = 1;
    const Td* w_data = data_vectors + w * dim;
    auto dist = utils::compute_distance(dim, w_data, query);
    ++num_distance_computation;
    if (dist < dist_bound) queue.push(w, dist);
  }
  return num_distance_computation;
}

template <typename Td, typename Tv>
int expand_nodes_in_parallel(Graph &g, int P, int dim, const Td* query, const Td* data_vectors,
                             std::vector<uint8_t> &is_visited, pqueue_t<Tv> &queue) {
  auto dist_bound = queue.get_tail_dist();
  uint64_t num_distance_computation = 0;
  auto nodes = queue.fetch_unexpanded_nodes(P);
  for (int i = 0; i < P; i++){
    auto idx  = nodes[i];
    auto u = queue[idx];
    // for each neighbor w of node u
    for (auto w : g.N(u)) {
      if (is_visited[w]) continue;
      is_visited[w] = 1;
      const Td* w_data = data_vectors + w * dim;
      auto dist = utils::compute_distance(dim, w_data, query);
      ++num_distance_computation;
      if (dist < dist_bound) queue.push(w, dist);
    }
  }
  return num_distance_computation;
}

// This is not a good starting strategy
inline std::vector<vid_t> get_init_nodes(Graph &g, int num) {
  std::vector<vid_t> init_ids;
  init_ids.clear();
  auto num_v_ = g.V();
  vid_t u = rand() % num_v_;
  //printf("randomly pick a node: %d\n", u);
  std::set<vid_t> id_set;
  for (auto v_id : g.N(u)) {
    if (id_set.find(v_id) == id_set.end()) {
      id_set.insert(v_id);
      //printf("add neighbor: %d\n", v_id);
      init_ids.push_back(v_id);
    }
  }
  /*
  // If u is neighbors are not enough, add other random vertices
  auto tmp_id = u + 1; // use tmp_id to replace rand().
  while (init_ids.size() < size_t(num)) {
    if (tmp_id == num_v_) tmp_id = 0;
    auto v_id = tmp_id++;
    if(id_set.find(v_id) == id_set.end()) {
      id_set.insert(v_id);
      init_ids.push_back(v_id);
    }
  }
  //*/
  return init_ids;
}

