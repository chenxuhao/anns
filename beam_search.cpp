#include <set>
#include <vector>
#include "graph.h"
#include "utils.hpp"
#include "distance.hpp"
// #include "graph_diskann.hpp"
#include "prefetch.hpp"


// main beam search
template<typename indexType = vidType, typename distanceType = float>
std::pair<std::pair<std::vector<std::pair<indexType, distanceType>>, std::vector<std::pair<indexType, distanceType>>>, size_t>
beam_search(Graph<indexType> &G, std::vector<indexType> starting_points, QueryParams QP,
            int K, int dim, size_t dsize,
            const float *q_data, const float *data_vectors,
            int po = 1, int pl = 1) {
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
    auto dist = compute_ip_distance_simd(dim, p_data, q_data);
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
    
    // Directly access neighbor nodes
    auto neighbors = G.N(current.first);
    
    // add to visited set
    visited.insert(std::upper_bound(visited.begin(), visited.end(), current, less), current);
    num_visited++;

    // keep neighbors that have not been visited (using approximate hash).
    // Note that if a visited node is accidentally kept due to approximate
    // hash it will be removed below by the union or will not bump anyone else.
    candidates.clear();
    keep.clear();
    
    for (auto a : neighbors) {
      if (has_been_seen(a)) continue;  // skip if already seen
      keep.push_back(a);
      
      // prefetch the neighbors
      if (po > 0 && pl > 0) {
        // Prefetch the current neighbor node's vector data
        for (int offset = 0; offset < po; offset++) {
          const float *vec_ptr = data_vectors + a * dim + offset * 64 / sizeof(float);
          for (int line = 0; line < pl; line++) {
            prefetch::prefetch(vec_ptr + line * 16, prefetch::CacheLevel::L1);
          }
        }
      }
    }

    // Further filter on whether distance is greater than current
    // furthest distance in current frontier (if full).
    distanceType cutoff = ((frontier.size() < size_t(QP.beamSize))
                           ? (distanceType)std::numeric_limits<int>::max()
                           : frontier[frontier.size() - 1].second);
    for (auto a : keep) {
      auto dist = compute_ip_distance_simd(dim, data_vectors + a * dim, q_data);
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

// Beam search optimization function - using real data samples
std::tuple<int, int, int> OptimizeBeamSearch(Graph<vidType> &G, int dim, size_t npoints, 
                      const float *data_vectors, int K) {
  // Setup for optimization sample queries
  const int sample_points_num = 1000; // Number of sample queries
  
  // Settings for prefetch parameters
  const int kTryPos = 7;  // Maximum prefetch offset to try
  const int kTryPls = 7;  // Maximum prefetch lines to try
  
  std::vector<float> optimize_queries(sample_points_num * dim);
  
  // Generate queries by randomly sampling from actual data
  srand(42); // Set random seed for consistent results
  std::vector<size_t> sample_indices(sample_points_num);
  
  for (int i = 0; i < sample_points_num; i++) {
    // Randomly select data point index
    sample_indices[i] = rand() % npoints;
    
    // Use selected data point as query (sample from actual data)
    memcpy(optimize_queries.data() + (int64_t)i * dim, 
           data_vectors + sample_indices[i] * dim, 
           dim * sizeof(float));
    
    // Add small noise to avoid exact same point
    for (int j = 0; j < dim; j++) {
      float noise = ((float)rand() / RAND_MAX - 0.5f) * 0.05f; // ±2.5% noise
      optimize_queries[i * dim + j] += noise;
    }
  }
  
  // Settings for beam sizes and prefetch parameters to try
  std::vector<int> try_beam_sizes = {500};
  
  // Prefetch offset and lines values (0 disables prefetch, 1+ enables)
  std::vector<int32_t> try_pos(kTryPos);
  std::vector<int32_t> try_pls(kTryPls);
  std::iota(try_pos.begin(), try_pos.end(), 0);  // 0, 1, 2, 3
  std::iota(try_pls.begin(), try_pls.end(), 0);  // 0, 1, 2, 3
  
  std::vector<vidType> dummy_results(K);
  
  // Function to benchmark beam search with current prefetch settings
  auto benchmark_func = [&](int beam_size, int po, int pl) {
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < sample_points_num; ++i) {
      // Setup starting points - exclude sampled data points
      std::vector<vidType> starting_points(K);
      for (int j = 0; j < K; j++) {
        vidType rand_idx;
        bool is_sample;
        do {
          rand_idx = rand() % npoints;
          is_sample = false;
          for (int k = 0; k < sample_points_num; k++) {
            if (rand_idx == sample_indices[k]) {
              is_sample = true;
              break;
            }
          }
        } while (is_sample); // Only use points that aren't samples
        
        starting_points[j] = rand_idx;
      }
      
      const float *query_data = optimize_queries.data() + (int64_t)i * dim;
      
      // Only perform prefetch if enabled
      if (po > 0 && pl > 0) {
        // Prefetch vector data for starting points
        for (int j = 0; j < K; j++) {
          // Prefetch po vectors
          for (int offset = 0; offset < po; offset++) {
            const float *vec_ptr = data_vectors + starting_points[j] * dim + offset * 64 / sizeof(float);
            
            // Prefetch pl chunks of 64 bytes each
            for (int line = 0; line < pl; line++) {
              prefetch::prefetch(vec_ptr + line * 16, prefetch::CacheLevel::L1); // 64 bytes = 16 floats
            }
          }
        }
      }
      
      // Set beam search parameters
      QueryParams QP(0, beam_size, 0, G.V(), G.max_degree());
      auto res = beam_search(G, starting_points, QP, K, dim, npoints, query_data, data_vectors, po, pl);
      
      // Store results in dummy_results (not actually needed for optimization)
      for (int j = 0; j < K; ++j) {
        if (j < static_cast<int>(res.first.first.size()))
          dummy_results[j] = res.first.first[j].first;
      }
    }
  };
  
  printf("=============Start beam search optimization=============\n");
  // Warmup run
  benchmark_func(500, 1, 1);
  
  float min_ela = std::numeric_limits<float>::max();
  int best_beam_size = 500;  // Default value
  int best_po = 1;
  int best_pl = 1;
  
  // Test all parameter combinations
  for (auto beam_size : try_beam_sizes) {
    for (auto po : try_pos) {
      for (auto pl : try_pls) {
        auto st = std::chrono::high_resolution_clock::now();
        benchmark_func(beam_size, po, pl);
        auto ed = std::chrono::high_resolution_clock::now();
        auto ela = std::chrono::duration<double>(ed - st).count();
        
        printf("Beam size = %d, po = %d, pl = %d: %.4f seconds\n", 
               beam_size, po, pl, ela);
        
        if (ela < min_ela) {
          min_ela = ela;
          best_beam_size = beam_size;
          best_po = po;
          best_pl = pl;
        }
      }
    }
  }
  
  // Compare with baseline setting (beam_size=K*5, po=1, pl=1)
  float baseline_ela;
  {
    auto st = std::chrono::high_resolution_clock::now();
    benchmark_func(500, 1, 1);
    auto ed = std::chrono::high_resolution_clock::now();
    baseline_ela = std::chrono::duration<double>(ed - st).count();
  }
  
  // Compare with po=0, pl=0 (prefetch disabled) setting
  float no_prefetch_ela;
  {
    auto st = std::chrono::high_resolution_clock::now();
    benchmark_func(500, 0, 0);
    auto ed = std::chrono::high_resolution_clock::now();
    no_prefetch_ela = std::chrono::duration<double>(ed - st).count();
  }
  
  printf("Best beam size = %d, Best po = %d, Best pl = %d\n"
         "Gaining %6.2f%% performance improvement compared to baseline\n"
         "Gaining %6.2f%% performance improvement compared to no prefetch\n"
         "=============Done optimization=============\n",
         best_beam_size, best_po, best_pl,
         100.0 * (baseline_ela / min_ela - 1),
         100.0 * (no_prefetch_ela / min_ela - 1));
  
  // Return optimization results
  return std::make_tuple(best_beam_size, best_po, best_pl);
}

// Keep original ANNS<T>::search function signature
template <typename T>
void ANNS<T>::search(int k, int qsize, int dim, size_t npoints,
                    const T* queries, const T* data_vectors,
                    int *results, const char *index_file) {
  // load graph
  Graph<vidType> g(index_file);

  // Default prefetch settings
  int po = 1;
  int pl = 1;
  int beam_size = k * 5; // Default beam size

  std::cout << "Running beam search optimization...\n";
  // Use values returned from optimization function
  auto opt_result = OptimizeBeamSearch(g, dim, npoints, data_vectors, k);
  
  // Get individual values instead of using C++17 structured binding
  beam_size = std::get<0>(opt_result);
  po = std::get<1>(opt_result);
  pl = std::get<2>(opt_result);
  
  std::cout << "Applied optimized parameters: beam_size=" << beam_size 
            << ", po=" << po << ", pl=" << pl << std::endl;

  // Set query parameters with optimized values
  QueryParams PQ(0, beam_size, 0, g.V(), g.max_degree());

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
    std::vector<vidType> starting_points(k);
    for (int i = 0; i < k; i++) starting_points[i] = rand() % npoints;
    const float *query_data = queries + query_id * dim;
    
    // Use optimized prefetch settings
    if (po > 0 && pl > 0) {
      for (int i = 0; i < k; i++) {
        // Prefetch po vector blocks
        for (int offset = 0; offset < po; offset++) {
          const float *vec_ptr = data_vectors + starting_points[i] * dim + offset * 64 / sizeof(float);
          
          // Prefetch pl chunks of 64 bytes each
          for (int line = 0; line < pl; line++) {
            prefetch::prefetch(vec_ptr + line * 16, prefetch::CacheLevel::L1);
          }
        }
      }
    }
    
    // Pass optimized prefetch parameters
    auto res = beam_search(g, starting_points, PQ, k, dim, npoints, query_data, data_vectors, po, pl);
    
    // Store results
    for (int i = 0; i < k; ++i) {
      if (i < static_cast<int>(res.first.first.size()))
        results[query_id * k + i] = res.first.first[i].first;
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
