#include <queue>
#include "utils.hh"
#include "common.hpp"
#include "quantizer.hpp"

typedef std::pair<float, vid_t> ele_t;
typedef std::priority_queue<ele_t> pq_t;

void kNN_search(int k, int qsize, int vecdim, size_t vecsize,
                const float *queries, const float *data_vectors,
                result_t &results) {
  int num_threads;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("num_threads = %d\n", num_threads);

  int m = 8;
  int nclusters = 256;
  Quantizer<float> quantizer(m, nclusters, vecdim, vecsize, data_vectors);

  printf("Start search\n");
  #pragma omp parallel for
  for (int qid = 0; qid < qsize; ++qid) {
    auto query = queries + vecdim * qid;
    quantizer.build_lookup_table(query);
    vector<pair<double, int>> distances;

    for (size_t i = 0; i < vecsize; ++i) {
      auto dist = quantizer.quantized_distance(i);
      distances.emplace_back(dist, i);
    }
    sort(distances.begin(), distances.end());
    for (int i = 0; i < k; ++i) {
      results[qid * k + i] = distances[i].second;
    }
  }
}

