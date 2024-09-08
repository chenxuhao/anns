#include "utils.hpp"
#include "distance.hpp"
#include "quantizer.hpp"

template <typename T>
void ANNS<T>::search(int k, int qsize, int dim, size_t npoints,
                     const T* queries, const T* data_vectors,
                     int *results, const char *index_file) {
  int num_threads;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("num_threads = %d\n", num_threads);

  int m = 8;
  int nclusters = 256;
  Quantizer<T> quantizer(m, nclusters, dim, npoints, data_vectors);

  printf("Start search\n");
  #pragma omp parallel for
  for (int qid = 0; qid < qsize; ++qid) {
    auto query = queries + dim * qid;
    quantizer.build_lookup_table(query);
    vector<pair<double, int>> distances;
    for (size_t i = 0; i < npoints; ++i) {
      auto dist = quantizer.quantized_distance(i);
      distances.emplace_back(dist, i);
    }
    sort(distances.begin(), distances.end());
    for (int i = 0; i < k; ++i) {
      results[qid * k + i] = distances[i].second;
    }
  }
}

template class ANNS<float>;
