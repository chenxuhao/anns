#include "utils.hpp"
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

  printf("Do search ...\n");
  Timer t;
  t.Start();

  // one at a time
  // #pragma omp parallel for
  // for (int qid = 0; qid < qsize; ++qid) {
    // auto query = queries + dim * qid;
    // quantizer.search(query, data_vectors, k, &results[qid * k]);
    // quantizer.faiss_search(query, k, &results[qid * k]);
  // }

  // many at a time
  quantizer.faiss_search_batch(qsize, queries, k, results);
  t.Stop();
  
  double runtime = t.Seconds();
  auto throughput = double(qsize) / runtime;
  auto latency = runtime / qsize * 1000.0;
  printf("time = %f\n", runtime);
  printf("avg latency: %f ms/query, throughput: %f queries/sec\n", latency, throughput);
}

template class ANNS<float>;
