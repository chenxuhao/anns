#include "utils.hpp"
#include "pqueue.hpp"
#include "distance.hpp"

template <typename T>
void ANNS<T>::search(int k, int qsize, int dim, size_t npoints,
                     const T* queries, const T* data_vectors,
                     int *results, const char *index_file) {
  int num_threads = 0;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP ANN search (" << num_threads << " threads)\n";

  Timer t;
  t.Start();
  #pragma omp parallel for //schedule(dynamic,1)
  for (int qid = 0; qid < qsize; ++qid) {
    const float *q_data = queries + qid * dim;
    pqueue_t<vidType> S(k); // priority queue
    for (size_t i = 0; i < npoints; ++ i) {
      auto *p_data = data_vectors + i * dim;
      auto dist = compute_distance_squared(dim, p_data, q_data);
      S.push(i, dist);
    }
    // write the top-k elements into results
    for (int i = 0; i < k; ++ i)
      results[qid * k + i] = S[i];
  }
  t.Stop();
  double runtime = t.Seconds();
  auto throughput = double(qsize) / runtime;
  auto latency = runtime / qsize * 1000.0;
  printf("runtime: %f sec\n", runtime);
  printf("avg latency: %f ms/query, throughput: %f queries/sec\n", latency, throughput);
  std::cout << "average # distance computation: " << npoints << "\n";
}

template class ANNS<float>;
