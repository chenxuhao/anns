#include <queue>
#include "utils.hpp"
#include "pqueue.hpp"
#include "common.hpp"

typedef std::pair<float, vid_t> ele_t;
typedef std::priority_queue<ele_t> pq_t;

void kNN_search(int k, int qsize, int dim, size_t vecsize,
    const float *queries, const float *data_vectors,
    result_t &results, char* index) {
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
  for (int qid = 0; qid < qsize; ++qid) {
    uint64_t count_dc = 0;

    const float *q_data = queries + qid * dim;
    pqueue_t<vid_t> S(k); // priority queue

    for (size_t i = 0; i < vecsize; ++ i) {
      auto *p_data = data_vectors + i * dim;
      auto dist = utils::compute_distance(dim, p_data, q_data);
      S.push(i, dist);
      count_dc ++;
    }
    // write the top-k elements into results
    for (int i = 0; i < k; ++ i)
      results[qid * k + i] = S[i];
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

