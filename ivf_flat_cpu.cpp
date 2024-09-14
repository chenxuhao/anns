#include "utils.hpp"
#include "kmeans.hpp"
#include "pqueue.hpp"
#include "distance.hpp"

template <typename T>
void ANNS<T>::search(int k, int qsize, int dim, size_t npoints,
                     const T* queries, const T* data_vectors,
                     int *results, const char *index_file) {
  assert(npoints >= 10000);
  int num_threads;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("num of threads = %d\n", num_threads);

  int nclusters = std::sqrt(npoints);
  Kmeans<T> kmeans(npoints, dim, nclusters, data_vectors);
  auto centroids = kmeans.cluster_cpu();
  auto clusters = kmeans.get_clusters();
  int num_top_clusters = nclusters / 10;
  if (num_top_clusters < 1) num_top_clusters = 1;

  std::cout << "OpenMP ANN search (" << num_threads << " threads)\n";
  uint64_t total_count_dc = 0;

  Timer t;
  t.Start();
  #pragma omp parallel for schedule(dynamic,1) reduction(+:total_count_dc)
  for (int qid = 0; qid < qsize; ++qid) {
    uint64_t count_dc = 0;
    auto q_data = queries + qid * dim;
    std::vector<float> c_dist(nclusters);
    // find the top clusters
    for (int cid = 0; cid < nclusters; ++ cid)
      c_dist[cid] = compute_distance_squared(dim, q_data, &centroids[cid * dim]);
    count_dc += nclusters;
    pqueue_t<int> top_centers(num_top_clusters); // priority queue
    for (int cid = 0; cid < nclusters; ++ cid) {
      top_centers.push(cid, c_dist[cid]);
    }
    // search inside each of the top clusters
    pqueue_t<vidType> S(K);
    for (int i = 0; i < num_top_clusters; ++ i) {
      int cid = top_centers[i];
      for (auto vid : clusters[cid]) {
        auto dist = compute_distance_squared(dim, q_data, &data_vectors[vid * dim]);
        count_dc ++;
        S.push(vid, dist);
      }
    }
    // write the top-K nodes into results
    for (int i = 0; i < K; ++ i) {
      results[qid * K + i] = S[i];
    }
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

template class ANNS<float>;
