#include <queue>
#include "utils.hh"
#include "common.hpp"

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
  std::vector<pq_t> local_queues(num_threads);
  for (int qid = 0; qid < qsize; ++qid) {
    // use a local queue for each thread
    #pragma omp parallel for
    for (int i = 0; i < num_threads; ++ i) {
      std::vector<ele_t> container;
      container.reserve(k+1);
      local_queues[i] = pq_t(std::less<ele_t>(), std::move(container));
    }
    // search in parallel (multi-threading)
    #pragma omp parallel for
    for (size_t i = 0; i < vecsize; ++ i) {
      int tid = omp_get_thread_num();
      float worst = 1e9; // assume its large enough
      if(local_queues[tid].size() == size_t(k))
        worst = local_queues[tid].top().first;
      auto dist = utils::compute_distance_squared_early_stop(vecdim, queries + vecdim * qid, data_vectors + vecdim * i, worst);
      if(dist < worst)
      {
        local_queues[tid].push({dist, vid_t(i)});
        if ((int)local_queues[tid].size() > k)
          local_queues[tid].pop();
      }
    }
    // (sequentially) merge local queues into a single queue
    pq_t queue = std::move(local_queues[0]);
    for (int i = 1; i < num_threads; ++ i) {
      while (!local_queues[i].empty()) {
        if(local_queues[i].top().first < queue.top().first)
        {
          queue.push(local_queues[i].top());
          if (int(queue.size()) > k) queue.pop();
        }
        local_queues[i].pop();
      }
    }
    // write the top-k elements into results
    for (int i = 0; i < k; ++ i) {
      results[qid * k + i] = queue.top().second;
      queue.pop();
    }
  }
}

