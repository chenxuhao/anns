#include <omp.h>
#include "bfs.hpp"
#include "pqueue.cuh"
#include "graph_gpu.cuh"
#include "cuda_profiler_api.h"
#include "cuda_launch_config.cuh"

template <typename Td, typename Tv>
__device__ uint64_t expand_node(GraphGPU g, Tv u, int dim, int wid,
                                const Td* query, 
                                const Td* data_vectors,
                                uint8_t* is_visited, 
                                PQ_GPU<Tv> queues) {
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  __shared__ uint64_t count[WARPS_PER_BLOCK];
  if (thread_lane == 0) count[warp_lane] = 0;
  auto deg = g.get_degree(u);
  // each warp takes one neighbor
  for (int idx = warp_lane; idx < deg; idx += WARPS_PER_BLOCK) {
    auto w = g.N(u)[idx];
    if (is_visited[w]) continue;
    if (thread_lane == 0) is_visited[w] = 1;
    const Td* w_data = data_vectors + w * dim;
    // use a warp to compute distance (SIMD)
    auto dist = cutils::compute_distance(dim, w_data, query);
    ++ count[warp_lane];
    // TODO: parallel push, incorrect
    auto dist_bound = queues.get_tail_dist(wid); // distance upper bound
    if (dist < dist_bound) queues.push(wid, w, dist);
  }
  return count[warp_lane];
}

typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;

__global__ void //__launch_bounds__(BLOCK_SIZE, 8)
BestFirstSearch(GraphGPU g, int K, int qsize, int dim, size_t dsize,
                const float *queries, 
                const float *data_vectors,
                vid_t *results, uint8_t* visited, 
                PQ_GPU<vid_t> queues,
                gpu_long_t* total_count_dc) {
  //int thread_id   = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  //int warp_id     = thread_id   / WARP_SIZE;               // global warp index
  int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
 
  int wid = blockIdx.x; // worker id is the same as thread block id
  uint8_t* is_visited = visited + wid*dsize;

  int L = queues.get_capacity();
  assert(WARP_SIZE == MAX_DEG);
  assert(MASTER_QUEUE_SIZE >= L);
  __shared__ vid_t candidates[BLOCK_SIZE];
  __shared__ float cand_distances[BLOCK_SIZE];
  __shared__ BlockScan::TempStorage temp_storage;
  __shared__ int unexpanded_nodes_indices[WARPS_PER_BLOCK];
  __shared__ uint64_t count_dc[WARPS_PER_BLOCK];

  // each thread block takes a query
  for (int qid = blockIdx.x; qid < qsize; qid += blockDim.x) {
    const float *query_data = queries + qid * dim;

    // initialize the queue with random nodes
    // each thread in the block initializes one element
    for (int i = threadIdx.x; i < L; i+=blockDim.x) {
      vid_t v = i; // can be randomized
      queues.set_vid(wid, i, v);
      is_visited[v] = 1;
    }
    // each warp computes a distance
    for (int i = warp_lane; i < queues.size(wid); i += WARPS_PER_BLOCK) {
      auto v = queues.get_vid(wid, i);
      auto *v_data = data_vectors + v * dim;
      auto dist = cutils::compute_distance(dim, v_data, query_data);
      if (thread_lane == 0) {
        queues.set_dist(wid, i, dist);
        count_dc[warp_lane] += 1;
      }
    }
    // sort the queue using a thread block
    queues.reorder(wid);

    // start search until no more un-expanded nodes in the queue
    int idx = 0; // Index of first un-expanded candidate in the queue
    if (thread_lane == 0) count_dc[warp_lane] = 0;
    while (idx < L) {
      int total_tasks = 0;
      // pick an expanded node from the master queue for each warp
      for (int i = threadIdx.x + idx; i < MASTER_QUEUE_SIZE; i += BLOCK_SIZE) {
        int unexpanded = queues.is_expanded(wid, i) ? 0 : 1;
        int num_tasks = 0, offset = 0;
        BlockScan(temp_storage).ExclusiveSum(unexpanded, offset, num_tasks);
        // write the indices of the unexpanded nodes to the shared memory
        int loc = total_tasks + offset;
        if (unexpanded && loc < WARPS_PER_BLOCK)
          unexpanded_nodes_indices[loc] = i;
        total_tasks += num_tasks;
        if (total_tasks >= WARPS_PER_BLOCK)
          break;
      }
      // each warp expand one node
      int vid_index = unexpanded_nodes_indices[warp_lane];
      auto vid = queues.get_vid(wid, vid_index);
      if (thread_lane == 0) queues.set_expanded(wid, vid);
      //count_dc += expand_node<float, vid_t>(g, cand, dim, wid,
      //                                          query_data, data_vectors,
      //                                          is_visited, candidates, queues);
      // each thread in the warp loads one neighbor
      auto w = g.N(vid)[thread_lane];
      int unvisited = is_visited[w] ? 0 : 1;
      int num_tasks = 0, offset = 0;
      BlockScan(temp_storage).ExclusiveSum(unvisited, offset, num_tasks);
      // collect only the unvisited neighbors, and set them to be visited
      if (unvisited) {
        candidates[offset] = w;
        if (thread_lane == 0) is_visited[w] = 1;
      }
      // each warp computes the distance for one neighbor at a time
      /*for (int i = warp_lane; i < num_tasks; i += WARPS_PER_BLOCK) {
        auto vid = candidates[i];
        const Td *v_data = data_vectors + vid * dim;
        // use a warp to compute distance (SIMD)
        auto dist = cutils::compute_distance(dim, v_data, query);
        cand_distances[i] = dist;
      }*/
      // sort the candidates in their distances
      //cutils::sort_array_cta(num_tasks, candidates, cand_distances);
      for (size_t j = warp_lane; j < num_tasks; j += WARPS_PER_BLOCK) {
        auto vid = candidates[j];
        const float *v_data = data_vectors + vid * dim;
        auto dist = cutils::compute_distance(dim, v_data, query_data);
        auto bound = queues.get_tail_dist(wid);
        if (thread_lane == 0) count_dc[warp_lane] += 1;
        //auto dist = cand_distances[j];
        if (dist < bound) queues.push(wid, vid, dist);
      }
      auto r = queues.get_next_index(wid);
      if (r < idx) idx = r;
      else ++idx;
    }

    for (int i = thread_lane; i < K; i += WARP_SIZE) {
      results[wid * K + i] = queues.get_vid(wid, i);
    }
  }
  if (thread_lane == 0) atomicAdd(total_count_dc, count_dc[warp_lane]);
}

void kNN_search(int K, int qsize, int dim, size_t dsize,
                const float *queries,
                const float *data_vectors,
                result_t &results, char *index_file) {
  // load graph
  Graph g(index_file);

  assert(qsize < 65536);
  const int L = K * 5;
  assert(L <= MASTER_QUEUE_SIZE);
  //size_t memsize = cutils::print_device_info(0);

  // hyper-parameters
  auto degree = g.get_max_degree(); // assuming all nodes have the same degree
  assert(degree <= MAX_DEG);

  size_t num_threads = BLOCK_SIZE;
  int max_blocks_per_SM = maximum_residency(BestFirstSearch, num_threads, 0);
  std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  size_t nblocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  size_t num_blocks = nblocks > 65536 ? 65536 : nblocks;
  std::cout << "num_blocks = " << num_blocks << " num_threads = " << num_threads << "\n";
  int num_workers = num_blocks;

  // allocate device memory
  float* d_queries, *d_data;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_queries, qsize * dim * sizeof(float)));
  CUDA_SAFE_CALL(cudaMemcpy(d_queries, queries, qsize * dim * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_data, dsize * dim * sizeof(float)));
  CUDA_SAFE_CALL(cudaMemcpy(d_data, data_vectors, dsize * dim * sizeof(float), cudaMemcpyHostToDevice));
  uint8_t * is_visited;
  CUDA_SAFE_CALL(cudaMalloc((void **)&is_visited, num_workers * dsize * sizeof(uint8_t)));
  CUDA_SAFE_CALL(cudaMemset((void *)is_visited, 0, num_workers * dsize * sizeof(uint8_t)));
  vid_t * h_results = &results[0];
  vid_t * d_results;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_results, qsize * K * sizeof(vid_t)));
  CUDA_SAFE_CALL(cudaMemcpy(d_results, h_results, qsize * K * sizeof(vid_t), cudaMemcpyHostToDevice));
  gpu_long_t *d_total_count_dc;
  gpu_long_t total_count_dc = 0; 
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total_count_dc, sizeof(gpu_long_t)));
  CUDA_SAFE_CALL(cudaMemcpy(d_total_count_dc, &total_count_dc, sizeof(gpu_long_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  GraphGPU gg(g); // GPU graph 
  // each thread block takes one query 
  PQ_GPU<vid_t> queues(num_workers, MASTER_QUEUE_SIZE);

  cudaProfilerStart();
  Timer t;
  t.Start();
  BestFirstSearch<<<num_blocks, num_threads>>>(gg, K, qsize, dim, dsize, 
                                               d_queries, d_data, d_results, 
                                               is_visited, queues, d_total_count_dc);
  t.Stop();
  cudaProfilerStop();

  std::cout << "runtime [bfs_gpu] = " << t.Seconds() << " sec\n";
  std::cout << "throughput = " << double(qsize) / t.Seconds() << " queries per second (QPS)\n";
  CUDA_SAFE_CALL(cudaMemcpy(h_results, d_results, qsize * K * sizeof(vid_t), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(&total_count_dc, d_total_count_dc, sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_queries));
  CUDA_SAFE_CALL(cudaFree(d_data));
  CUDA_SAFE_CALL(cudaFree(d_results));
}
