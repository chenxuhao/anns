#include "utils.hpp"
#include <cub/cub.cuh>
#include "graph_gpu.cuh"
#include "cutil_subset.cuh"
#include "cuda_profiler_api.h"
#include "cuda_launch_config.cuh"

// hyperparameters
#define BEAM_SIZE 128
#define M BEAM_SIZE/BLOCK_SIZE
#define MAX_DEG 32
#define NUM_START 64
#define LIMIT 1000000000

// hash filter
__device__ bool has_been_seen(vid_t v) {
  return false;
}

template <typename T>
__global__ void //__launch_bounds__(BLOCK_SIZE, 8)
BeamSearch(int K, int qsize, int dim, size_t npoints,
           const T* queries, const T* data_vectors,
           int *results,
           gpu_long_t* total_count_dc,
           vid_t *starting_points,
           GraphGPU<vid_t> g) {
  //int thread_id   = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  //int warp_id     = thread_id   / WARP_SIZE;               // global warp index
  int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
 
  __shared__ vid_t frontier[BEAM_SIZE];
  __shared__ vid_t unvisited_frontier[BEAM_SIZE];
  __shared__ vid_t visited[2*BEAM_SIZE];
  __shared__ vid_t new_frontier[BEAM_SIZE+MAX_DEG];
  __shared__ vid_t candidates[MAX_DEG];
  __shared__ float fro_dist[BEAM_SIZE];
  __shared__ float ufr_dist[BEAM_SIZE];
  __shared__ float visited_dist[2*BEAM_SIZE];
  __shared__ float nfr_dist[BEAM_SIZE+MAX_DEG];
  __shared__ float cand_dist[MAX_DEG];
  __shared__ uint64_t count_dc[WARPS_PER_BLOCK];
  __shared__ int num_visited, remain;
 
  // for sorting
  typedef cub::BlockRadixSort<float, BLOCK_SIZE, M, vid_t> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  // each thread block takes a query
  for (int qid = blockIdx.x; qid < qsize; qid += blockDim.x) {
    const float *q_data = queries + qid * dim;

    if (thread_lane == 0) count_dc[warp_lane] = 0;
    // insert nodes into frontier
    // each warp computes a distance
    for (int i = warp_lane; i < NUM_START; i += WARPS_PER_BLOCK) {
      auto v = starting_points[i];
      auto *v_data = data_vectors + v * dim;
      auto dist = cutils::compute_distance(dim, v_data, q_data);
      if (thread_lane == 0) {
        count_dc[warp_lane] += 1;
        frontier[i] = v;
        fro_dist[i] = dist;
      }
    }
    // sort frontier
    float thread_key[M];
    vid_t thread_val[M];
    for (int j = 0; j < M; j++) {
      thread_key[j] = frontier[j+M*threadIdx.x];
      thread_val[j] = fro_dist[j+M*threadIdx.x];
    }
    BlockRadixSort(temp_storage).Sort(thread_key, thread_val);
    for (int j = 0; j < M; j++) {
      frontier[j+M*threadIdx.x] = thread_key[j];
      fro_dist[j+M*threadIdx.x] = thread_val[j];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      unvisited_frontier[0] = frontier[0];
      num_visited = 0;
      remain = 1;
    }
    __syncthreads();

    while (remain > 0 && num_visited < LIMIT) {
      // the next node to visit is the unvisited frontier node that is closest to q
      auto v = unvisited_frontier[0];

      // add to visited set
      //visited.insert(std::upper_bound(visited.begin(), visited.end(), v, less), v);
      if (threadIdx.x == 0) num_visited++;
      __syncthreads();

      // keep neighbors that have not been visited (using approximate hash).
      // Note that if a visited node is accidentally kept due to approximate
      // hash it will be removed below by the union or will not bump anyone else.

      auto v_ptr = g.N(v);
      auto v_size = g.getOutDegree(v);

      // each warp takes one neighbor
      for (auto e = warp_lane; e < v_size; e += WARPS_PER_BLOCK) {
        auto u = v_ptr[e];
        if (has_been_seen(u)) continue;  // skip if already seen
        auto *u_data = data_vectors + u * dim;
        auto dist = cutils::compute_distance(dim, u_data, q_data);
        if (thread_lane == 0) {
          candidates[e] = u;
          cand_dist[e] = dist;
        }
      }
      // sort the candidates by distance from query

      // union the frontier and candidates into new_frontier, both are sorted
      // auto new_frontier_size = std::set_union(frontier, candidates, new_frontier, less);

      // trim to at most beam size
      // new_frontier_size = std::min<size_t>(QP.beamSize, new_frontier_size);

      // if a k is given (i.e. k != 0) then trim off entries that have a
      // distance greater than cut * current-kth-smallest-distance.

      // copy new_frontier back to the frontier
      // for (indexType i = 0; i < new_frontier_size; i++)
      //   frontier[i] = new_frontier[i];

      // get the unvisited frontier (we only care about the first one) and update "remain"
      // remain = std::set_difference(frontier, visited, unvisited_frontier, less);
    }
    for (int i = thread_lane; i < K; i += WARP_SIZE) {
      results[qid * K + i] = candidates[i];
    }
  }
  if (thread_lane == 0) atomicAdd(total_count_dc, count_dc[warp_lane]);
}

template <typename T>
void ANNS<T>::search(int k, int qsize, int dim, size_t npoints,
                     const T* queries, const T* data_vectors,
                     int *results, const char *index_file) {
  size_t memsize = cutils::print_device_info(0);
  Graph<vid_t> g(index_file);
  assert(g.max_degree() <= MAX_DEG);

  // GPU lauch configuration
  size_t num_threads = BLOCK_SIZE;
  int max_blocks_per_SM = maximum_residency(BeamSearch<T>, num_threads, 0);
  std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  size_t num_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  assert(num_blocks < 65536);
  std::cout << "num_blocks = " << num_blocks << " num_threads = " << num_threads << "\n";

  // allocate device memory
  T* d_queries, *d_data;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_queries, qsize * dim * sizeof(T)));
  CUDA_SAFE_CALL(cudaMemcpy(d_queries, queries, qsize * dim * sizeof(T), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_data, npoints * dim * sizeof(T)));
  CUDA_SAFE_CALL(cudaMemcpy(d_data, data_vectors, npoints * dim * sizeof(T), cudaMemcpyHostToDevice));

  int *h_results = &results[0];
  int *d_results;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_results, qsize * K * sizeof(int)));

  gpu_long_t *d_total_count_dc;
  gpu_long_t total_count_dc = 0; 
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total_count_dc, sizeof(gpu_long_t)));
  CUDA_SAFE_CALL(cudaMemcpy(d_total_count_dc, &total_count_dc, sizeof(gpu_long_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  GraphGPU<vid_t> gg(g); // GPU graph 

  vid_t *starting_points = new vid_t[NUM_START];
  vid_t *d_starting_points;
  for (int i = 0; i < NUM_START; i++) starting_points[i] = rand() % npoints;
  CUDA_SAFE_CALL(cudaMemcpy(d_starting_points, starting_points, NUM_START * sizeof(vid_t), cudaMemcpyHostToDevice));

  //cudaProfilerStart();
  Timer t;
  t.Start();
  BeamSearch<T><<<num_blocks, num_threads>>>(K, qsize, dim, npoints, 
                                          d_queries, d_data, d_results, 
                                          d_total_count_dc,
                                          d_starting_points,
                                          gg);
  t.Stop();
  //cudaProfilerStop();

  double runtime = t.Seconds();
  auto throughput = double(qsize) / runtime;
  auto latency = runtime / qsize * 1000.0;
  std::cout << "runtime [brute_force_gpu] = " << runtime << " sec\n";
  std::cout << "throughput = " << throughput << " queries per second (QPS)\n";
  //printf("avg latency: %f ms/query\n", latency);
  CUDA_SAFE_CALL(cudaMemcpy(h_results, d_results, qsize * K * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(&total_count_dc, d_total_count_dc, sizeof(gpu_long_t), cudaMemcpyDeviceToHost));
  std::cout << "average # distance computation: " << npoints << "\n";

  CUDA_SAFE_CALL(cudaFree(d_queries));
  CUDA_SAFE_CALL(cudaFree(d_data));
  CUDA_SAFE_CALL(cudaFree(d_results));
}

template class ANNS<float>;
