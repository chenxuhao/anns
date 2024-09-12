#include "utils.hpp"
#include <cub/cub.cuh>
#include "graph_gpu.cuh"
#include "cutil_subset.cuh"
#include "cuda_profiler_api.h"
#include "cuda_launch_config.cuh"

// hyperparameters
#define BEAM_SIZE BLOCK_SIZE
#define MAX_DEG 32
#define NUM_START 64
#define MB BEAM_SIZE/BLOCK_SIZE
#define MC 1
#define M 1
#define LIMIT 1000000000

//#define bits std::max<int>(10, std::ceil(std::log2(BEAM_SIZE * BEAM_SIZE)) - 2)
#define bits 12
#define HASH_FILTER_SIZE 1<<bits

template <typename T>
__device__ T myhash(T key) {
  // A simple hash function (based on MurmurHash)
  key ^= key >> 16;
  key *= 0x85ebca6b;
  key ^= key >> 13;
  key *= 0xc2b2ae35;
  key ^= key >> 16;
  return key;
}

// hash filter
template <typename T>
__device__ bool has_been_seen(T a, T* filter) {
  int loc = myhash<T>(a) & (HASH_FILTER_SIZE - 1);
  if (filter[loc] == a) return true;
  filter[loc] = a;
  return false;
}

template <typename T>
__global__ void //__launch_bounds__(BLOCK_SIZE, 8)
BeamSearch(int K, int qsize, int dim, size_t npoints,
           const T* queries, const T* data_vectors,
           int *results,
           gpu_long_t* total_count_dc,
           vidType *starting_points,
           GraphGPU<vidType> g) {
  //int thread_id   = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  //int warp_id     = thread_id   / WARP_SIZE;               // global warp index
  int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
 
  __shared__ vidType frontier[BEAM_SIZE];
  __shared__ float fro_dist[BEAM_SIZE];

  // The subset of the frontier that has not been visited
  // Use the first of these to pick next vertex to visit.
  __shared__ vidType unvisited_frontier[BEAM_SIZE];
  __shared__ float ufr_dist[BEAM_SIZE];

  // maintains sorted set of visited vertices (id-distance pairs)
  __shared__ vidType visited[2*BEAM_SIZE];
  __shared__ float visited_dist[2*BEAM_SIZE];

  // used as temporaries in the loop
  __shared__ vidType new_frontier[BEAM_SIZE+MAX_DEG];
  __shared__ float nfr_dist[BEAM_SIZE+MAX_DEG];
  __shared__ vidType candidates[MAX_DEG];
  __shared__ float cand_dists[MAX_DEG];
  __shared__ vidType keep[MAX_DEG];

  // counters
  __shared__ uint64_t count_dc[WARPS_PER_BLOCK];
  __shared__ int num_visited, remain, frontier_size, cand_size, keep_size;
 
  // for sorting
  typedef cub::BlockRadixSort<float, BLOCK_SIZE, M, vidType> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  __shared__ vidType hash_filter[HASH_FILTER_SIZE];
  for (int i = threadIdx.x; i < HASH_FILTER_SIZE; i+=BLOCK_SIZE) {
    hash_filter[i] = vidType(-1);
  }

  // each thread block takes a query
  for (int qid = blockIdx.x; qid < qsize; qid += blockDim.x) {
    const float *q_data = queries + qid * dim;
    if (thread_lane == 0) count_dc[warp_lane] = 0;

    // insert starting points into frontier
    for (int i = warp_lane; i < NUM_START; i += WARPS_PER_BLOCK) {
      auto v = starting_points[i];
      auto *v_data = data_vectors + v * dim;
      // each warp computes a distance
      auto dist = cutils::compute_distance(dim, v_data, q_data);
      if (thread_lane == 0) {
        count_dc[warp_lane] += 1;
        frontier[i] = v;
        fro_dist[i] = dist;
      }
    }
    {
    // sort frontier
    float thread_key[MB];
    vidType thread_val[MB];
    for (int j = 0; j < MB; j++) {
      thread_key[j] = frontier[j+MB*threadIdx.x];
      thread_val[j] = fro_dist[j+MB*threadIdx.x];
    }
    BlockRadixSort(temp_storage).Sort(thread_key, thread_val);
    for (int j = 0; j < MB; j++) {
      frontier[j+MB*threadIdx.x] = thread_key[j];
      fro_dist[j+MB*threadIdx.x] = thread_val[j];
    }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      unvisited_frontier[0] = frontier[0];
      ufr_dist[0] = frontier[0];
      num_visited = 0;
      remain = 1;
      frontier_size = NUM_START;
    }
    __syncthreads();

    // The main loop.  Terminate beam search when the entire frontier
    // has been visited or have reached max_visit.
    while (remain > 0 && num_visited < LIMIT) {
      // the next node to visit is the unvisited frontier node that is closest to q
      auto v = unvisited_frontier[0];
      auto v_dist = ufr_dist[0];

      // add to visited set
      auto position = cutils::upper_bound(num_visited, visited_dist, visited, v_dist);
      visited[position] = v;
      visited_dist[position] = v_dist;

      if (threadIdx.x == 0) {
        cand_size = 0;
        keep_size = 0;
        num_visited++;
      }
      __syncthreads();

      auto v_ptr = g.N(v);
      auto v_size = g.getOutDegree(v);

      // keep neighbors that have not been visited (using approximate hash).
      // Note that if a visited node is accidentally kept due to approximate
      // hash it will be removed below by the union or will not bump anyone else.
      // keep.clear();
      // for (auto a : G.N(v)) {
      //   if (has_been_seen(a)) continue;  // skip if already seen
      //   keep.push_back(a);
      // }

      // Further filter on whether distance is greater than current
      // furthest distance in current frontier (if full).
      // T cutoff = ((frontier.size() < size_t(QP.beamSize))
      //              ? (T)std::numeric_limits<int>::max()
      //              : frontier[frontier.size() - 1].second);
      // each warp takes one neighbor
      for (auto e = warp_lane; e < v_size; e += WARPS_PER_BLOCK) {
        auto u = v_ptr[e];
        if (has_been_seen(u, hash_filter)) continue;  // skip if already seen
        auto *u_data = data_vectors + u * dim;
        auto dist = cutils::compute_distance(dim, u_data, q_data);
        count_dc[warp_lane] += 1;
        if (thread_lane == 0) {
          candidates[e] = u;
          cand_dists[e] = dist;
        }
      }

      // sort the candidates by distance from query
      float thread_key[MC];
      vidType thread_val[MC];
      for (int j = 0; j < MC; j++) {
        auto index = j+MC*threadIdx.x;
        if (index < MAX_DEG) {
          thread_key[j] = cand_dists[index];
          thread_val[j] = candidates[index];
        } else {
          thread_val[j] = FLT_MAX;
        }
      }
      BlockRadixSort(temp_storage).Sort(thread_key, thread_val);
      for (int j = 0; j < MC; j++) {
        auto index = j+MC*threadIdx.x;
        if (index < MAX_DEG) {
          cand_dists[index] = thread_key[j];
          candidates[index] = thread_val[j];
        }
      }
      __syncthreads();
 
      // union the frontier and candidates into new_frontier, both are sorted
      auto new_frontier_size = cutils::set_union(frontier_size, fro_dist, frontier, cand_size, cand_dists, candidates, nfr_dist, new_frontier);

      // trim to at most beam size
      new_frontier_size = new_frontier_size > BEAM_SIZE ? BEAM_SIZE : new_frontier_size;

      // if a k is given (i.e. k != 0) then trim off entries that have a
      // distance greater than cut * current-kth-smallest-distance.
      // if (QP.k > 0 && new_frontier_size > QP.k)
      //   new_frontier_size = (std::upper_bound(new_frontier, new_frontier_size, QP.cut * nfr_dist[QP.k]);

      // copy new_frontier back to the frontier
      if (threadIdx.x == 0) frontier_size = 0;
      for (int i = threadIdx.x; i < new_frontier_size; i+=BLOCK_SIZE) {
        frontier[i] = new_frontier[i];
        fro_dist[i] = nfr_dist[i];
      }

      // get the unvisited frontier (we only care about the first one) and update "remain"
      remain = cutils::set_difference(frontier_size, frontier, fro_dist,    // set A
                                      num_visited, visited, visited_dist,  // set B
                                      unvisited_frontier, ufr_dist);        // set C = A - B
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
  Graph<vidType> g(index_file);
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

  GraphGPU<vidType> gg(g); // GPU graph 

  vidType *starting_points = new vidType[NUM_START];
  vidType *d_starting_points;
  for (int i = 0; i < NUM_START; i++) starting_points[i] = rand() % npoints;
  CUDA_SAFE_CALL(cudaMemcpy(d_starting_points, starting_points, NUM_START * sizeof(vidType), cudaMemcpyHostToDevice));

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
