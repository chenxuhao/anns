#include "utils.hpp"
#include <cub/cub.cuh>
#include "graph_gpu.cuh"
#include "cutil_subset.cuh"
#include "cuda_profiler_api.h"
#include "cuda_launch_config.cuh"

// hyperparameters
#define BEAM_SIZE BLOCK_SIZE
#define MAX_DEG 32
#define NUM_START BLOCK_SIZE
#define LIMIT 10000
#define EXPAND_RATE 1 //BLOCK_SIZE/MAX_DEG
#define CANDIDATE_SIZE EXPAND_RATE*MAX_DEG
#define FRONTIER_SIZE (BEAM_SIZE+CANDIDATE_SIZE)
#define MC FRONTIER_SIZE/BLOCK_SIZE
#define MB BEAM_SIZE/BLOCK_SIZE
#define VISITED_SIZE 2*BEAM_SIZE

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
  //int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
 
  __shared__ vidType frontier[FRONTIER_SIZE];
  __shared__ T fro_dist[FRONTIER_SIZE];

  // The subset of the frontier that has not been visited
  // Use the first of these to pick next vertex to visit.
  __shared__ vidType unvisited_frontier[BEAM_SIZE];
  __shared__ T ufr_dist[BEAM_SIZE];

  // maintains sorted set of visited vertices (id-distance pairs)
  __shared__ vidType visited[VISITED_SIZE];
  __shared__ T visited_dist[VISITED_SIZE];
  __shared__ int visited_size;

  // used as temporaries in the loop
  //__shared__ vidType new_frontier[BEAM_SIZE+MAX_DEG];
  //__shared__ T nfr_dist[BEAM_SIZE+MAX_DEG];

  auto candidates = frontier + BEAM_SIZE;
  auto cand_dists = fro_dist + BEAM_SIZE;

  // counters
  __shared__ uint64_t count_dc[WARPS_PER_BLOCK];
  __shared__ int num_visited, remain, cand_size, num_hops;
  // frontier_size 
 
  // for sorting
  typedef cub::BlockRadixSort<T, BLOCK_SIZE, MB, vidType> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;
  typedef cub::BlockRadixSort<T, BLOCK_SIZE, MC, vidType> BlockRadixSortC;
  __shared__ typename BlockRadixSortC::TempStorage temp_storageC;

  // fro PrefixSum
  typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;
  __shared__ BlockScan::TempStorage temp_storageS;

  // initialize the hash filter
  __shared__ vidType hash_filter[HASH_FILTER_SIZE];

  // each thread block takes a query
  for (int qid = blockIdx.x; qid < qsize; qid += gridDim.x) {
    if (threadIdx.x < 1) printf("qid=%d\n", qid);
    const T *q_data = queries + qid * dim;
    if (thread_lane == 0) count_dc[warp_lane] = 0;

    // initialize the frontier
    for (int j = 0; j < MC; j++) fro_dist[j+MB*threadIdx.x] = FLT_MAX;

    // initialize the hash filter
    for (int i = threadIdx.x; i < HASH_FILTER_SIZE; i+=BLOCK_SIZE)
      hash_filter[i] = vidType(-1);
    __syncthreads();

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
    __syncthreads();
    {
    // sort frontier
    T thread_key[MB];
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

    if (threadIdx.x < EXPAND_RATE) {
      unvisited_frontier[threadIdx.x] = frontier[threadIdx.x];
      ufr_dist[threadIdx.x] = fro_dist[threadIdx.x];
    }
    if (threadIdx.x == 0) {
      num_visited = 0;
      remain = EXPAND_RATE;
      visited_size = 0;
      num_hops = 0;
    }
    __syncthreads();

    // The main loop.  Terminate beam search when the entire frontier
    // has been visited or have reached max_visit.
    while (remain > 0 && num_visited < LIMIT) {
      auto num = EXPAND_RATE <= remain ? EXPAND_RATE : remain;
      if (threadIdx.x == 0) cand_size = 0;
      if (threadIdx.x == 0) num_hops += 1;
      __syncthreads();
      for (int i = 0; i < num; i++) {
        // the next node to expand is the unexpanded frontier node that is closest to q
        auto v = unvisited_frontier[i];
        auto v_dist = ufr_dist[i];
        ///*
        // add to visited set
        int vsize = visited_size;
        //if (qid == 135)
        auto position = cutils::upper_bound_cta(vsize, visited_dist, v_dist);
        //if (position > 0 && position >= vsize && (threadIdx.x < 128 || threadIdx.x == 127)) printf("qid=%d, nhops=%d, tid=%d, overflow: location=%d, length=%d\n", qid, num_hops, threadIdx.x, position, vsize);
        assert(position == 0 || position < vsize);
        assert(position < VISITED_SIZE);
        if (threadIdx.x == 0) {
          num_visited += 1;
          visited[position] = v;
          visited_dist[position] = v_dist;
          if (visited_size < VISITED_SIZE) visited_size += 1;
        }
        __syncthreads();
        //*/

        auto v_ptr = g.N(v);
        auto v_size = g.getOutDegree(v);

        // keep neighbors that have not been visited (using approximate hash).
        // Note that if a visited node is accidentally kept due to approximate
        // hash it will be removed below by the union or will not bump anyone else.
        for (auto e = threadIdx.x; e < v_size; e += BLOCK_SIZE) {
          auto u = v_ptr[e];
          int found = 1;
          // check each neighbor if it has been seen
          if (has_been_seen(u, hash_filter)) found = 0; // skip if already seen
          int position = 0, total_num = 0;
          BlockScan(temp_storageS).ExclusiveSum(found, position, total_num);
          if (found) candidates[cand_size+position] = u;
          if (threadIdx.x == 0) cand_size += total_num;
          __syncthreads();
        }
      }
///*
      // Further filter on whether distance is greater than current
      // furthest distance in current frontier (if full).
      // T cutoff = ((frontier.size() < size_t(QP.beamSize))
      //              ? (T)std::numeric_limits<int>::max()
      //              : frontier[frontier.size() - 1].second);

      // each warp takes one neighbor, to compute distance
      for (auto e = warp_lane; e < cand_size; e += WARPS_PER_BLOCK) {
        auto u = candidates[e];
        auto *u_data = data_vectors + u * dim;
        auto dist = cutils::compute_distance(dim, u_data, q_data);
        if (thread_lane == 0) {
          count_dc[warp_lane] += 1;
          cand_dists[e] = dist;
        }
      }

      // sort the candidates by distance from query
      T thread_key[MC];
      vidType thread_val[MC];
      for (int j = 0; j < MC; j++) {
        auto index = j+MC*threadIdx.x;
        thread_key[j] = fro_dist[index];
        thread_val[j] = frontier[index];
      }
      BlockRadixSortC(temp_storageC).Sort(thread_key, thread_val);
      for (int j = 0; j < MC; j++) {
        auto index = j+MC*threadIdx.x;
        fro_dist[index] = thread_key[j];
        frontier[index] = thread_val[j];
      }
      __syncthreads();
 
      // union the frontier and candidates into new_frontier, both are sorted
      //auto new_frontier_size = cutils::set_union(frontier_size, fro_dist, frontier, cand_size, cand_dists, candidates, nfr_dist, new_frontier);

      // trim to at most beam size
      //new_frontier_size = new_frontier_size > BEAM_SIZE ? BEAM_SIZE : new_frontier_size;

      // if a k is given (i.e. k != 0) then trim off entries that have a
      // distance greater than cut * current-kth-smallest-distance.
      // if (QP.k > 0 && new_frontier_size > QP.k)
      //   new_frontier_size = (std::upper_bound(new_frontier, new_frontier_size, QP.cut * nfr_dist[QP.k]);

      // copy new_frontier back to the frontier
      //if (threadIdx.x == 0) frontier_size = 0;
      //for (int i = threadIdx.x; i < new_frontier_size; i+=BLOCK_SIZE) {
      //  frontier[i] = new_frontier[i];
      //  fro_dist[i] = nfr_dist[i];
      //}

      // get the unvisited frontier (we only care about the first one) and update "remain"
      remain = cutils::set_difference_cta(BEAM_SIZE, frontier, fro_dist,        // set A
                                          visited_size, visited, visited_dist,  // set B
                                          unvisited_frontier, ufr_dist);        // set C = A - B
    //*/
    }
    for (int i = threadIdx.x; i < K; i += blockDim.x)
      results[qid * K + i] = candidates[i];
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
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_starting_points, NUM_START * sizeof(vidType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_starting_points, starting_points, NUM_START * sizeof(vidType), cudaMemcpyHostToDevice));

  //cudaProfilerStart();
  Timer t;
  t.Start();
  BeamSearch<T><<<num_blocks, num_threads>>>(K, qsize, dim, npoints, 
                                          d_queries, d_data, d_results, 
                                          d_total_count_dc,
                                          d_starting_points,
                                          gg);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();
  //cudaProfilerStop();

  double runtime = t.Seconds();
  CUDA_SAFE_CALL(cudaMemcpy(h_results, d_results, qsize * K * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(&total_count_dc, d_total_count_dc, sizeof(gpu_long_t), cudaMemcpyDeviceToHost));

  CUDA_SAFE_CALL(cudaFree(d_queries));
  CUDA_SAFE_CALL(cudaFree(d_data));
  CUDA_SAFE_CALL(cudaFree(d_results));

  auto throughput = double(qsize) / runtime;
  auto latency = runtime / qsize * 1000.0;
  std::cout << "runtime [bfs_search_gpu] = " << runtime << " sec\n";
  std::cout << "throughput = " << throughput << " queries per second (QPS)\n";
  std::cout << "average # distance computation: " << npoints << "\n";
}

template class ANNS<float>;
