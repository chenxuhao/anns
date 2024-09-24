#include "utils.hpp"
#include <cub/cub.cuh>
#include "graph_gpu.cuh"
#include "cutil_subset.cuh"
#include "cuda_profiler_api.h"
#include "cuda_launch_config.cuh"

// hyperparameters
#define BEAM_SIZE BLOCK_SIZE
#define MAX_DEG 32
#define NUM_START 128
#define LIMIT 10000
#define EXPAND_RATE 4 //BLOCK_SIZE/MAX_DEG
#define CANDIDATE_SIZE EXPAND_RATE*MAX_DEG
#define FRONTIER_SIZE ((BEAM_SIZE+CANDIDATE_SIZE-1)/BLOCK_SIZE+1)*BLOCK_SIZE
#define MC FRONTIER_SIZE/BLOCK_SIZE
#define MB BEAM_SIZE/BLOCK_SIZE
#define VISITED_SIZE 2*BEAM_SIZE // TODO: (BEAM_SIZE+EXPAND_RATE) is enough

//#define bits std::max<int>(10, std::ceil(std::log2(BEAM_SIZE * BEAM_SIZE)) - 2)
//#define bits 12
//#define HASH_FILTER_SIZE 1<<bits
#define HASH_FILTER_SIZE 4096

// hash filter
template <typename T>
__device__ bool has_been_seen(T a, T* filter) {
  int loc = cutils::myhash<T>(a) & (HASH_FILTER_SIZE - 1);
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
           vidType *filters,
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
  __shared__ vidType unvisited_frontier[EXPAND_RATE];
  __shared__ T ufr_dist[EXPAND_RATE];

  // maintains sorted set of visited vertices (id-distance pairs)
  __shared__ vidType visited[VISITED_SIZE];
  __shared__ T visited_dist[VISITED_SIZE];
  __shared__ int visited_size;

  auto candidates = frontier + BEAM_SIZE;
  auto cand_dists = fro_dist + BEAM_SIZE;

  // counters
  __shared__ uint64_t count_dc[WARPS_PER_BLOCK];
  __shared__ int num_visited, remain, num_hops;
 
  // for sorting
  typedef cub::BlockRadixSort<T, BLOCK_SIZE, MB, vidType> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;
  typedef cub::BlockRadixSort<T, BLOCK_SIZE, MC, vidType> BlockRadixSortC;
  __shared__ typename BlockRadixSortC::TempStorage temp_storageC;

  // initialize the hash filter
  //__shared__ vidType hash_filter[HASH_FILTER_SIZE];
  vidType *hash_filter = &filters[blockIdx.x * HASH_FILTER_SIZE];
  auto max_deg = g.get_max_degree();

  // initialize the counter
  if (thread_lane == 0) count_dc[warp_lane] = 0;

  // each thread block takes a query
  for (int qid = blockIdx.x; qid < qsize; qid += gridDim.x) {
    const T *q_data = queries + qid * dim;

    // initialize the frontier
    for (int j = 0; j < MC; j++) fro_dist[j*BLOCK_SIZE+threadIdx.x] = FLT_MAX;
    for (int j = 0; j < MC; j++) frontier[j*BLOCK_SIZE+threadIdx.x] = 0;
    for (int j = 0; j < 2; j++) visited_dist[j*BLOCK_SIZE+threadIdx.x] = FLT_MAX;
    for (int j = 0; j < 2; j++) visited[j*BLOCK_SIZE+threadIdx.x] = 0;

    // initialize the hash filter
    for (int i = threadIdx.x; i < HASH_FILTER_SIZE; i+=BLOCK_SIZE)
      hash_filter[i] = vidType(-1);
    __syncthreads();

    // insert starting points into frontier
    for (int i = warp_lane; i < NUM_START; i += WARPS_PER_BLOCK) {
      auto v = starting_points[i];
      auto *v_data = data_vectors + v * dim;
      // each warp computes a distance
      auto dist = cutils::compute_distance<T>(dim, v_data, q_data);
      if (thread_lane == 0) {
        count_dc[warp_lane] += 1;
        frontier[i] = v;
        fro_dist[i] = dist;
      }
    }
    __syncthreads();

    // sort the beam (in frontier)
    {
    T thread_key[MB];
    vidType thread_val[MB];
    for (int j = 0; j < MB; j++) {
      thread_key[j] = fro_dist[j+MB*threadIdx.x];
      thread_val[j] = frontier[j+MB*threadIdx.x];
    }
    BlockRadixSort(temp_storage).Sort(thread_key, thread_val);
    for (int j = 0; j < MB; j++) {
      fro_dist[j+MB*threadIdx.x] = thread_key[j];
      frontier[j+MB*threadIdx.x] = thread_val[j];
    }
    }
    __syncthreads();
    // deduplication
    if (threadIdx.x < BEAM_SIZE && frontier[threadIdx.x + 1] == frontier[threadIdx.x])
      fro_dist[threadIdx.x + 1] = FLT_MAX;
    __syncthreads();

    // get ready for expansion
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

    // The main loop. Terminate beam search when the entire frontier
    // has been visited or have reached max_visit.
    while (remain > 0 && num_visited < LIMIT) {
      //if (threadIdx.x == 0) cand_size = 0;
      if (threadIdx.x == 0) num_hops += 1;
      __syncthreads();

      //if (qid==0 && threadIdx.x == 0) printf("qid=%d, remain=%d\n", qid, remain);
      int num = remain;
      // start expansion
      num = EXPAND_RATE <= num ? EXPAND_RATE : num;
      __syncthreads();
      // each warp exapnd a node
      for (int i = warp_lane; i < num; i += WARPS_PER_BLOCK) {
        // the next node to expand is the unexpanded frontier node that is closest to q
        auto v = unvisited_frontier[i];
        auto v_dist = ufr_dist[i];
        auto v_ptr = g.N(v);
        // add to visited set
        if (thread_lane == 0) {
          visited[BEAM_SIZE+i] = v;
          visited_dist[BEAM_SIZE+i] = v_dist;
        }
        if (thread_lane < max_deg) candidates[max_deg*i+thread_lane] = v_ptr[thread_lane];
      }
      __syncthreads();

      // sort visited queue
      {
        T thread_key[2];
        vidType thread_val[2];
        for (int j = 0; j < 2; j++) {
          thread_key[j] = visited_dist[j+2*threadIdx.x];
          thread_val[j] = visited[j+2*threadIdx.x];
        }
        BlockRadixSortC(temp_storageC).Sort(thread_key, thread_val);
        for (int j = 0; j < 2; j++) {
          visited_dist[j+2*threadIdx.x] = thread_key[j];
          visited[j+2*threadIdx.x] = thread_val[j];
        }
      }
      __syncthreads();
 
      if (threadIdx.x == 0) {
        num_visited += num;
        if (visited_size+num <= BEAM_SIZE) visited_size += num;
        else visited_size = BEAM_SIZE;
      }
      //if (qid==0 && threadIdx.x == 0) printf("qid=%d, visited_size=%d\n", qid, visited_size);

      // Further filter on whether distance is greater than current
      // furthest distance in current frontier (if full).
      // T cutoff = ((frontier.size() < size_t(QP.beamSize))
      //              ? (T)std::numeric_limits<int>::max()
      //              : frontier[frontier.size() - 1].second);

      // each warp takes one neighbor, to compute distance
      for (auto e = warp_lane; e < max_deg*num; e += WARPS_PER_BLOCK) {
        auto u = candidates[e];
        // keep neighbors that have not been visited (using approximate hash).
        if (!has_been_seen(u, hash_filter)) {
          auto *u_data = data_vectors + u * dim;
          auto dist = cutils::compute_distance<T>(dim, u_data, q_data);
          if (thread_lane == 0) {
            count_dc[warp_lane] += 1;
            cand_dists[e] = dist;
            //cand_size += 1;
          }
        } else if (thread_lane == 0) cand_dists[e] = FLT_MAX;
      }
      //if (qid==0 && threadIdx.x == 0) printf("qid=%d, visited_size=%d, cand_size=%d\n", qid, visited_size, cand_size);
      //if (num_hops < 3 && qid==0 && threadIdx.x < cand_size) printf("qid=%d, cand[%d]=%d, dist=%f\n", qid, threadIdx.x, candidates[threadIdx.x], cand_dists[threadIdx.x]);

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

      // deduplication
      for (int j = threadIdx.x; j < FRONTIER_SIZE-1; j+=BLOCK_SIZE) {
        if (frontier[j+1] == frontier[j])
          fro_dist[j+1] = FLT_MAX;
      }
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
      //if (num_hops < 3 && qid==0 && threadIdx.x < 20) printf("qid=%d, frontier[%d]=%d, dist=%f\n", qid, threadIdx.x, frontier[threadIdx.x], fro_dist[threadIdx.x]);
 
      //if (qid==0 && threadIdx.x < visited_size) printf("qid=%d, visited[%d]=%d, frontier[%d]=%d\n", qid, threadIdx.x, visited[threadIdx.x], threadIdx.x, frontier[threadIdx.x]);
      // get the unvisited frontier (we only care about the first one) and update "remain"
      int ndiff = cutils::set_difference_cta<vidType, T>(BEAM_SIZE, frontier, fro_dist,        // set A
                                                         visited_size, visited, visited_dist,  // set B
                                                         EXPAND_RATE, unvisited_frontier, ufr_dist);        // set C = A - B
      if (threadIdx.x == 0) remain = ndiff;
      __syncthreads();
      if (threadIdx.x < ndiff && ufr_dist[threadIdx.x] == FLT_MAX) atomicSub(&remain, 1);
      //if (qid==0 && threadIdx.x < 1) printf("qid=%d, nhops=%d, visited_size=%d, remain=%d, next=%d\n", qid, num_hops, visited_size, remain, unvisited_frontier[0]);
    }
    for (int i = threadIdx.x; i < K; i += blockDim.x)
      results[qid * K + i] = frontier[i];
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
  assert(EXPAND_RATE <= BLOCK_SIZE);

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
  vidType *d_filters;
  //printf("filter size = %d bytes\n", num_blocks * HASH_FILTER_SIZE * sizeof(vidType));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_filters, num_blocks * HASH_FILTER_SIZE * sizeof(vidType)));

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

  Timer t;
  t.Start();
  cudaProfilerStart();
  BeamSearch<T><<<num_blocks, num_threads>>>(K, qsize, dim, npoints, 
                                             d_queries, d_data, d_results, 
                                             d_total_count_dc,
                                             d_starting_points,
                                             d_filters,
                                             gg);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  cudaProfilerStop();
  t.Stop();

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
  std::cout << "total # distance computation: " << total_count_dc << "\n";
  std::cout << "average # distance computation: " << total_count_dc / qsize << "\n";
}

template class ANNS<float>;
