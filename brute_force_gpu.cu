#include "common.hpp"
#include "utils.hpp"
#include "pqueue.cuh"
#include "graph_gpu.cuh"
#include "cuda_profiler_api.h"
#include "cuda_launch_config.cuh"

typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;

__global__ void //__launch_bounds__(BLOCK_SIZE, 8)
BruteForceSearch(int K, int qsize, int dim, size_t npoints,
                 const float *queries, 
                 const float *data_vectors,
                 int *results, 
                 PQ_GPU<vid_t> queues, // priority queue
                 gpu_long_t* total_count_dc) {
  //int thread_id   = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  //int warp_id     = thread_id   / WARP_SIZE;               // global warp index
  int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA

  int wid = blockIdx.x; // worker id is the same as thread block id
  int L = queues.get_capacity();
  assert(MASTER_QUEUE_SIZE >= L);
  __shared__ vid_t candidates[BLOCK_SIZE];
  __shared__ float cand_distances[BLOCK_SIZE];
  __shared__ BlockScan::TempStorage temp_storage;
  __shared__ uint64_t count_dc[WARPS_PER_BLOCK];

  // each thread block takes a query
  for (int qid = blockIdx.x; qid < qsize; qid += blockDim.x) {
    const float *q_data = queries + qid * dim;
    // each warp compares one point in the database
    for (size_t pid = warp_lane; pid < npoints; pid += WARPS_PER_BLOCK) {
      auto *p_data = data_vectors + pid * dim;
      auto dist = cutils::compute_distance(dim, p_data, q_data);
      if (thread_lane == 0) count_dc[warp_lane] += 1;
      auto bound = queues.get_tail_dist(wid);
      if (dist < bound) queues.push(wid, pid, dist);
    }
    // write the top-k elements into results
    for (int i = threadIdx.x; i < K; i += blockIdx.x)
      results[qid * K + i] = queues.get_vid(wid, i);
  }
  if (thread_lane == 0) atomicAdd(total_count_dc, count_dc[warp_lane]);
}

template <typename T>
void ANNS<T>::search(int k, int qsize, int dim, size_t npoints,
                     const T* queries, const T* data_vectors,
                     int *results, const char *index_file) {
  assert(qsize < 65536);
  assert(k <= BLOCK_SIZE);
  //size_t memsize = cutils::print_device_info(0);

  // hyper-parameters
  const int L = K * 5;
  assert(L <= MASTER_QUEUE_SIZE);

  size_t num_threads = BLOCK_SIZE;
  int max_blocks_per_SM = maximum_residency(BruteForceSearch, num_threads, 0);
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
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_data, npoints * dim * sizeof(float)));
  CUDA_SAFE_CALL(cudaMemcpy(d_data, data_vectors, npoints * dim * sizeof(float), cudaMemcpyHostToDevice));
  int * h_results = &results[0];
  int * d_results;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_results, qsize * K * sizeof(int)));
  CUDA_SAFE_CALL(cudaMemcpy(d_results, h_results, qsize * K * sizeof(int), cudaMemcpyHostToDevice));
  gpu_long_t *d_total_count_dc;
  gpu_long_t total_count_dc = 0; 
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total_count_dc, sizeof(gpu_long_t)));
  CUDA_SAFE_CALL(cudaMemcpy(d_total_count_dc, &total_count_dc, sizeof(gpu_long_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  // each thread block takes one query 
  PQ_GPU<vid_t> queues(num_workers, MASTER_QUEUE_SIZE);

  cudaProfilerStart();
  Timer t;
  t.Start();
  BruteForceSearch<<<num_blocks, num_threads>>>(K, qsize, dim, npoints, 
                                                d_queries, d_data, d_results, 
                                                queues, d_total_count_dc);
  t.Stop();
  cudaProfilerStop();
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
