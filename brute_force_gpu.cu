#include <cub/cub.cuh>

#include "utils.hpp"
#include "common.hpp"
#include "cutil_subset.cuh"
#include "cuda_profiler_api.h"
#include "cuda_launch_config.cuh"

#define M 2

__global__ void //__launch_bounds__(BLOCK_SIZE, 8)
BruteForceSearch(int K, int qsize, int dim, size_t npoints,
                 const float *queries, 
                 const float *data_vectors,
                 int *results, 
                 gpu_long_t* total_count_dc) {
  //int thread_id   = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  //int warp_id     = thread_id   / WARP_SIZE;               // global warp index
  //int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA

  __shared__ uint64_t count_dc[WARPS_PER_BLOCK];
  __shared__ vidType candidates[BLOCK_SIZE*M];
  __shared__ float distances[BLOCK_SIZE*M];
  if (thread_lane == 0) count_dc[warp_lane] = 0;
 
  // for sorting
  typedef cub::BlockRadixSort<float, BLOCK_SIZE, M, vidType> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  int ROUNDS = (BLOCK_SIZE*M - K) / WARPS_PER_BLOCK;
  int NTASKS = ROUNDS * WARPS_PER_BLOCK;
  // each thread block takes a query
  for (int qid = blockIdx.x; qid < qsize; qid += gridDim.x) {
    const float *q_data = queries + qid * dim;
    for (int i = 0; i < M; i++) {
      distances[BLOCK_SIZE*i+threadIdx.x] = FLT_MAX;
      candidates[BLOCK_SIZE*i+threadIdx.x] = BLOCK_SIZE*i+threadIdx.x;
    }
    __syncthreads();
    // insert the first K points
    for (size_t pid = warp_lane; pid < K; pid += WARPS_PER_BLOCK) {
      auto *p_data = data_vectors + pid * dim;
      auto dist = cutils::compute_distance(dim, p_data, q_data);
      if (thread_lane == 0) {
        count_dc[warp_lane] += 1;
        distances[pid] = dist;
      }
    }
    __syncthreads();
    // sort the queue by distance
    float thread_key[M];
    vidType thread_val[M];
    // each warp compares one point in the database
    auto NUM = (npoints-K-1) / NTASKS + 1;
    for (size_t i = 0; i < NUM; i += 1) {
    //for (size_t i = K+warp_lane; i < npoints; i += NTASKS) {
      for (int j = 0; j < ROUNDS; j++) {
        // in each rounds, every warp processes one data point
        auto pid = K + warp_lane + i * NTASKS + j * WARPS_PER_BLOCK;
        if (pid < npoints) {
          auto *p_data = data_vectors + pid * dim;
          auto dist = cutils::compute_distance(dim, p_data, q_data);
          if (thread_lane == 0) {
            count_dc[warp_lane] += 1;
            distances[warp_lane+K+j*WARPS_PER_BLOCK] = dist;
            candidates[warp_lane+K+j*WARPS_PER_BLOCK] = pid;
          }
        }
      }
      __syncthreads();

      // sort the queue by distance
      for (int j = 0; j < M; j++) {
        thread_key[j] = distances[j+M*threadIdx.x];
        thread_val[j] = candidates[j+M*threadIdx.x];
      }
      BlockRadixSort(temp_storage).Sort(thread_key, thread_val);
      for (int j = 0; j < M; j++) {
        distances[j+M*threadIdx.x] = thread_key[j];
        candidates[j+M*threadIdx.x] = thread_val[j];
      }
      __syncthreads();
    }
    // write the top-k elements into results
    for (int i = threadIdx.x; i < K; i += blockDim.x)
      results[qid * K + i] = candidates[i];
  }
  if (thread_lane == 0) atomicAdd(total_count_dc, count_dc[warp_lane]);
}

template <typename T>
void ANNS<T>::search(int k, int qsize, int dim, size_t npoints,
                     const T* queries, const T* data_vectors,
                     int *results, const char *index_file) {
  assert(K+WARPS_PER_BLOCK <= M*BLOCK_SIZE);
  size_t memsize = cutils::print_device_info(0);

  // GPU lauch configuration
  size_t num_threads = BLOCK_SIZE;
  int max_blocks_per_SM = maximum_residency(BruteForceSearch, num_threads, 0);
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
  //CUDA_SAFE_CALL(cudaMemcpy(d_results, h_results, qsize * K * sizeof(int), cudaMemcpyHostToDevice));

  gpu_long_t *d_total_count_dc;
  gpu_long_t total_count_dc = 0; 
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total_count_dc, sizeof(gpu_long_t)));
  CUDA_SAFE_CALL(cudaMemcpy(d_total_count_dc, &total_count_dc, sizeof(gpu_long_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  //cudaProfilerStart();
  Timer t;
  t.Start();
  BruteForceSearch<<<num_blocks, num_threads>>>(K, qsize, dim, npoints, 
                                                d_queries, d_data, d_results, 
                                                d_total_count_dc);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
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
