#include <cub/cub.cuh>

#include "utils.hpp"
#include "common.hpp"
#include "cutil_subset.cuh"
#include "cuda_profiler_api.h"
#include "cuda_launch_config.cuh"

float* kmeans_cluster(size_t npoints, int dim, int nclusters,
                      const float *features, std::vector<int> &membership);
 
#define MP 2
#define MAX_NUM_CLUSTERS 128
#define MC (MAX_NUM_CLUSTERS/BLOCK_SIZE)

__global__ void //__launch_bounds__(BLOCK_SIZE, 8)
IVFsearch(int K, int qsize, int dim, size_t npoints,
          const float *queries, 
          const float *data_vectors,
          int *results, 
          gpu_long_t* total_count_dc,
          int nclusters,
          const float* centroids,
          const int* clusters,
          const int* cluster_sizes,
          int max_cluster_size) {
  //int thread_id   = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  //int warp_id     = thread_id   / WARP_SIZE;               // global warp index
  //int num_warps   = (BLOCK_SIZE / WARP_SIZE) * gridDim.x;  // total number of active warps
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA

  __shared__ uint64_t count_dc[WARPS_PER_BLOCK];
  __shared__ vidType candidates[BLOCK_SIZE*MP];
  __shared__ float distances[BLOCK_SIZE*MP];
  __shared__ vidType sorted_cids[MAX_NUM_CLUSTERS];
  __shared__ float c_dists[MAX_NUM_CLUSTERS];
  if (thread_lane == 0) count_dc[warp_lane] = 0;
  const int num_top_clusters = nclusters / 10;

  // for sorting: call cub library
  typedef cub::BlockRadixSort<float, BLOCK_SIZE, MC, vidType> BlockRadixSortC;
  __shared__ typename BlockRadixSortC::TempStorage temp_storageC;
  typedef cub::BlockRadixSort<float, BLOCK_SIZE, MP, vidType> BlockRadixSort;
  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  const int ROUNDS = (BLOCK_SIZE*MP - K) / WARPS_PER_BLOCK;
  const int NTASKS = ROUNDS * WARPS_PER_BLOCK;

  // each thread block takes a query
  for (int qid = blockIdx.x; qid < qsize; qid += gridDim.x) {
    const float *q_data = queries + qid * dim;

    // initialize centroid queue
    for (int i = threadIdx.x; i < nclusters; i += blockDim.x) {
      sorted_cids[i] = i;
      c_dists[i] = FLT_MAX;
    }
    // compute the distance between the query and centroids
    // each warp takes one centroid
    for (size_t cid = warp_lane; cid < nclusters; cid += WARPS_PER_BLOCK) {
      auto *c_data = centroids + cid * dim;
      auto dist = cutils::compute_distance(dim, q_data, c_data);
      if (thread_lane == 0) {
        count_dc[warp_lane] += 1;
        c_dists[cid] = dist;
      }
    }

    float c_key[MC];
    vidType c_val[MC];
    // sort the centroids by distance
    for (int j = 0; j < MC; j++) {
      c_key[j] = c_dists[j+MC*threadIdx.x];
      c_val[j] = sorted_cids[j+MC*threadIdx.x];
    }
    BlockRadixSortC(temp_storageC).Sort(c_key, c_val);
    for (int j = 0; j < MC; j++) {
      c_dists[j+MC*threadIdx.x] = c_key[j];
      sorted_cids[j+MC*threadIdx.x] = c_val[j];
    }
    __syncthreads();

    // start traversing the data points in the top clusters
    for (int i = 0; i < MP; i++) {
      distances[BLOCK_SIZE*i+threadIdx.x] = FLT_MAX;
    }
    __syncthreads();

    // the first (i.e. closest) cluster
    int cid = sorted_cids[0];
    auto cluster_0 = clusters + cid*max_cluster_size;
    auto num = cluster_sizes[cid] >= K ? K : cluster_sizes[cid];

    // insert the first K points in the first cluster
    for (size_t id = warp_lane; id < num; id += WARPS_PER_BLOCK) {
      auto pid = cluster_0[id];
      auto *p_data = data_vectors + pid * dim;
      auto dist = cutils::compute_distance(dim, p_data, q_data);
      if (thread_lane == 0) {
        count_dc[warp_lane] += 1;
        distances[id] = dist;
        candidates[id] = pid;
      }
    }
    __syncthreads();

    float thread_key[MP];
    vidType thread_val[MP];
    // insert the rest of the points in the first cluster
    if (cluster_sizes[cid] > K) {
      // each warp compares one point in the database
      for (size_t i = K+warp_lane; i < cluster_sizes[cid]; i += NTASKS) {
        for (int j = 0; j < ROUNDS; j++) {
          // in each rounds, every warp processes one data point
          auto pid = cluster_0[i + j * WARPS_PER_BLOCK];
          auto *p_data = data_vectors + pid * dim;
          auto dist = cutils::compute_distance(dim, p_data, q_data);
          if (thread_lane == 0) {
            count_dc[warp_lane] += 1;
            distances[warp_lane+K+j*WARPS_PER_BLOCK] = dist;
            candidates[warp_lane+K+j*WARPS_PER_BLOCK] = pid;
          }
        }
        __syncthreads();

        // sort the queue by distance
        for (int j = 0; j < MP; j++) {
          thread_key[j] = distances[j+MP*threadIdx.x];
          thread_val[j] = candidates[j+MP*threadIdx.x];
        }
        BlockRadixSort(temp_storage).Sort(thread_key, thread_val);
        for (int j = 0; j < MP; j++) {
          distances[j+MP*threadIdx.x] = thread_key[j];
          candidates[j+MP*threadIdx.x] = thread_val[j];
        }
        __syncthreads();
      }
    }

    // for the rest of the top clusters
    for (int ii = 1; ii < num_top_clusters; ii++) {
      cid = sorted_cids[ii];
      auto *cluster_i = clusters + cid*max_cluster_size;

      // each warp compares one point in the database
      for (size_t i = warp_lane; i < cluster_sizes[cid]; i += NTASKS) {
        for (int j = 0; j < ROUNDS; j++) {
          // in each rounds, every warp processes one data point
          auto pid = cluster_i[i + j * WARPS_PER_BLOCK];
          auto *p_data = data_vectors + pid * dim;
          auto dist = cutils::compute_distance(dim, p_data, q_data);
          if (thread_lane == 0) {
            count_dc[warp_lane] += 1;
            distances[warp_lane+K+j*WARPS_PER_BLOCK] = dist;
            candidates[warp_lane+K+j*WARPS_PER_BLOCK] = pid;
          }
        }
        __syncthreads();

        // sort the queue by distance
        for (int j = 0; j < MP; j++) {
          thread_key[j] = distances[j+MP*threadIdx.x];
          thread_val[j] = candidates[j+MP*threadIdx.x];
        }
        BlockRadixSort(temp_storage).Sort(thread_key, thread_val);
        for (int j = 0; j < MP; j++) {
          distances[j+MP*threadIdx.x] = thread_key[j];
          candidates[j+MP*threadIdx.x] = thread_val[j];
        }
        __syncthreads();
      }
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
  assert(npoints >= 10000);
  assert(K+WARPS_PER_BLOCK <= MP*BLOCK_SIZE);
  size_t memsize = cutils::print_device_info(0);

  // clustering the data points
  int nclusters = std::sqrt(npoints);
  assert(MAX_NUM_CLUSTERS >= nclusters);
  std::vector<int> membership(npoints, 0);
  auto centroids = kmeans_cluster(npoints, dim, nclusters, data_vectors, membership);
  std::vector<std::vector<int>> clusters(nclusters);
  for (size_t pt = 0; pt < npoints; ++pt) {
    auto cid = membership[pt];
    clusters[cid].push_back(pt);
  }
  std::vector<float> c_dist(nclusters);
  int max_cluster_size = 0;
  std::vector<int> cluster_sizes(nclusters);
  int cidx = 0;
  for (auto cluster : clusters) {
    cluster_sizes[cidx++] = cluster.size();
    if (cluster.size() > max_cluster_size)
      max_cluster_size = cluster.size();
  }
  std::cout << "kmeans done: num_clusters = " << nclusters << " max_cluster_size = " << max_cluster_size << "\n";

  // GPU lauch configuration
  size_t num_threads = BLOCK_SIZE;
  int max_blocks_per_SM = maximum_residency(IVFsearch, num_threads, 0);
  std::cout << "max_blocks_per_SM = " << max_blocks_per_SM << "\n";
  cudaDeviceProp deviceProp;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
  size_t num_blocks = max_blocks_per_SM * deviceProp.multiProcessorCount;
  assert(num_blocks < 65536);
  std::cout << "num_blocks = " << num_blocks << " num_threads = " << num_threads << "\n";

  // allocate device memory
  T* d_queries, *d_data, *d_centroids;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_queries, qsize * dim * sizeof(T)));
  CUDA_SAFE_CALL(cudaMemcpy(d_queries, queries, qsize * dim * sizeof(T), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_data, npoints * dim * sizeof(T)));
  CUDA_SAFE_CALL(cudaMemcpy(d_data, data_vectors, npoints * dim * sizeof(T), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_centroids, nclusters * dim * sizeof(T)));
  CUDA_SAFE_CALL(cudaMemcpy(d_centroids, centroids, nclusters * dim * sizeof(T), cudaMemcpyHostToDevice));

  int *h_results = &results[0];
  int *d_results;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_results, qsize * K * sizeof(int)));

  int *d_clusters, *d_cluster_sizes;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_clusters, nclusters * max_cluster_size * sizeof(int)));
  for (int i = 0; i < nclusters; i++) {
    CUDA_SAFE_CALL(cudaMemcpy(d_clusters + i * max_cluster_size, clusters[i].data(), clusters[i].size() * sizeof(int), cudaMemcpyHostToDevice));
  }
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_cluster_sizes, nclusters * sizeof(int)));
  CUDA_SAFE_CALL(cudaMemcpy(d_cluster_sizes, cluster_sizes.data(), nclusters * sizeof(int), cudaMemcpyHostToDevice));

  gpu_long_t *d_total_count_dc;
  gpu_long_t total_count_dc = 0; 
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total_count_dc, sizeof(gpu_long_t)));
  CUDA_SAFE_CALL(cudaMemcpy(d_total_count_dc, &total_count_dc, sizeof(gpu_long_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  //cudaProfilerStart();
  Timer t;
  t.Start();
  IVFsearch<<<num_blocks, num_threads>>>(K, qsize, dim, npoints, 
                                         d_queries, d_data, d_results, 
                                         d_total_count_dc,
                                         nclusters, d_centroids, d_clusters, d_cluster_sizes, max_cluster_size);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();
  //cudaProfilerStop();

  double runtime = t.Seconds();
  auto throughput = double(qsize) / runtime;
  auto latency = runtime / qsize * 1000.0;
  std::cout << "runtime [ivf_flat_gpu] = " << runtime << " sec\n";
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
