#include <cuda.h>
#include <float.h>
#include "kmeans.hpp"
#include "kmeans_kernel.cuh"
#include "cutil_subset.cuh"

template <typename T>
T* Kmeans<T>::cluster_gpu() {
  std::vector<int> new_centers_len(nclusters, 0);
  std::vector<float> new_centers(nclusters * dim, 0.);
  std::vector<float> block_new_centers(nclusters*dim, 0.f);

  float *feature_flipped_d;/* original (not inverted) data array */
  cudaMalloc((void**) &feature_flipped_d, npoints*dim*sizeof(float));
  cudaMemcpy(feature_flipped_d, &data[0], npoints*dim*sizeof(float), cudaMemcpyHostToDevice);
  float *feature_d;/* inverted data array */
  cudaMalloc((void**) &feature_d, npoints*dim*sizeof(float));

  size_t num_threads = BLOCK_SIZE;
  size_t num_blocks = (npoints - 1) / num_threads + 1;
  invert_mapping<<<num_blocks,num_threads>>>(feature_flipped_d, feature_d, npoints, dim);

  int *membership_d; /* membership on the device */
  cudaMalloc((void**) &membership_d, npoints*sizeof(int));
  cudaMemset(membership_d, 0, npoints*sizeof(int));
  float *centroids_d; /* cluster centers on the device */
  cudaMalloc((void**) &centroids_d, nclusters*dim*sizeof(float));
  gpu_long_t *delta_d; /* per block calculation of deltas */
  cudaMalloc((void**)&delta_d, sizeof(int));

  //float *block_centroids_d; /* per block calculation of cluster centers */
  //cudaMalloc((void**) &block_centroids_d, num_blocks_perdim * num_blocks_perdim * nclusters * dim * sizeof(float));

  /* iterate until convergence */
  for (int iter = 0; iter < 500; iter ++) {
    cudaMemcpy(centroids_d, &centroids[0], nclusters*dim*sizeof(float), cudaMemcpyHostToDevice);
    find_closest_center<<<num_blocks, num_threads>>>(dim, npoints, nclusters, feature_d, membership_d, centroids_d, delta_d);
    //update_new_centers<<<grid, threads>>>(dim, npoints, nclusters, feature_flipped_d, membership_d, centroids_d, block_centroids_d);
    cudaThreadSynchronize();
    cudaMemcpy(&membership[0], membership_d, npoints*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_long_t delta = 0;
    cudaMemcpy(&delta, delta_d, sizeof(gpu_long_t), cudaMemcpyDeviceToHost);

    //float * block_clusters_h = (float *) malloc(num_blocks_perdim * num_blocks_perdim * nclusters * dim * sizeof(float));
    //cudaMemcpy(block_clusters_h, block_centroids_d, num_blocks_perdim * num_blocks_perdim * nclusters * dim * sizeof(float), cudaMemcpyDeviceToHost);

    //#pragma omp parallel for schedule(static)
    for (int i = 0; i < npoints; i++) {
      int cluster_id = membership[i];
      new_centers_len[cluster_id]++;
      for (int j = 0; j < dim; j++)
        new_centers[cluster_id*dim+j] += data[i*dim+j];
    }

    // replace old cluster centers with new_centers 
    #pragma omp parallel for
    for (int i = 0; i < nclusters; i++) {
      assert(new_centers_len[i] > 0);
      #pragma omp simd
      for (int j = 0; j < dim; j++) {
        centroids[i*dim+j] = new_centers[i*dim+j] / new_centers_len[i];
        new_centers[i*dim+j] = 0.;
      }
      new_centers_len[i] = 0;
    }
    if(delta < threshold) break;
  }
  cudaFree(feature_d);
  cudaFree(feature_flipped_d);
  cudaFree(membership_d);
  cudaFree(centroids_d);
  //cudaFree(block_centroids_d);
  cudaFree(delta_d);
  return &centroids[0];
}

template class Kmeans<float>;
