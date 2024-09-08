#include <float.h>
#include "utils.hpp"
#include "distance.hpp"

template <typename T>
T* kmean_cluster(size_t npoints, int dim, int nclusters, size_t threshold,
                 const T* features, std::vector<int> &membership, int iterations) {
  assert(size_t(nclusters) < npoints);
  assert(membership.size() == npoints);
  T *centroids;
  if ((dim*sizeof(T)) % 32 == 0) {
    centroids = (T*)aligned_alloc(32, nclusters*dim*sizeof(T));
    std::cout << "centroids aligned\n";
  } else centroids = new T[nclusters * dim];
  // randomly pick cluster centers
  // srand(7);
  for (int i = 0; i < nclusters; i++) {
    //int n = (int)rand() % npoints;
    #pragma omp simd
    for (int j = 0; j < dim; j++)
      centroids[i*dim+j] = features[i*dim+j];
  }

  int nthreads = 0;
  #pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }
  //printf("num of threads = %d\n", nthreads);
  std::vector<int> new_centers_len(nclusters, 0); // [nclusters]: no. of points in each cluster
  std::vector<int> partial_new_centers_len(nthreads*nclusters, 0);
  std::vector<T> new_centers(nclusters * dim, 0.);
  std::vector<std::vector<std::vector<T>>> partial_new_centers(nthreads);
  for (int i = 0; i < nthreads; i++) {
    partial_new_centers[i].resize(nclusters);
    for (int j = 0; j < nclusters; j++) {
      partial_new_centers[i][j].resize(dim);
      std::fill(partial_new_centers[i][j].begin(), partial_new_centers[i][j].end(), 0.);
    }
  }

  // start clustering
  for (int iter = 0; iter < iterations; iter ++) {
    size_t delta = 0;
    #pragma omp parallel for schedule(static) reduction(+:delta)
    for (size_t pt = 0; pt < npoints; pt++) {
      int tid = omp_get_thread_num();
      // find the closest cluster center to point pt
      int index = 0;
      // TODO: fix it for an arbitrary type
      T min_dist = FLT_MAX;
      for (int j = 0; j < nclusters; j++) {
        auto dist = compute_distance_squared(dim, &features[pt*dim], &centroids[j*dim]);  /* no need square root */
        if (dist < min_dist) {
          min_dist = dist;
          index = j;
        }
      }
      if (membership[pt] != index) delta += 1;
      membership[pt] = index;
      partial_new_centers_len[tid*nclusters+index]++;				
      for (int j = 0; j < dim; j++)
        partial_new_centers[tid][index][j] += features[pt*dim+j];
    }

    /* let the main thread perform the array reduction */
    for (int i = 0; i < nclusters; i++) {
      for (int j = 0; j < nthreads; j++) {
        new_centers_len[i] += partial_new_centers_len[j*nclusters+i];
        partial_new_centers_len[j*nclusters+i] = 0;
        for (int k = 0; k < dim; k++) {
          new_centers[i*dim+k] += partial_new_centers[j][i][k];
          partial_new_centers[j][i][k] = 0.;
        }
      }
    }

    /* replace old cluster centers with new_centers */
    for (int i=0; i<nclusters; i++) {
      for (int j=0; j<dim; j++) {
        if (new_centers_len[i] > 0)
          centroids[i*dim+j] = new_centers[i*dim+j] / new_centers_len[i];
        new_centers[i*dim+j] = 0.;
      }
      new_centers_len[i] = 0;
    }
    if(delta < threshold) break;
  }
  return centroids;
}

template float* kmean_cluster(size_t npoints, int dim, int nclusters, size_t threshold,
                              const float* features, std::vector<int> &membership, int iterations);
 
//template int* kmean_cluster(size_t npoints, int dim, int nclusters, size_t threshold,
//                            const int* features, std::vector<int> &membership, int iterations);
 
