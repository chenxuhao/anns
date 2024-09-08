#include <float.h>
#include "utils.hpp"
#include "distance.hpp"
using namespace std;

// everything currently only works for euclidean distance
template <typename T>
T* kmeans_cluster(size_t npoints, int dim, int nclusters, const T *data, 
                  vector<int> &membership) {
  assert(size_t(nclusters) < npoints);
  assert(membership.size() == npoints);
  T *centroids = new T[int64_t(nclusters)*dim];

  // initialize centroids at first nclusters points
  for (int i = 0; i < nclusters; i++) {
    #pragma omp simd
    for (int j = 0; j < dim; j++)
      centroids[i*dim+j] = data[i*dim+j];
  }

  int nthreads = 0;
  #pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }
  //cout << "kmeans " << nclusters << " clusters, " << nthreads << " threads" << endl;

  vector<int> new_centers_len(nclusters);
  vector<vector<float> > new_centers(nclusters,vector<float>(dim));
  vector<vector<int> > partial_new_centers_len(nthreads,vector<int>(nclusters));
  vector<vector<vector<float> > > partial_new_centers(nthreads,
      vector<vector<float> >(nclusters,vector<float>(dim)));

  // start clustering
  const int max_iter = 500;
  const int threshold = 0;
  for (int iter = 0; iter < max_iter; iter ++) {
    size_t delta = 0;
    float sumdist = 0;
    #pragma omp parallel for schedule(static) reduction(+:delta,sumdist)
    for (size_t pt = 0; pt < npoints; pt++) {
      int tid = omp_get_thread_num();
      // find the closest cluster center to point pt
      int index = 0;
      float min_dist = FLT_MAX;
      for (int j = 0; j < nclusters; j++) {
        auto dist = compute_distance_squared(dim, &data[pt*dim], &centroids[j*dim]);
        if (dist < min_dist) {
          min_dist = dist;
          index = j;
        }
      }
      sumdist += min_dist;
      if (membership[pt] != index) delta += 1;
      membership[pt] = index;
      partial_new_centers_len[tid][index]++;				
      for (int j = 0; j < dim; j++)
        partial_new_centers[tid][index][j] += data[pt*dim+j];
    }

    // let the main thread perform the array reduction
    for (int i = 0; i < nclusters; i++) {
      for (int j = 0; j < nthreads; j++) {
        new_centers_len[i] += partial_new_centers_len[j][i];
        partial_new_centers_len[j][i] = 0;
        for (int k = 0; k < dim; k++) {
          new_centers[i][k] += partial_new_centers[j][i][k];
          partial_new_centers[j][i][k] = 0;
        }
      }
    }

    // replace old cluster centers with new_centers
    for (int i = 0; i < nclusters; i++) {
      for (int j = 0; j < dim; j++) {
        if (new_centers_len[i] > 0)
          centroids[i*dim+j] = new_centers[i][j] / new_centers_len[i];
        new_centers[i][j] = 0;
      }
      //cout << new_centers_len[i] << " ";
      new_centers_len[i] = 0;
    }
    //cout << endl;
    if (delta <= threshold) {
      //cout << iter << "," << delta << "," << sumdist << endl;
      break;
    }
  }
  return centroids;
}

