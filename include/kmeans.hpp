#include <vector>
#include <cassert>

template <typename T>
class Kmeans {
  private:
  std::vector<int> membership;
  std::vector<T> centroids;
  std::vector<std::vector<int>> clusters;
  size_t npoints;
  int dim;
  int nclusters;
  const T *data;
  unsigned long long threshold;
  int max_iter;

  public:
  Kmeans(size_t np, int d, int nc, const T *ptr, 
         unsigned long long thr = 0, int iter = 500) :
      npoints(np),
      dim(d), 
      nclusters(nc),
      data(ptr),
      threshold(thr),
      max_iter(iter) {
    assert(size_t(nclusters) < npoints);
    membership.resize(npoints);
    std::fill(membership.begin(), membership.end(), 0);
    centroids.resize(int64_t(nclusters)*dim);
    // initialize centroids at first nclusters points
    for (int i = 0; i < nclusters; i++) {
      #pragma omp simd
      for (int j = 0; j < dim; j++)
        centroids[i*dim+j] = data[i*dim+j];
    }
  }
  T* cluster_cpu();
  T* cluster_gpu();
  std::vector<std::vector<int>> get_clusters() {
    clusters.resize(nclusters);
    for (size_t pt = 0; pt < npoints; ++pt) {
      auto cid = membership[pt];
      clusters[cid].push_back(pt);
    }
    return clusters;
  }
  std::vector<int> get_membership() {
    return membership;
  }
};
