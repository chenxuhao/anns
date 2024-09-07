#include <cstring>
#include <algorithm>
#include "utils.hpp"

using namespace std;

template <typename T=float, typename CT=uint16_t>
class Quantizer {
private:
  int m; // number of sub-spaces
  int nclusters; // number of clusters in each sub-space
  int dim; // number of dimentions of each vector
  int sub_dim; // number of dimentions of each subvector
  size_t n; // number of vectors in the database
  vector<vector<CT>> codebook; // n * m; compressed type CT
  vector<T*> centroids; // m * ncluster * sub_dim
  vector<vector<float>> lookup_table; // m * ncluster

public:
  Quantizer(int m_, int nclusters_, int dim_, size_t n_, const T* data_vectors) :
    m(m_), nclusters(nclusters_), dim(dim_), n(n_) {
    assert(dim % m == 0);
    sub_dim = dim / m;
    printf("Quantizer: m=%d, nclusters=%d, dim=%d, n=%lu\n", m, nclusters, dim, n);
    codebook.resize(n);
    centroids.resize(m);
    lookup_table.resize(m);
    for (size_t i = 0; i < n; i++)
      codebook[i].resize(m);
    for (int i = 0; i < m; i++)
      lookup_table[i].resize(nclusters);

    int iterations = 200;
    printf("Train centroid ... \n");
    Timer t;
    t.Start();
    train_centroids(data_vectors, iterations);
    t.Stop();
    printf("time = %f\n", t.Seconds());

    //printf("Build codebook ... \n");
    //t.Start();
    //build_codebook(data_vectors);
    //t.Stop();
    //printf("time = %f\n", t.Seconds());
  }

  ~Quantizer() {
    for (int i = 0; i < m; i++) {
      delete [] centroids[i];
    }
  }

  void train_centroids(const T *data, int iterations = 200) {
    vector<T> subvectors(sub_dim*n);
    size_t threshold = n / 10000;
    //float threshold = 0;
    // for each subvector space
    for (int i = 0; i < m; i++) {
      // i-th sub-vector
      auto start = data + i*sub_dim;
      // gather the i-th subvector from each vector in the database
      for (size_t j = 0; j < n; j++)
        std::memcpy(&subvectors[j*sub_dim], start+j*dim, sub_dim*sizeof(T));
      std::vector<int> membership(n);
      // find the centroids in the i-th subvector space using k-means clustering
      centroids[i] = utils::kmean_cluster<float>(n, sub_dim, nclusters, threshold, &subvectors[0], membership, iterations);
      // update the codebook
      for (size_t j = 0; j < n; j++)
        codebook[j][i] = CT(membership[j]);
      subvectors.clear();
      for (int j = 0; j < 10; j++)
        printf("c[%d][%d]=%f\t", i, j, centroids[i][j]);
      printf("\n");
    }
  }

  void build_codebook(const T *data) {
    // for each data point
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
      // i-th data point
      auto data_i = data + i*dim;
      // for each sub-vector space
      for (int j = 0; j < m; ++j) {
        // j-th sub-vector
        auto subvector = data_i + j*sub_dim;
        // centroids in the j-th sub-space
        auto c_j = centroids[j];
        // find the closest centroid
        uint32_t bestIndex = 0;
        auto minDist = utils::compute_distance_squared(sub_dim, subvector, c_j);
        for (int k = 1; k < nclusters; ++k) {
          // k-th centroid
          c_j += sub_dim;
          auto dist = utils::compute_distance_squared(sub_dim, subvector, c_j);
          // is it closer?
          if (dist < minDist) {
            minDist = dist;
            bestIndex = k;
          }
        }
        codebook[i][j] = CT(bestIndex);
      }
      if (i > 10) continue;
      for (int j = 0; j < m; j++)
        printf("codebook[%ld][%d]=%d\t", i, j, codebook[i][j]);
      printf("\n");
    }
  }

  // build the lookup table given a query
  void build_lookup_table(const T * query) {
    // for each sub-vector space
    //#pragma omp parallel for
    for (int i = 0; i < m; i++) {
      // centroids in i-th sub-space
      auto c_i = centroids[i];
      // i-th sub-vector of the query
      auto q_i = query+i*sub_dim;
      // for each centroid
      for (int j = 0; j < nclusters; j++) {
        // j-th centroid in i-th sub-space
        auto c_ij = c_i + j*sub_dim;
        lookup_table[i][j] = utils::compute_distance_squared(sub_dim, q_i, c_ij);
      }
    }
  }

  float quantized_distance(size_t vec_id) {
    float distance = 0;
    // for each sub-vector space
    for (int i = 0; i < m; i++) {
      auto cid = codebook[vec_id][i];
      distance += lookup_table[i][cid];
    }
    return distance;
  }
};

