#pragma once

#include <queue>
#include <random>
#include <cstdio>
#include <string>
#include <chrono>
#include <cassert>
#include <clocale>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <omp.h>
#include "common.hpp"

typedef long unsigned int label_t;
typedef std::priority_queue<std::pair<float, label_t> > heap_t;

class data_loader {
  private:
    int k;      // k of the top-k
    int qsize;  // number of queries
    size_t dsize;  // number of vectors in the database
    int vecdim; // dimention of vector
    int ef_lower_origin;
    int ef_upper_origin;
    std::vector<float> data_vectors;
    std::vector<float> queries;
    result_t ground_truth;

    void load_ground_truth(int k_ref, int num_queries, const char *filename);

  public:
    data_loader() {}
    data_loader(int argc, char *argv[], std::string scheme = "");
    ~data_loader() {}
    int get_k() { return k; }
    int get_qsize() { return qsize; }
    size_t get_dsize() { return dsize; }
    int get_vecdim() { return vecdim; }
    int get_ef_lower_origin() { return ef_lower_origin; }
    int get_ef_upper_origin() { return ef_upper_origin; }
    float *get_data_vectors() { return data_vectors.data(); }
    float *get_queries() { return queries.data(); }
    vid_t *get_ground_truth() { return ground_truth.data(); }
    template <typename T>
    void load_vectors(size_t &num, int &dim, const char *filename, T* data);
};

template <typename T>
double compute_avg_recall_1D(int K, int qsize, const T* results, const vid_t *ground_truth) {
  int64_t correct = 0;
  #pragma omp parallel for reduction(+:correct)
  for (int q_i = 0; q_i < qsize; ++q_i) {
    for (int top_i = 0; top_i < K; ++top_i) {
      auto true_id = ground_truth[q_i*K+top_i];
      for (int n_i = 0; n_i < K; ++n_i) {
        if (results[q_i*K + n_i] == true_id) {
          correct ++;
        }
      }
    }
  }
  int64_t total = K * qsize;
  return double(correct) / double(total);
}

/*
static inline double compute_avg_recall(int K, int qsize, const result_t &results, const result_t &ground_truth) {
  int64_t correct = 0;
  #pragma omp parallel for reduction(+:correct)
  for (int q_i = 0; q_i < qsize; ++q_i) {
    auto L = int(results[q_i].size());
    assert(L == K); // comment this when L != K
    for (int top_i = 0; top_i < K; ++top_i) {
      auto true_id = ground_truth[q_i][top_i];
      for (int n_i = 0; n_i < L; ++n_i) {
        if (results[q_i][n_i] == true_id) {
          correct ++;
        }
      }
    }
  }
  int64_t total = K * qsize;
  return double(correct) / double(total);
}
*/
/*
static inline double compute_avg_recall(int qsize, std::vector<heap_t> &results, const std::vector<heap_t> &ground_truth) {
  size_t correct = 0;
  size_t total = 0;
  for (int i = 0; i < qsize; i++) {
    heap_t gt(ground_truth[i]);
    std::unordered_set<label_t> g;
    total += gt.size();
    while (gt.size()) {
      g.insert(gt.top().second);
      gt.pop();
    }
    while (results[i].size()) {
      if (g.find(results[i].top().second) != g.end()) {
        correct ++;
      }
      results[i].pop();
    }
  }
  return 1.0f * double(correct) / double(total);
}
*/
template <typename idx_t>
void compute_recalls_for_all_queries_1D(int K, int num_queries, const idx_t *ground_truth, const idx_t *results,
                                        std::unordered_map<unsigned, double> &recalls) {
  recalls[1] = 0.0;
  recalls[5] = 0.0;
  recalls[10] = 0.0;
  recalls[20] = 0.0;
  recalls[50] = 0.0;
  recalls[100] = 0.0;
  for (int q_i = 0; q_i < num_queries; ++q_i) {
    for (int top_i = 0; top_i < K; ++top_i) {
      //auto true_id = ground_truth[q_i][top_i];
      auto true_id = ground_truth[q_i*K+top_i];
      for (int n_i = 0; n_i < K; ++n_i) {
        if (results[q_i * K + n_i] == true_id) {
          if (n_i < 1) recalls[1] += 1;
          if (n_i < 5) recalls[5] += 1;
          if (n_i < 10) recalls[10] += 1;
          if (n_i < 20) recalls[20] += 1;
          if (n_i < 50) recalls[50] += 1;
          if (n_i < 100) recalls[100] += 1;
        }
      }
    }
  }
  recalls[1] /= 1.0 * num_queries;
  recalls[5] /= 5.0 * num_queries;
  recalls[10] /= 10.0 * num_queries;
  recalls[20] /= 20.0 * num_queries;
  recalls[50] /= 50.0 * num_queries;
  recalls[100] /= 100.0 * num_queries;
}
/*
template <typename gt_t = vid_t, typename res_t = vid_t>
void compute_recalls_for_all_queries(int K, int num_queries,
                                     const std::vector<std::vector<gt_t>> &ground_truth,
                                     const std::vector<std::vector<res_t>> &results,
                                     std::unordered_map<unsigned, double> &recalls) {
  if (ground_truth[0].size() < 100) {
    fprintf(stderr, "Error: Number of true nearest neighbors of a query is smaller than 100.\n");
    exit(EXIT_FAILURE);
  }
  recalls[1] = 0.0;
  recalls[5] = 0.0;
  recalls[10] = 0.0;
  recalls[20] = 0.0;
  recalls[50] = 0.0;
  recalls[100] = 0.0;

  for (int q_i = 0; q_i < num_queries; ++q_i) {
    for (int top_i = 0; top_i < 100; ++top_i) {
      auto true_id = ground_truth[q_i][top_i];
      for (int n_i = 0; n_i < K; ++n_i) {
        if (results[q_i][n_i] == true_id) {
          if (n_i < 1) recalls[1] += 1;
          if (n_i < 5) recalls[5] += 1;
          if (n_i < 10) recalls[10] += 1;
          if (n_i < 20) recalls[20] += 1;
          if (n_i < 50) recalls[50] += 1;
          if (n_i < 100) recalls[100] += 1;
        }
      }
    }
  }
  recalls[1] /= 1.0 * num_queries;
  recalls[5] /= 5.0 * num_queries;
  recalls[10] /= 10.0 * num_queries;
  recalls[20] /= 20.0 * num_queries;
  recalls[50] /= 50.0 * num_queries;
  recalls[100] /= 100.0 * num_queries;
}
*/
static inline void save_result(char* filename, std::vector<std::vector<unsigned> >& results) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  for (size_t i = 0; i < results.size(); i++) {
    auto GK = (unsigned)results[i].size();
    out.write((char*)&GK, sizeof(unsigned));
    out.write((char*)results[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

