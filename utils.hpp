#pragma once

#include <cmath>
#include <chrono>
#include <vector>
#include <cassert>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <random>

#include <omp.h>
#include <cxxabi.h>
#include <unistd.h>
#include <immintrin.h>
#include <sys/resource.h>

#include "timer.hpp"

struct QueryParams{
  long k;
  long beamSize; 
  double cut;
  long limit;
  long degree_limit;
  QueryParams(long k, long Q, double cut, long limit, long dg) : k(k), beamSize(Q), cut(cut), limit(limit), degree_limit(dg) {}
  QueryParams() {}
};

template <typename T>
T* kmean_cluster(size_t npoints, int dim, int nclusters, size_t threshold,
                 const T* features, std::vector<int> &membership, int iterations = 500);
 
template <typename T>
class vector_dataset {
  public:
    size_t num;
    int dim;
    T *dptr;
    bool INNER_PRODUCT = false;

    vector_dataset() { num = dim = 0; }
    vector_dataset(const char *filename, size_t m = 0) {
      num = m; dim = 0;
      load_vectors(filename);
    }
    vector_dataset(size_t nq, int k) {
      num = nq;
      dim = k;
      dptr = new T[nq*k];
    }

    T *operator[](size_t i) { return &dptr[i*dim]; }
    T* data() { return dptr; }

    // fbin, ibin, u8bin
    void load_vectors(const char *filename) {
      //std::cout << filename << " ";
      std::ifstream in(filename, std::ios::binary);
      if (!in.is_open()) {
        fprintf(stderr, "Error: cannot open file %s\n", filename);
        exit(EXIT_FAILURE);
      }

      in.read((char*)&dim, 4);
      in.seekg(0, std::ios::end);
      std::ios::pos_type ss = in.tellg();
      size_t fsize = (size_t)ss;
      num = size_t(fsize / (dim + 1) / 4);

      //size_t nvecs = 0;
      //in.read((char*)&nvecs, 4);
      //if (num == 0) num = nvecs;

      //std::cout << " n = " << num << " dim = " << dim << " \t";

      if ((dim*sizeof(T)) % 32 == 0) {
      //if (0) {
        //std::cout << " aligned\n";
        dptr = (T*)aligned_alloc(32, num*dim*sizeof(T));
      } else {
        //std::cout << " unaligned\n";
        dptr = (T*)malloc(num*dim*sizeof(T));
      }

      in.seekg(0, std::ios::beg);
      for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(dptr + i * dim), dim * 4);
      }
      //for (size_t i = 0; i < num; i++) in.read((char*)(dptr+i*dim), sizeof(T)*dim);
      in.close();
    }
};

template <typename T>
class ANNS {
private:
  int K;
  int nqueries;
  int vecdim;
  size_t npoints;
public:
  ANNS(int k, int qsize, int dim, size_t dsize,
       const char* query_file, const char* data_file, 
       const char* gt_file, const char* out_file, const char *index) :
      K(k), nqueries(qsize), vecdim(dim), npoints(dsize) {
    int degree = 32;
    QueryParams QP(K, 5*K, 0, npoints, degree);
    vector_dataset<T> data_vectors(data_file);
    vector_dataset<T> queries(query_file);
    vector_dataset<int> gt_all(gt_file);
    vector_dataset<int> results(nqueries, k);
    Timer t;
    t.Start();
    search(K, nqueries, vecdim, npoints, queries.data(), data_vectors.data(), results.data(), index);
    t.Stop();
    auto runtime = t.Seconds();
    auto recall = compute_avg_recall_1D(results, gt_all);
    printf("total runtime: %f sec, recall: %f\n", runtime, recall);
  }

  void search(int k, int qsize, int dim, size_t npoints,
              const T* queries, const T* data_vectors,
              int *results, const char *index);

  inline double compute_avg_recall_1D(vector_dataset<int> &results, vector_dataset<int> &gt) {
    assert(results.num == gt.num);
    assert(results.dim <= gt.dim);
    size_t qsize = results.num;
    int K = results.dim;
    int64_t correct = 0;
    #pragma omp parallel for reduction(+:correct)
    for (size_t q_i = 0; q_i < qsize; ++q_i) {
      for (int top_i = 0; top_i < K; ++top_i) {
        auto true_id = gt[q_i][top_i];
        for (int n_i = 0; n_i < K; ++n_i) {
          if (results[q_i][n_i] == true_id) {
            correct ++;
            break;
          }
        }
      }
    }
    int64_t total = K * qsize;
    return double(correct) / double(total);
  }
};

