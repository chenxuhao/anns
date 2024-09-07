#include "data_loader.hpp"
#include "utils.hpp"
#include <cassert>

data_loader::data_loader(int argc, char *argv[], std::string scheme) {
  double subset_size_millions = strtof(argv[4], nullptr); // number of vectors in millions
  vecdim = strtoull(argv[5], nullptr, 0); // dimention of vector
  qsize = strtoull(argv[6], nullptr, 0); // number of queries
  k = 100;//strtoull(argv[7], nullptr, 0); // k in k-NN
  dsize = (size_t)(subset_size_millions * 1000000 + 0.5);

  printf("Vector Search: dsize=%ld, qsize=%d, dim=%d, k=%d\n", dsize, qsize, vecdim, k);

  const char *path_d = argv[1]; // data vectors file path
  const char *path_q = argv[2]; // query file path
  const char *path_gt = argv[3]; // ground truth file path

  if (argc > 7) ef_lower_origin = strtoull(argv[7], nullptr, 0);
  if (argc > 8) ef_upper_origin = strtoull(argv[8], nullptr, 0);
  //printf("ef_lower=%d, ef_upper=%d\n", ef_lower_origin, ef_upper_origin);

  // Read data vectors in the vector database
  //printf("loading data vectors from %s ...\n", path_d);
  data_vectors.resize(dsize * vecdim);
  size_t num_vecs = 0;
  int dim = 0;
  utils::load_vectors<float>(num_vecs, dim, path_d, data_vectors.data());
  if(num_vecs != dsize) {
    fprintf(stderr, "\033[31mData vectors: Expected %ld vectors. Found %ld vectors\033[0m\n", dsize, num_vecs);
  }
  assert(num_vecs == dsize);
  assert(dim == vecdim);

  // Read queries
  //printf("loading queries from %s ...\n", path_q);
  queries.resize(qsize * vecdim);
  num_vecs = 0, dim = 0;
  utils::load_vectors<float>(num_vecs, dim, path_q, queries.data());
  if(num_vecs != size_t(qsize)) {
    fprintf(stderr, "\033[31mQuery vectors: Expected %d vectors. Found %ld vectors\033[0m\n", qsize, num_vecs);
  }
  assert(num_vecs == size_t(qsize));
  assert(dim == vecdim);

  // Read ground truth
  //printf("loading gt %s ...\n", path_gt);
  //load_ground_truth(k, qsize, path_gt);
  num_vecs = 0, dim = 0;
  ground_truth.resize(qsize * k);
  utils::load_vectors<vid_t>(num_vecs, dim, path_gt, ground_truth.data());
  assert(num_vecs == size_t(qsize));
  assert(dim == k);
}
/*
void data_loader::load_ground_truth(int k_ref, int num_queries, const char *filename) {
  std::ifstream fin(filename);
  if (!fin.is_open()) {
    fprintf(stderr, "Error: cannot open file %s\n", filename);
    exit(EXIT_FAILURE);
  }
  int t_query_num;
  int t_K;
  fin.read(reinterpret_cast<char *>(&t_query_num), sizeof(t_query_num));
  fin.read(reinterpret_cast<char *>(&t_K), sizeof(t_K));
  if (t_query_num != num_queries) {
    fprintf(stderr, "Error: t_query_num %d is not num_queries_ %d\n", t_query_num, num_queries);
    exit(EXIT_FAILURE);
  }
  if (t_K != k_ref) {
    fprintf(stderr, "Error: t_K %d is not %d.\n", t_K, k_ref);
    exit(EXIT_FAILURE);
  }
  ground_truth.resize(num_queries);
  for (int q_i = 0; q_i < t_query_num; ++q_i) {
    ground_truth[q_i].resize(t_K);
    for (int n_i = 0; n_i < t_K; ++n_i) {
      unsigned id;
      float dist;
      fin.read(reinterpret_cast<char *>(&id), sizeof(id));
      fin.read(reinterpret_cast<char *>(&dist), sizeof(dist));
      ground_truth[q_i][n_i] = id;
    }
  }
  fin.close();
}
*/
/*
void load_true_NN(const char *filename, int num_queries, std::vector<std::vector<int>> &true_nn_list) {
  std::ifstream fin(filename);
  if (!fin.is_open()) {
    fprintf(stderr, "Error: cannot open file %s\n", filename);
    exit(EXIT_FAILURE);
  }
  int t_query_num;
  int t_K;
  fin.read(reinterpret_cast<char *>(&t_query_num), sizeof(t_query_num));
  fin.read(reinterpret_cast<char *>(&t_K), sizeof(t_K));
  if (t_query_num < num_queries) {
    fprintf(stderr, "Error: t_query_num %u is smaller than num_queries_ %u\n", t_query_num, num_queries);
    exit(EXIT_FAILURE);
  }
  if (t_K < 100) {
    fprintf(stderr, "Error: t_K %u is smaller than 100.\n", t_K);
    exit(EXIT_FAILURE);
  }
  true_nn_list.resize(t_query_num);
  for (int q_i = 0; q_i < t_query_num; ++q_i) {
    true_nn_list[q_i].resize(t_K);
  }
  for (int q_i = 0; q_i < t_query_num; ++q_i) {
    for (int n_i = 0; n_i < t_K; ++n_i) {
      int id;
      float dist;
      fin.read(reinterpret_cast<char *>(&id), sizeof(id));
      fin.read(reinterpret_cast<char *>(&dist), sizeof(dist));
      true_nn_list[q_i][n_i] = id;
    }
  }
  fin.close();
}
*/
 
/*
void load_data(const char *filename, float *data, size_t &num, size_t &dim) {  // load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    fprintf(stderr, "Error: cannot open file %s\n", filename);
    exit(EXIT_FAILURE);
  }
  //    in.read((char*)&dim, 4);
  uint32_t t_d;
  in.read((char*) &t_d, 4);
  dim = (size_t) t_d;
  //    std::cout << "data dimension: " << dim << std::endl;
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = (fsize / (dim + 1) / 4);
  //    data = new float[static_cast<uint64_t>(num) * static_cast<uint64_t>(dim)];

  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();
}
//*/

