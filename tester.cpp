#include "utils.hpp"

int main(int argc, char *argv[]) {
  if (argc < 7) {
    fprintf(stderr, "Usage: %s <data_file> <query_file> <groundtrue_file> "
            "<size_in_millions> <dimension> <num_queries> <output_file> \n", argv[0]);
    exit(1);
  }
  double subset_size_millions = strtof(argv[4], nullptr); // number of vectors in millions
  int vecdim = strtoull(argv[5], nullptr, 0); // dimention of vector
  int qsize = strtoull(argv[6], nullptr, 0); // number of queries
  int k = 100;//strtoull(argv[7], nullptr, 0); // k in k-NN
  size_t dsize = (size_t)(subset_size_millions * 1000000 + 0.5);

  printf("Vector Search: dsize=%ld, qsize=%d, dim=%d, k=%d\n", dsize, qsize, vecdim, k);

  const char *path_d = argv[1]; // data vectors file path
  const char *path_q = argv[2]; // query file path
  const char *path_gt = argv[3]; // ground truth file path
  const char *path_out = argv[7]; // output file path
  const char *path_idx = argv[8]; // index file path
  ANNS<float> anns(k, qsize, vecdim, dsize, path_q, path_d, path_gt, path_out, path_idx);
}

