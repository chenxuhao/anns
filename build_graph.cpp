#include "graph.hpp"
#include "utils.hpp"
#include "data_loader.hpp"

void build_knn_graph(int k, size_t N, size_t dim, float* data_points, Graph &g);
void build_approx_knn_graph(int k, size_t N, size_t dim, float* data_points, Graph &g,
  double rho = 1.0, double delta = 0.001);
void pruning(Graph &g);

void usage(int argc, char *argv[]) {
  if (argc < 4) {
    fprintf(stderr, "Usage: %s "
            "<k> "
            "<size_in_millions> "
            "<dimension> "
            "<data_file> "
            "<output_file> \n"
            , argv[0]);
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]) {
  usage(argc, argv);
  int k = strtoull(argv[1], nullptr, 0); // k in k-NN
  double subset_size_millions = strtof(argv[2], nullptr); // number of vectors in millions
  int vecdim = strtoull(argv[3], nullptr, 0); // dimention of vector
  size_t dsize = (size_t)(subset_size_millions * 1000000 + 0.5);
  printf("Vector Search: dsize=%ld, dim=%d, k=%d\n", dsize, vecdim, k);

  // Read data vectors in the vector database
  const char *path_d = argv[4]; // data vectors file path
  printf("Load data vectors from %s ...\n", path_d);
  std::vector<float> data_vectors(dsize * vecdim);

  size_t num_vecs = 0;
  int dim = 0;
  utils::load_vectors(num_vecs, dim, path_d, data_vectors.data());
  assert(num_vecs == dsize);
  assert(dim == vecdim);

  Timer t;
  t.Start();
  Graph g;
  //printf("Build directed kNN graph\n");
  //build_knn_graph(k, dsize, vecdim, data_vectors.data(), g);
  //printf("Done building directed kNN graph\n");
  printf("Build approx directed kNN graph\n");
  build_approx_knn_graph(k, dsize, vecdim, data_vectors.data(), g);
  printf("Done building approx directed kNN graph\n");
  // symmetrize the graph (bidirected)
  g.symmetrize();
  pruning(g);
  t.Stop();
  std::string out_file = "../data/sift1m/";
  if (argc > 5) out_file = argv[5];
  printf("Writing %s to disk file\n", out_file.c_str());
  g.write_to_file(out_file);
  printf("Done writing to disk file\n");

  ofstream meta(out_file+".meta.txt");
  meta << g.V() << endl;
  meta << g.E() << endl;
  meta << "4 8 1 4" << endl;
  meta << "32" << endl;
  meta << "0" << endl;
  meta << "0" << endl;
  meta << "0" << endl;

  return 0;
}

