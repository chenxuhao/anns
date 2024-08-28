#include "timer.h"
#include "data_loader.hh"

void kNN_search(int k, int qsize, int dim, size_t npoints,
                const float *queries, const float *data_vectors,
                result_t &results);

int main(int argc, char *argv[]) {
  if (argc < 7) {
    fprintf(stderr, "Usage: %s <data_file> <query_file> <groundtrue_file> "
            "<size_in_millions> <dimension> <num_queries> <output_file> \n", argv[0]);
    exit(1);
  }
  data_loader dl(argc, argv);
  auto k = dl.get_k();
  auto qsize = dl.get_qsize();
  auto dsize = dl.get_dsize();
  auto vecdim = dl.get_vecdim();
  auto data_vectors = dl.get_data_vectors();
  auto queries = dl.get_queries();
  auto gt = dl.get_ground_truth();
  result_t results(qsize*k);
  Timer t;
  t.Start();
  kNN_search(k, qsize, vecdim, dsize, queries, data_vectors, results);
  t.Stop();
  auto runtime = t.Seconds();
  auto throughput = double(qsize) / runtime;
  auto latency = runtime / qsize * 1000.0;
  auto recall = compute_avg_recall_1D<vid_t>(k, qsize, results.data(), gt);
  printf("runtime: %f sec, recall: %f, avg latency: %f ms/query, throughput: %f queries/sec\n", runtime, recall, latency, throughput);
  return 0;
}

