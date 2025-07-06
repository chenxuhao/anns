#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include "utils.h"
#include "ctimer.h"

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] 
                  << " <format: bin|vecs> <xxx_base.fvecs> <xxx_query.fvecs> <groundtruth.ivecs>"
                  << " [k: 100] <xxx.index> [metric: l2|ip] [nc: 64]\n";
        return 1;
    }
    std::string format     = argv[1]; // bin or vecs
    const char* base_file  = argv[2]; // data vectors
    const char* query_file = argv[3]; // queries
    const char* gt_file    = argv[4]; // groundtruth
    // optional
    int topk = (argc >= 6) ? std::stoi(argv[5]) : 100;
    const char* index_file = (argc >= 7) ? argv[6] : nullptr;
    std::string metric = (argc >= 8) ? argv[7] : "l2";
    int nclusters = (argc >= 9) ? std::stoi(argv[8]) : 64;
    std::cout << "topk: "  << topk << "\n";
    std::cout << "Using " << metric << " distance metric\n";
    std::cout << "# clusters to search: "  << nclusters << "\n";

    size_t nb, nq, dim1, dim2;
    float* xb = read_fvecs(base_file, nb, dim1);
    float* xq = read_fvecs(query_file, nq, dim2);
    if (dim1 != dim2) {
        std::cerr << "Dimension mismatch\n";
        return 1;
    }

    faiss::IndexIVFFlat* index = nullptr;
    faiss::IndexFlatL2 quantizer(dim1);
    // Load or build index
    if (index_file && std::ifstream(index_file).good()) {
        std::cout << "Loading IVF index from: " << index_file << std::endl;
        auto base_index = faiss::read_index(index_file);
        index = dynamic_cast<faiss::IndexIVFFlat*>(base_index);
    } else {
        size_t nlists = lookup_nlist(nb);
        index = new faiss::IndexIVFFlat(&quantizer, dim1, nlists, faiss::METRIC_L2);
        // Train on a subset (ideally use a training set)
        size_t train_size = std::min(nb, std::max(nlists * 100, size_t(100000)));
        std::cout << "Training IVF index on " << train_size << " vectors..." << std::endl;
        index->train(train_size, xb);
        std::cout << "Adding vectors to index...\n";
        index->add(nb, xb);
        if (index_file) {
            std::cout << "Saving index to: " << index_file << std::endl;
            faiss::write_index(index, index_file);
        }
    }
    assert(nb == size_t(index->ntotal));
    assert(dim1 == size_t(index->d));
    std::cout << "Loaded " << nb << " vectors of dimension " << dim1 << std::endl;
    std::cout << "Total # clusters: " << index->nlist << "\n";
    std::cout << "Avg cluster size: " << nb / index->nlist << "\n";

    std::vector<faiss::idx_t> I(nq * topk);
    std::vector<float> D(nq * topk);

    index->nprobe = nclusters;
    assert(xq != nullptr);
    std::cout << "Searching...\n";
    ctimer_t t;
    ctimer_start(&t);
    index->search(nq, xq, topk, D.data(), I.data());
    ctimer_stop(&t);
    ctimer_measure(&t);
    ctimer_print(t, "faiss-search-ivf-flat");

    std::cout << "Checking results...\n";
    size_t gt_k = 0, nq_gt = 0;
    int* gt = read_ivecs(gt_file, nq_gt, gt_k);
    assert(nq == nq_gt);
    //std::cout << "gt_k: "  << gt_k << "\n";
    assert(size_t(topk) <= gt_k);
    float recall = compute_recall(I.data(), gt, nq, topk, gt_k);
    std::cout << "Recall@" << topk << ": " << recall << "\n";

    double search_time_sec = (double)timespec_nsec(t.elapsed) / 1e9;
    std::cout << "\n==== Search Statistics ====\n";
    std::cout << "nprobe (clusters per query): " << index->nprobe << "\n";
    std::cout << "Total time: " << search_time_sec * 1000 << " ms\n";
    std::cout << "Avg latency: " << (search_time_sec * 1000 / nq) << " ms/query\n";
    std::cout << "Throughput: " << (nq / search_time_sec) << " queries/sec\n";
    size_t est_dists = size_t(index->nprobe) * (index->ntotal / index->nlist) * nq;
    std::cout << "Estimated distance computations: " << est_dists << "\n\n";

    delete[] xb;
    delete[] xq;
    delete[] gt;
    delete index;
    return 0;
}
