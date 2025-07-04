#include <faiss/Index.h>
#include <faiss/index_io.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include "utils.h"
#include "ctimer.h"

struct NListEntry {
    size_t nb;
    size_t nlist;
};

const NListEntry nlist_table[] = {
    {1'000'000,     1024},     // sqrt(1e6) ≈ 1,000 → round to 1024
    {2'000'000,     2048},     // sqrt(2e6) ≈ 1414 → round to 2048
    {5'000'000,     2048},     // sqrt(5e6) ≈ 2236 → round to 2048
    {10'000'000,    4096},     // sqrt(1e7) ≈ 3162 → round to 4096
    {20'000'000,    4096},     // sqrt(2e7) ≈ 4472 → round to 4096
    {50'000'000,    8192},     // sqrt(5e7) ≈ 7071 → round to 8192
    {100'000'000,   16384},    // sqrt(1e8) ≈ 10,000 → round to 16K
    {200'000'000,   16384},    // sqrt(2e8) ≈ 14,142 → round to 16K
    {500'000'000,   32768},    // sqrt(5e8) ≈ 22,360
    {1'000'000'000, 32768},    // sqrt(1e9) ≈ 31,622 → round to 32K
};

inline int lookup_nlist(size_t nb) {
    for (const auto& entry : nlist_table) {
        if (nb <= entry.nb) {
            return entry.nlist;
        }
    }
    return 65536;
}

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

    size_t dim = 0, nq = 0, nb = 0;
    float* queries = nullptr, *xb = nullptr;
    faiss::IndexIVFFlat* index = nullptr;
    
    // Load or build index
    if (index_file && std::ifstream(index_file).good()) {
        std::cout << "Loading IVF index from: " << index_file << std::endl;
        auto base_index = faiss::read_index(index_file);
        index = dynamic_cast<faiss::IndexIVFFlat*>(base_index);
        if (index) {
            index->nprobe = nclusters;
            std::cout << "Set nprobe as " << nclusters << "\n";
        }
        nb = index->ntotal;
        dim = index->d;
    } else {
        std::cout << "Building IVF index from base: " << base_file << std::endl;
        if (format == "vecs")
            xb = read_vecs(base_file, nb, dim);
        else if (format == "bin")
            xb = read_bin(base_file, nb, dim);
        else {
            std::cerr << "Error: unsupported format \"" << format << "\". Use 'vecs' or 'bin'.\n";
            return 1;
        }
        size_t nlist = lookup_nlist(nb);
        faiss::IndexFlatL2 quantizer(dim);
        if (metric == "l2") {
            // IVF index (L2 metric, with flat quantizer)
            index = new faiss::IndexIVFFlat(&quantizer, dim, nlist, faiss::METRIC_L2);
        } else if (metric == "ip") {
            index = new faiss::IndexIVFFlat(&quantizer, dim, nlist, faiss::METRIC_INNER_PRODUCT);
        } else
            std::cerr << "Error: unsupported metric \"" << metric << "\". Use 'l2' or 'ip'.\n";

        // Train on a subset (ideally use a training set)
        size_t train_size = std::min(nb, std::max(nlist * 100, size_t(100000)));
        std::cout << "Training IVF index on " << train_size << " vectors..." << std::endl;
        index->train(train_size, xb);
        // Add all vectors to the index
        std::cout << "Adding " << nb << " vectors to index..." << std::endl;
        index->add(nb, xb);
        // Save index to disk
        if (index_file) {
            std::cout << "Saving index to: " << index_file << std::endl;
            faiss::write_index(index, index_file);
        }
        delete [] xb;
    }
    std::cout << "Loaded " << nb << " vectors of dimension " << dim << std::endl;
    std::cout << "Total # clusters: " << index->nlist << "\n";
    std::cout << "Avg cluster size: " << nb / index->nlist << "\n";

    // Load queries
    if (format == "bin") {
        queries = read_bin(query_file, nq, dim);
    } else if (format == "vecs") {
        queries = read_vecs(query_file, nq, dim);
    } else {
        std::cerr << "file format unsupported\n";
        return 1;
    }
    std::cout << "Num queries: "  << nq << "\n";
    // Allocate result arrays
    std::vector<faiss::idx_t> I(nq * topk);
    std::vector<float> D(nq * topk);

    ctimer_t t;
    ctimer_start(&t);
    index->search(nq, queries, topk, D.data(), I.data());
    ctimer_stop(&t);
    ctimer_measure(&t);
    ctimer_print(t, "faiss-search-ivf-flat");

    std::cout << "Loading groundtruth\n";
    int* groundtruth = nullptr;
    size_t gt_k = 0, nq_gt = 0;
    if (format == "bin") {
        groundtruth = read_ibin(gt_file, nq_gt, gt_k);
    } else if (format == "vecs") {
        groundtruth = read_ivecs(gt_file, nq_gt, gt_k);
    } else {
        std::cerr << "file format unsupported";
        return 1;
    }
    assert(nq == nq_gt);
    std::cout << "gt_k: "  << gt_k << "\n";
    assert(size_t(topk) <= gt_k);

    // Compute Recall@k
    float recall = compute_recall(I.data(), groundtruth, nq, topk, gt_k);
    std::cout << "Recall@" << topk << ": " << recall << std::endl;

    double search_time_sec = (double)timespec_nsec(t.elapsed) / 1e9;
    std::cout << "\n==== Search Statistics ====\n";
    std::cout << "nprobe (clusters per query): " << index->nprobe << "\n";
    std::cout << "Total time: " << search_time_sec * 1000 << " ms\n";
    std::cout << "Avg latency: " << (search_time_sec * 1000 / nq) << " ms/query\n";
    std::cout << "Throughput: " << (nq / search_time_sec) << " queries/sec\n";
    size_t est_dists = size_t(index->nprobe) * (index->ntotal / index->nlist) * nq;
    std::cout << "Estimated distance computations: " << est_dists << "\n\n";

    delete[] groundtruth;
    delete[] queries;
    delete index;
    return 0;
}
