#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>

struct NListEntry {
    size_t nb;
    size_t nlist;
};

const NListEntry nlist_table[] = {
    {10'000,        128},      // sqrt(1e6) ≈ 1,000 → round to 1024
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

// Utility: read .fvecs format
float* read_fvecs(const char* filename, size_t& n_out, size_t& d_out) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open file");
    std::vector<float> vecs;
    int dim;
    n_out = 0;
    while (input.read((char*)&dim, 4)) {
        std::vector<float> buf(dim);
        input.read((char*)buf.data(), 4 * dim);
        vecs.insert(vecs.end(), buf.begin(), buf.end());
        ++n_out;
    }
    d_out = dim;
    float* data = new float[vecs.size()];
    std::memcpy(data, vecs.data(), vecs.size() * sizeof(float));
    return data;
}

// Utility: read .ivecs format (int32 ground truth)
int* read_ivecs(const char* filename, size_t& n_out, size_t& k_out) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open file");
    std::vector<int> vecs;
    int k;
    n_out = 0;
    while (input.read((char*)&k, 4)) {
        std::vector<int> buf(k);
        input.read((char*)buf.data(), 4 * k);
        vecs.insert(vecs.end(), buf.begin(), buf.end());
        ++n_out;
    }
    k_out = k;
    int* data = new int[vecs.size()];
    std::memcpy(data, vecs.data(), vecs.size() * sizeof(int));
    return data;
}

float compute_recall(const faiss::idx_t* pred, const int* gt, size_t nq, size_t topk, size_t gt_k) {
    size_t match = 0;
    for (size_t i = 0; i < nq; ++i) {
        for (size_t j = 0; j < topk; ++j) {
            for (size_t g = 0; g < gt_k; ++g) {
                if (pred[i * topk + j] == gt[i * gt_k + g]) {
                    ++match;
                    break;
                }
            }
        }
    }
    return float(match) / (nq * topk);
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

    size_t nb, nq, dim1, dim2;
    float* xb = read_fvecs(base_file, nb, dim1);
    float* xq = read_fvecs(query_file, nq, dim2);
    if (dim1 != dim2) {
        std::cerr << "Dimension mismatch\n";
        return 1;
    }
    size_t gt_k;
    int* gt = read_ivecs(gt_file, nq, gt_k);

    size_t nlists = lookup_nlist(nb);
    faiss::IndexFlatL2 quantizer(dim1);
    faiss::IndexIVFFlat index(&quantizer, dim1, nlists, faiss::METRIC_L2);

    // Train on a subset (ideally use a training set)
    size_t train_size = std::min(nb, std::max(nlists * 100, size_t(100000)));
    std::cout << "Training IVF index on " << train_size << " vectors..." << std::endl;
    index.train(train_size, xb);
    // Add all vectors to the index
    std::cout << "Adding " << nb << " vectors to index..." << std::endl;
    //index.train(nb, xb);
    std::cout << "Adding vectors...\n";
    index.add(nb, xb);
    std::cout << "Loaded " << nb << " vectors of dimension " << dim1 << std::endl;
    std::cout << "Total # clusters: " << index.nlist << "\n";
    std::cout << "Avg cluster size: " << nb / index.nlist << "\n";

    // Save index to disk
    if (index_file) {
        std::cout << "Saving index to: " << index_file << std::endl;
        faiss::write_index(&index, index_file);
    }

    std::vector<faiss::idx_t> I(nq * topk);
    std::vector<float> D(nq * topk);

    index.nprobe = nclusters;

    std::cout << "Searching...\n";
    index.search(nq, xq, topk, D.data(), I.data());

    float recall = compute_recall(I.data(), gt, nq, topk, gt_k);
    std::cout << "Recall@" << topk << ": " << recall << "\n";

    delete[] xb;
    delete[] xq;
    delete[] gt;
    return 0;
}
