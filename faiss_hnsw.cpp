#include <faiss/Index.h>
#include <faiss/index_io.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <string>
#include "utils.h"
#include "ctimer.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <format: bin|vecs> <base_path> <query_path> <gt_path>"
                  << " [k: 100] <xxx.index> [metric: l2|ip] [beam_size: 128] [degree: 32]\n";
        return 1;
    }

    std::string format = argv[1];
    std::string base_path = argv[2];
    std::string query_path = argv[3];
    std::string gt_file = argv[4];
    // optional
    int topk = (argc >= 6) ? std::stoi(argv[5]) : 100;
    const char* index_path = (argc >= 7) ? argv[6] : nullptr;
    std::string metric = (argc >= 8) ? argv[7] : "l2";
    int degree = (argc >= 9) ? std::stoi(argv[8]) : 32; // M is the number of neighbors per node
    int beam_size = (argc >= 10) ? std::stoi(argv[9]) : 128;
    std::cout << "Search topk: "  << topk << "\n";
    std::cout << "Using " << metric << " distance metric\n";
    std::cout << "Beam size: "  << beam_size << "\n";
    std::cout << "Graph degree: "  << degree << "\n";

    size_t nb = 0, dim = 0, nq = 0;
    float* xb = nullptr, *xq = nullptr;
    uint8_t* xb_u8 = nullptr, *xq_u8 = nullptr;

    faiss::IndexHNSWFlat* index = nullptr;

    // Load or build index
    if (index_path && std::ifstream(index_path).good()) {
        std::cout << "Loading HNSW index from: " << index_path << std::endl;
        faiss::Index* loaded = faiss::read_index(index_path);
        index = dynamic_cast<faiss::IndexHNSWFlat*>(loaded);
        if (!index) {
            std::cerr << "Error: loaded index is not IndexHNSWFlat!" << std::endl;
            return 1;
        }
        dim = index->d;
        nb = index->ntotal;
    } else {
        std::cout << "Building HNSW index from base: " << base_path << std::endl;
        if (format == "bvecs") {
            xb_u8 = read_vecs<uint8_t>(base_path.c_str(), nb, dim);
            xb = new float[nb * dim];
            for (size_t i = 0; i < nb * dim; ++i) xb[i] = static_cast<float>(xb_u8[i]);
        } else if (format == "vecs") {
            xb = read_vecs(base_path.c_str(), nb, dim);
        } else if (format == "bin") {
            xb = read_bin(base_path.c_str(), nb, dim);
        } else if (format == "u8bin") {
            xb_u8 = read_bin<uint8_t>(base_path.c_str(), nb, dim);
            xb = new float[nb * dim];
            for (size_t i = 0; i < nb * dim; ++i) xb[i] = static_cast<float>(xb_u8[i]);
        } else {
            std::cerr << "Error: unsupported format \"" << format << "\". Use 'vecs' or 'bin'.\n";
            return 1;
        }
        if (metric == "l2") {
            index = new faiss::IndexHNSWFlat(dim, degree, faiss::METRIC_L2);
        } else if (metric == "ip") {
            index = new faiss::IndexHNSWFlat(dim, degree, faiss::METRIC_INNER_PRODUCT);
        } else
            std::cerr << "Error: unsupported metric \"" << metric << "\". Use 'l2' or 'ip'.\n";
        index->hnsw.efConstruction = 200;
        index->verbose = true;
        index->add(nb, xb);
        if (index_path) {
            std::cout << "Saving index to: " << index_path << std::endl;
            faiss::write_index(index, index_path);
        }
        delete [] xb_u8;
        delete [] xb;
    }
    std::cout << "Loaded " << nb << " vectors of dimension " << dim << std::endl;

    // Load queries
    if (format == "bvecs") {
        xq_u8 = read_vecs<uint8_t>(query_path.c_str(), nq, dim);
        xq = new float[nq * dim];
        for (size_t i = 0; i < nq * dim; ++i) xq[i] = static_cast<float>(xq_u8[i]);
    } else if (format == "vecs")
        xq = read_vecs(query_path.c_str(), nq, dim);
    else if (format == "bin")
        xq = read_bin(query_path.c_str(), nq, dim);
    else if (format == "u8bin") {
        xq_u8 = read_bin<uint8_t>(query_path.c_str(), nq, dim);
        xq = new float[nq * dim];
        for (size_t i = 0; i < nq * dim; ++i) xq[i] = static_cast<float>(xq_u8[i]);
    } else {
        std::cerr << "Error: unsupported format \"" << format << "\". Use 'vecs' or 'bin'." << std::endl;
        return 1;
    }
    std::cout << "Num queries: "  << nq << "\n";
    // Allocate result arrays
    std::vector<faiss::idx_t> I(nq * topk);
    std::vector<float> D(nq * topk);

    index->hnsw.efSearch = beam_size;
    ctimer_t t;
    ctimer_start(&t);
    index->search(nq, xq, topk, D.data(), I.data());
    ctimer_stop(&t);
    ctimer_measure(&t);
    ctimer_print(t, "faiss-search-hnsw");

    // load groundtruth
    int* groundtruth;
    size_t gt_k, nq_gt;
    if (format == "bin" || format == "u8bin") {
        groundtruth = read_ibin(gt_file.c_str(), nq_gt, gt_k);
    } else if (format == "vecs" || format == "bvecs") {
        groundtruth = read_ivecs(gt_file.c_str(), nq_gt, gt_k);
    } else {
        std::cerr << "file format unsupported\n";
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
    std::cout << "Total time: " << search_time_sec * 1000 << " ms\n";
    std::cout << "Avg latency: " << (search_time_sec * 1000 / nq) << " ms/query\n";
    std::cout << "Throughput: " << (nq / search_time_sec) << " queries/sec\n";
    size_t est_dists = nq * index->hnsw.efSearch;
    std::cout << "Estimated distance computations: " << est_dists << "\n\n";

    delete[] xq;
    if (xb) delete[] xb;
    delete index;

    return 0;
}
