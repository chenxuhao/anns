#include "utils.h"
#include "ctimer.h"
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <cmath>

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " vecs/bin <xxx_base.fvecs> <xxx_query.fvecs> <output.ivecs> <k>\n";
        return 1;
    }

    std::string format = argv[1]; // 'vecs' or 'bin'
    const char* base_file = argv[2];
    const char* query_file = argv[3];
    const char* out_file = argv[4];
    size_t dim_b, nb, dim_q, nq;
    float* base, *queries;

    std::cout << "Loading base vectors...\n";
    if (format == "bin")
        base = read_fbin(base_file, dim_b, nb);
    else
        base = read_fvecs(base_file, dim_b, nb);
    std::cout << "Loading query vectors...\n";
    if (format == "bin")
        queries = read_fbin(query_file, dim_q, nq);
    else
        queries = read_fvecs(query_file, dim_q, nq);
    assert(dim_b == dim_q);
    int dim = dim_b;
    int k = atoi(argv[5]);

    std::cout << "Building exact search index...\n";
    faiss::IndexFlatL2 index(dim);
    index.add(nb, base);

    std::cout << "Running top-" << k << " brute-force search...\n";
    std::vector<faiss::idx_t> I(nq * k);
    std::vector<float> D(nq * k);
    index.search(nq, queries, k, D.data(), I.data());

    std::vector<int> result(nq * k);
    for (size_t i = 0; i < nq * k; ++i) {
        result[i] = static_cast<int>(I[i]);
    }

    std::cout << "Saving top-" << k << " ground truth to " << out_file << "\n";
    if (format == "bin")
        write_ibin(out_file, result.data(), k, nq);
    else
        write_ivecs(out_file, result.data(), k, nq);

    delete[] base;
    delete[] queries;
    return 0;
}
