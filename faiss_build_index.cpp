#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/AutoTune.h>
#include <faiss/index_io.h>
#include <faiss/impl/io.h>
#include "utils.h"
#include "ctimer.h"

typedef float T;

void build_ivf_index(const T* xb, size_t dim, size_t num_points, const char* output_index_path, size_t nlist) {
    // Build a flat L2 quantizer
    faiss::IndexFlatL2 quantizer(dim);

    // IVF index (L2 metric, with flat quantizer)
    faiss::IndexIVFFlat index(&quantizer, dim, nlist, faiss::METRIC_L2);

    // Train on a subset (ideally use a training set)
    size_t train_size = std::min(num_points, std::max(nlist * 100, size_t(100000)));
    std::cout << "Training IVF index on " << train_size << " vectors..." << std::endl;
    index.train(train_size, xb);

    // Add all vectors to the index
    std::cout << "Adding " << num_points << " vectors to index..." << std::endl;
    index.add(num_points, xb);

    // Save index to disk
    std::cout << "Saving index to: " << output_index_path << std::endl;
    faiss::write_index(&index, output_index_path);

    delete[] xb;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <dataset.fvecs> <output.index> <nlist>\n";
        return 1;
    }
    const char* dataset_path = argv[1];
    const char* output_path = argv[2];
    size_t nlist = std::stoi(argv[3]);

    std::string format = "bin";
    size_t dim, num_points;
    T* xb;
    if (format == "bin") {
        xb = read_fbin(dataset_path, num_points, dim);
    } else
        xb = read_fvecs(dataset_path, num_points, dim);
    std::cout << "Loaded " << num_points << " vectors of dimension " << dim << std::endl;
    std::cout << "Avg cluster size: " << num_points / nlist << "\n";
    ctimer_t t;
    ctimer_start(&t);

    build_ivf_index(xb, dim, num_points, output_path, nlist);

    ctimer_stop(&t);
    ctimer_measure(&t);
    ctimer_print(t, "faiss-build-ivf-flat-index");

    return 0;
}

