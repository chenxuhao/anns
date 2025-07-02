#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>

typedef int64_t idx_t;

template <typename T = float>
T* read_vecs(const char* filename, size_t& n_out, size_t& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open fvecs file");

    int d;
    input.read((char*)&d, sizeof(int));
    if (d <= 0 || d > 10000) {
        throw std::runtime_error("Invalid dimension read from bvecs header");
    }
    dim = d;
    std::cout << "dim: " << dim << std::endl;
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    size_t record_size = sizeof(int) + dim * sizeof(T);
    input.seekg(0, std::ios::beg);
    size_t num = file_size / record_size;
    std::cout << "num: " << num << std::endl;
    T* data = new T[num * d];

    for (size_t i = 0; i < num; ++i) {
        int vec_dim = 0;
        input.read((char*)&vec_dim, sizeof(int));
        assert(vec_dim == d);
        input.read((char*)(data + i * d), sizeof(T) * d);
    }
    n_out = num;
    std::cout << "nb: " << num << std::endl;
    return data;
}

inline
int* read_ivecs(const char* filename, size_t& n_out, size_t& d_out) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open ivecs file");

    int d;
    input.read((char*)&d, sizeof(int));
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    size_t num = file_size / ((d + 1) * sizeof(int));
    int* data = new int[num * d];

    for (size_t i = 0; i < num; ++i) {
        int dim;
        input.read((char*)&dim, sizeof(int));
        assert(dim == d);
        input.read((char*)(data + i * d), sizeof(int) * d);
    }

    d_out = d;
    n_out = num;
    return data;
}

inline
void write_ivecs(const char* filename, const int* data, size_t n, size_t dim) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write to ivecs file");

    for (size_t i = 0; i < n; ++i) {
        int d = (int)dim;
        out.write((char*)&d, 4);
        out.write((char*)(data + i * dim), sizeof(int) * dim);
    }
}

template <typename T = float>
T* read_bin(const char* filename, size_t& n_out, size_t& d_out) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open bin file");
    int32_t n = 0, d = 0;
    input.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
    input.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
    //std::cout << "dim: " << d << std::endl;
    //std::cout << "num: " << n << std::endl;
    if (n <= 0 || d <= 0)
        throw std::runtime_error("Invalid dimensions read from file");
    size_t total = static_cast<size_t>(n) * d;
    T* data = new T[total];
    input.read(reinterpret_cast<char*>(data), sizeof(T) * total);
    if (!input) throw std::runtime_error("Error reading bin vector data");
    n_out = size_t(n);
    d_out = size_t(d);
    return data;
}

inline
int* read_ibin(const char* filename, size_t& n_out, size_t& d_out, size_t start_idx = 0, size_t chunk_size = 0) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open ibin file");

    int32_t n = 0, d = 0;
    input.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
    input.read(reinterpret_cast<char*>(&d), sizeof(int32_t));

    if (start_idx >= size_t(n)) throw std::runtime_error("start_idx out of bounds");

    size_t read_n = (chunk_size == 0 || start_idx + chunk_size > size_t(n)) ?
                    size_t(n) - start_idx : chunk_size;

    int* data = new int[read_n * d];

    input.seekg(sizeof(int32_t) * d * start_idx, std::ios::cur);
    input.read(reinterpret_cast<char*>(data), sizeof(int32_t) * read_n * d);

    if (!input) throw std::runtime_error("Error reading ibin vector data");

    n_out = read_n;
    d_out = d;
    return data;
}

inline
void write_fbin(const char* filename, const float* data, size_t n, size_t d) {
    std::ofstream output(filename, std::ios::binary);
    if (!output) throw std::runtime_error("Cannot open output fbin");

    int32_t nn = static_cast<int32_t>(n);
    int32_t dd = static_cast<int32_t>(d);
    output.write(reinterpret_cast<const char*>(&nn), sizeof(int32_t));
    output.write(reinterpret_cast<const char*>(&dd), sizeof(int32_t));
    output.write(reinterpret_cast<const char*>(data), sizeof(float) * n * d);
}

inline
void write_ibin(const char* filename, const int* data, size_t n, size_t d) {
    std::ofstream output(filename, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Cannot open output ibin file");
    }

    int32_t nn = static_cast<int32_t>(n);
    int32_t dd = static_cast<int32_t>(d);
    output.write(reinterpret_cast<const char*>(&nn), sizeof(int32_t));
    output.write(reinterpret_cast<const char*>(&dd), sizeof(int32_t));
    output.write(reinterpret_cast<const char*>(data), sizeof(int32_t) * n * d);

    if (!output) {
        throw std::runtime_error("Error writing to ibin file");
    }
}

inline
float compute_recall(const idx_t* predicted, const int* groundtruth,
                     size_t nq, int topk, int gt_k) {
    //idx_t max_pred = 0; int max_gt = 0;
    //for (size_t i = 0; i < nq*size_t(topk); i++) max_pred = std::max(max_pred, predicted[i]);
    //for (size_t i = 0; i < nq*size_t(gt_k); i++) max_gt = std::max(max_gt, groundtruth[i]);
    //std::cout << "max_pred=" << max_pred << " max_gt=" << max_gt << std::endl;

    size_t correct = 0;
    for (size_t i = 0; i < nq; ++i) {
        for (int j = 0; j < topk; ++j) {
            int pred = predicted[i * topk + j];
            //if (i<10 && j<3) std::cout << "pred[" << i << "][" << j << "]=" << pred << "\n";
            for (int k = 0; k < gt_k; ++k) {
                int gt = groundtruth[i * gt_k + k];
                //if (j == 0 && i<10 && k<3) std::cout << "gt[" << i << "][" << k << "]=" << gt << "\n";
                if (pred == gt) {
                    ++correct;
                    break;
                }
            }
        }
    }
    return float(correct) / (nq * topk);
}