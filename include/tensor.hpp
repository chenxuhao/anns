#pragma once

#include <cstdint>
#include <cstdlib>   
#include <cstring>   

#include "prefetch.hpp"

class Tensor {
    public:
        int32_t nbits = 0; // number of bits/value
        int32_t dim = 0; // dimension of each vector
        int32_t num_vectors = 0; // number of vectors
        int32_t csize = 0; // number of bytes/vector
        int32_t align_width = 0; // padding for alignment
        int32_t dalign = 0; // actual alignment 
        char * codes = nullptr; // array of quantized data

        Tensor(int32_t dim, int32_t nbits, int32_t align_width) : 
            dim(dim), nbits(nbits), align_width(align_width) {
                dalign = ((dim + align_width - 1) / align_width * align_width); // make dalign a nearest multiple of align_width
                csize = nbits / 8 * dalign;
            }
        
    ~Tensor() { 
        free(codes);
    }

    void init(int32_t n) {
        num_vectors = n;

        /*More aggressive alignment*/

        codes = (char *)aligned_alloc(dalign, (int64_t) csize * num_vectors);
         
        /*
        Less aggressive alignment (case by case)
        if (csize % 64 == 0){
            codes = (char *)aligned_alloc(64, (int64_t) csize * num_vectors);
        } else if (csize % 32 == 0) {
            codes = (char *)aligned_alloc(32, (int64_t) csize * num_vectors);
        } else {
            codes = (char *)malloc((int64_t)csize * num_vectors);
        }
        */    

    }

    char * get_code_at(int32_t u) {
        return codes + (int64_t)(u * csize);
    }

    char * get_full_codes() {
        return codes;
    }

    void prefetch(int32_t u, int32_t num_lines) {
        prefetch::prefetch_range(get_code_at(u), num_lines);
    }

    int32_t size() const { 
        return num_vectors; 
    }
    int32_t get_dim() const { 
        return dim; 
    }
    int32_t dim_align() const { 
        return dalign; 
    }
    int32_t code_size() const { 
        return csize; 
    }

};