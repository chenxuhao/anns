#pragma once

#include "calibrator.hpp"
#include "tensor.hpp"

#include <omp.h>

enum class Metric {
    IP,
    L2
};

class SQ8UQuantizer : public Tensor, public AffineCalibrator, public SymCalibrator {
    public:
        Metric metric;
        int32_t dim;
        int32_t num_vectors;
        bool for_query;
        int8_t * q_query = nullptr;

        SQ8UQuantizer(Metric metric, int32_t dim, int32_t num_vectors, bool for_query) : 
        Tensor(dim, 8, 64), AffineCalibrator(127), SymCalibrator(127), 
        metric(metric), dim(dim), num_vectors(num_vectors), for_query(for_query) {}

        ~SQ8UQuantizer() {
            if (for_query && q_query) {
                free(q_query);
                q_query = nullptr;
            }
        }
        void train(const float * data) {
            if (metric == Metric::L2) {
                AffineCalibrator::calibrate(data, num_vectors * this->get_dim(), 0);
            } else {
                SymCalibrator::calibrate(data, num_vectors * this->get_dim(), 0);
            }
        }

        void encode(const float * raw_vector, int8_t * quantized_vector) {
            if (metric == Metric::IP) {
                for (int i = 0; i < this->get_dim(); i++){
                    quantized_vector[i] = SymCalibrator::transform(raw_vector[i]);
                    if (for_query == false) {
                        quantized_vector[i] += SymCalibrator::mul;
                    }
                }
            } else {
                for (int i = 0; i < this->get_dim(); i++){
                    quantized_vector[i] = AffineCalibrator::transform(raw_vector[i]);
                }
            }
        }

        void decode(const int8_t *quantized_vector, float *raw_vector) {  
            if (metric == Metric::IP) {
                for (int i = 0; i < this->get_dim(); i++) {
                    raw_vector[i] = SymCalibrator::revert(quantized_vector[i]);
                }    
            } else {
                for (int i = 0; i < this->get_dim(); i++) {
                    raw_vector[i] = AffineCalibrator::revert(quantized_vector[i]);
                }
            }      
        }

        void add(const float * raw_data) {
            if (for_query == true) {
                int32_t csize = this->code_size();
                
                /*More aggressive alignment*/
                q_query = (int8_t *)aligned_alloc(this->dim_align(), csize);

                /*
                Less aggressive alignment (case by case)
                if (csize % 64 == 0){
                    q_query = (int8_t *)aligned_alloc(64, csize)
                } else if (csize % 32 == 0) {
                    q_query = (int8_t *)aligned_alloc(32, csize)
                } else {
                    q_query = (int8_t *)malloc(csize * num_vectors);
                }

                encode(raw_data, (int8_t *)q_query);
                */

                encode(raw_data, (int8_t *)q_query);
            } else {
                this->init(num_vectors);

                #pragma omp parallel for schedule(dynamic)
                for(int64_t i = 0; i < num_vectors; i++){
                    encode(raw_data + i * this->get_dim(), (int8_t *)this->get_code_at(i));
                }   
            }
        }

        uint8_t * get_quantized_at(int32_t index) {
            return (uint8_t *)this->get_code_at(index);
        }

        int8_t * get_quantized_query() {
            return q_query;
        }
};
