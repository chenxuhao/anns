#include <iostream>
#include "calibrator.hpp"
#include "pyanns_utils.hpp"
#include "tensor.hpp"
#include "sq8u_quant.hpp"
#include "distance.hpp"

int main() {
    const float data[] = {-3.4, 5.2, -5.6, -1.5, 8.2, 3.8, -7.7, 2.1, 0.3};
    const float query[] = {4.3, 3.2, 6.4};

    int32_t dim = 3;
    int32_t num_vectors = 3;

    SQ8UQuantizer quantizer_data(Metric::IP, dim, num_vectors, false);
    SQ8UQuantizer quantizer_query(Metric::IP, dim, 1, true);

    quantizer_data.train(data);
    quantizer_query.train(query);

    quantizer_data.add(data);
    quantizer_query.add(query);

    std::cout << std::endl;
    std::cout << "Quantized data: \n";
    for (int i = 0; i < num_vectors; i++){
        uint8_t * data_vector = (uint8_t *)quantizer_data.get_quantized_at(i);
        for (int j = 0; j < dim; j++){
            std::cout << static_cast<int32_t>(data_vector[j]) << ", ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Quantized query: \n";
    int8_t * data_query = quantizer_query.get_quantized_query();
    for (int j = 0; j < dim; j++){
        std::cout << static_cast<int32_t>(data_query[j]) << ", ";
    }

    std::cout << std::endl;

    /*
    for (int i = 0; i < num_vectors; i++){
        int32_t dist = compute_ip_distance_int8(dim, (uint8_t *)quantizer_data.get_quantized_at(i), data_query);
        std::cout << dist << std::endl;
    }*/
    
}