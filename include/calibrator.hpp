#pragma once

#include <iostream>
#include <cmath>
#include "pyanns_utils.hpp" 

class Multiplier {
    protected:
        float mul;

    public:
        Multiplier(float multiplier) : mul(multiplier) {}

        float apply(float x) const {
            return x * mul;
        }
};

class AffineCalibrator : public Multiplier {
    private:
        float min;
        float dif;

    public:
        AffineCalibrator(float multiplier) 
            : Multiplier(multiplier), min(0.0f), dif(0.0f) {}

        void calibrate(const float* data, int32_t num_items, float drop_ratio = 0.0f) {
            auto [max, min_val] = find_minmax(data, num_items, drop_ratio);
            this->min = min_val;
            this->dif = max - this->min;
        }

        int32_t transform(float x) {
            float result = (x - min) / dif;
            result = limit_range(result);
            return std::round(apply(result));
        }

        float revert(float x) {
            return (x / mul) * dif + min;
        }
};

class SymCalibrator : public Multiplier {
    private:
        float max;

    public:
        SymCalibrator(float multiplier) 
            : Multiplier(multiplier), max(0.0f) {}

        void calibrate(const float* data, int32_t num_items, float drop_ratio = 0.0f) {
            if (drop_ratio > 0) {
                this->max = find_absmax(data, num_items, drop_ratio);
            } else {
                this->max = find_absmax_without_drop(data, num_items);
            }
        }

        int32_t transform(float x) {
            float result = x / max;
            result = limit_range_sym(result);
            return std::round(apply(result));
        }

        float revert(float x) {
            return (x / mul) * max;
        }
};