#pragma once

#include <queue>
#include <cmath>
#include <utility>

inline float limit_range(float x) {
    if (x < 0) {
        x = 0;
    } else if (x > 1) {
        x = 1;
    }
    return x;
}

inline float limit_range_sym(float x) {
    if (x < -1) {
        x = -1;
    } else if (x > 1) {
        x = 1;
    }
    return x;
}

std::pair<float, float> find_minmax(const float* data, int32_t num_items, float ratio = 0.0f) {
    size_t top = int64_t(num_items * ratio) + 1;

    std::priority_queue<float, std::vector<float>, std::greater<float>> max_heap;
    std::priority_queue<float, std::vector<float>, std::greater<float>> min_heap;

    for (int i = 0; i < num_items; ++i) {
        float val = data[i];

        if (max_heap.size() < top) {
            max_heap.push(val);
        } else if (val > max_heap.top()) {
            max_heap.pop();
            max_heap.push(val);
        }

        if (min_heap.size() < top) {
            min_heap.push(-val);
        } else if (val < -min_heap.top()) {
            min_heap.pop();
            min_heap.push(-val);
        }
    }

    return std::make_pair(max_heap.top(), -min_heap.top());
}

float find_absmax(const float* data, int32_t num_items, float ratio = 0.0f) {
    size_t top = int64_t(num_items * ratio) + 1;

    std::priority_queue<float, std::vector<float>, std::greater<float>> heap;

    for (int i = 0; i < num_items; ++i) {
        float x = std::abs(data[i]);
        if (heap.size() < top) {
            heap.push(x);
        } else if (x > heap.top()) {
            heap.pop();
            heap.push(x);
        }
    }

    return heap.top();
}

float find_absmax_without_drop(const float* data, int32_t num_items) {
    float result = 0.0f;
    for (int i = 0; i < num_items; ++i) {
        result = std::max(result, std::abs(data[i]));
    }
    return result;
}