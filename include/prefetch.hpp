#pragma once

#if defined(__SSE2__)
#include <immintrin.h>
#endif

namespace prefetch {

/**
 * @brief CPU 캐시 레벨을 나타내는 열거형
 */
enum class CacheLevel {
    L1,  // 가장 빠른 캐시
    L2,  // 중간 속도의 캐시
    L3   // 가장 느린 캐시
};

/**
 * @brief 메모리 주소를 지정된 캐시 레벨로 프리페치
 * @param address 프리페치할 메모리 주소
 * @param level 캐시 레벨
 */
inline void prefetch(const void* address, CacheLevel level) {
#if defined(__SSE2__)
    switch (level) {
        case CacheLevel::L1:
            _mm_prefetch((const char*)address, _MM_HINT_T0);
            break;
        case CacheLevel::L2:
            _mm_prefetch((const char*)address, _MM_HINT_T1);
            break;
        case CacheLevel::L3:
            _mm_prefetch((const char*)address, _MM_HINT_T2);
            break;
    }
#else
    switch (level) {
        case CacheLevel::L1:
            __builtin_prefetch(address, 0, 3);
            break;
        case CacheLevel::L2:
            __builtin_prefetch(address, 0, 2);
            break;
        case CacheLevel::L3:
            __builtin_prefetch(address, 0, 1);
            break;
    }
#endif
}

/**
 * @brief 연속된 메모리 영역을 L1 캐시로 프리페치
 * @param ptr 시작 주소
 * @param num_lines 프리페치할 캐시 라인 수 (각 라인은 64바이트)
 */
inline void prefetch_range(char* ptr, const int num_lines) {
    for (int i = 0; i < num_lines; i++) {
        prefetch(ptr + i * 64, CacheLevel::L1);
    }
}

/**
 * @brief 여러 주소를 동시에 프리페치
 * @param addresses 프리페치할 주소들의 배열
 * @param count 주소의 개수
 * @param level 캐시 레벨
 */
inline void prefetch_multiple(const void** addresses, size_t count, CacheLevel level) {
    for (size_t i = 0; i < count; i++) {
        prefetch(addresses[i], level);
    }
}

} // namespace prefetch