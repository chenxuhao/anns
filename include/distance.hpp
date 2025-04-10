#pragma once
#include <omp.h>

inline int compute_distance_squared(int dim, const unsigned char* __restrict__ a, const unsigned char* __restrict__ b) {
  int ans = 0;
  #pragma omp simd reduction(+ : ans) aligned(a, b : 8)
  for(int i = 0;i < dim; ++ i)
    ans += (int(a[i]) - int(b[i])) * (int(a[i]) - int(b[i]));
  return ans;
}

template <typename T>
inline T compute_distance(int dim, const T* __restrict__ a, const T* __restrict__ b) {
  T ans = 0;
  #pragma omp simd reduction(+ : ans)
  for(int i = 0;i < dim; ++ i)
    ans += (a[i] - b[i]) * (a[i] - b[i]);
  return ans;
  //return sqrt(compute_distance_squared(dim,a,b));
}

inline int compute_ip_distance(int dim, const unsigned char* __restrict__ a, const unsigned char* __restrict__ b) {
  int ans = 0;
  #pragma omp simd
  for(int i = 0;i < dim; ++ i)
    ans += int(a[i]) * int(b[i]);
  return -ans;
}

inline float compute_ip_distance(int dim, const float* __restrict__ a, const float* __restrict__ b) {
  float ans = 0.;
  #pragma omp simd
  for(int i = 0; i < dim; ++i)
    ans += a[i] * b[i];
  return -ans;
}

// from DiskANN
#include <immintrin.h>
static inline float _mm256_reduce_add_ps(__m256 x) {
  /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
  const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
  /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
  const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
  /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
  const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
  /* Conversion to float is a no-op on x86-64 */
  return _mm_cvtss_f32(x32);
}

inline float compute_distance_squared(int dim, const float* __restrict__ a, const float* __restrict__ b) {
  a = (const float *)__builtin_assume_aligned(a, 32);
  b = (const float *)__builtin_assume_aligned(b, 32);

  // assume size is divisible by 8
  uint16_t niters = (uint16_t)(dim / 8);
  __m256 sum = _mm256_setzero_ps();
  for (uint16_t j = 0; j < niters; j++) {
    // scope is a[8j:8j+7], b[8j:8j+7]
    if (j+1 < niters) {
      _mm_prefetch((char *)(a + 8 * (j + 1)), _MM_HINT_T0);
      _mm_prefetch((char *)(b + 8 * (j + 1)), _MM_HINT_T0);
    }
    __m256 a_vec = _mm256_load_ps(a + 8 * j);
    // load b_vec
    __m256 b_vec = _mm256_load_ps(b + 8 * j);
    // a_vec - b_vec
    __m256 tmp_vec = _mm256_sub_ps(a_vec, b_vec);
    sum = _mm256_fmadd_ps(tmp_vec, tmp_vec, sum);
  }
  // horizontal add sum
  return _mm256_reduce_add_ps(sum);
}

inline float compute_ip_distance_simd(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    a = (const float *)__builtin_assume_aligned(a, 32);
    b = (const float *)__builtin_assume_aligned(b, 32);

    // assume size is divisible by 8
    uint16_t niters = (uint16_t)(dim / 8);
    __m256 sum = _mm256_setzero_ps();
    
    for (uint16_t j = 0; j < niters; j++) {
        // prefetch next iteration's data
        if (j+1 < niters) {
            _mm_prefetch((char *)(a + 8 * (j + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(b + 8 * (j + 1)), _MM_HINT_T0);
        }
        // load 8 floats from each vector
        __m256 a_vec = _mm256_load_ps(a + 8 * j);
        __m256 b_vec = _mm256_load_ps(b + 8 * j);
        // multiply and add to accumulator
        sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
    }
    // horizontal add and negate (following the convention of other ip_distance functions)
    return -_mm256_reduce_add_ps(sum);
}


#include <immintrin.h>
#include <cstdint>
#include <cassert>

inline int32_t compute_ip_distance_int8(int32_t dim, const uint8_t* __restrict__ x, const int8_t* __restrict__ y) {
    // Assume pointers are 32-byte aligned.
    x = (const uint8_t *)__builtin_assume_aligned(x, 32);
    y = (const int8_t *)__builtin_assume_aligned(y, 32);

    // Process in blocks of 32 elements.
    int nblocks = dim / 32;         // number of full 32-element blocks

    __m256i sum = _mm256_setzero_si256();

    for (int i = 0; i < nblocks * 32; i += 32) {
        // Process first 16 elements in this block.
        __m128i x_chunk1 = _mm_loadu_si128((__m128i const*)(x + i));         // 16 bytes = 16 uint8_t
        __m128i y_chunk1 = _mm_loadu_si128((__m128i const*)(y + i));         // 16 bytes = 16 int8_t

        // Process next 16 elements.
        __m128i x_chunk2 = _mm_loadu_si128((__m128i const*)(x + i + 16));      
        __m128i y_chunk2 = _mm_loadu_si128((__m128i const*)(y + i + 16));

        // Convert 16 uint8_t to 16 int16_t for each chunk.
        __m256i vx1 = _mm256_cvtepu8_epi16(x_chunk1);
        __m256i vx2 = _mm256_cvtepu8_epi16(x_chunk2);

        // Convert 16 int8_t to 16 int16_t for each chunk.
        __m256i vy1 = _mm256_cvtepi8_epi16(y_chunk1);
        __m256i vy2 = _mm256_cvtepi8_epi16(y_chunk2);

        // Multiply pairwise (producing 16-bit results).
        __m256i prod1 = _mm256_mullo_epi16(vx1, vy1);
        __m256i prod2 = _mm256_mullo_epi16(vx2, vy2);

        // Now widen from 16-bit to 32-bit and accumulate.
        __m128i low1  = _mm256_extracti128_si256(prod1, 0);
        __m128i high1 = _mm256_extracti128_si256(prod1, 1);
        __m256i wide1  = _mm256_cvtepi16_epi32(low1);
        __m256i wide2  = _mm256_cvtepi16_epi32(high1);

        __m128i low2  = _mm256_extracti128_si256(prod2, 0);
        __m128i high2 = _mm256_extracti128_si256(prod2, 1);
        __m256i wide3  = _mm256_cvtepi16_epi32(low2);
        __m256i wide4  = _mm256_cvtepi16_epi32(high2);

        sum = _mm256_add_epi32(sum, wide1);
        sum = _mm256_add_epi32(sum, wide2);
        sum = _mm256_add_epi32(sum, wide3);
        sum = _mm256_add_epi32(sum, wide4);
    }

    // Process any remaining elements (tail loop)
    int32_t tail_sum = 0;
    for (int i = nblocks * 32; i < dim; i++) {
        tail_sum += static_cast<int32_t>(x[i]) * static_cast<int32_t>(y[i]);
    }

    // Horizontal add of the sum vector (8 int32 values in __m256i)
    alignas(32) int32_t tmp[8];
    _mm256_store_si256((__m256i*)tmp, sum);
    int32_t total = tail_sum;
    for (int i = 0; i < 8; ++i) {
        total += tmp[i];
    }

    // Return the negative of the dot product (following your convention)
    return -total;
}