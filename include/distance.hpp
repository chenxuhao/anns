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
  for(int i = 0;i < dim; ++ i)
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

