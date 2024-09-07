#pragma once

#include <cmath>
#include <chrono>
#include <vector>
#include <cassert>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <random>

#include <omp.h>
#include <cxxabi.h>
#include <unistd.h>
#include <immintrin.h>
#include <sys/resource.h>

#include "timer.hpp"

namespace utils {

template <typename T>
T* kmean_cluster(size_t npoints, int dim, int nclusters, size_t threshold,
                 const T* features, std::vector<int> &membership, int iterations = 500);
 
template<typename T>
std::string type_name() {
  int status;
  std::string tname = typeid(T).name();
  char *demangled_name = abi::__cxa_demangle(tname.c_str(), NULL, NULL, &status);
  if(status == 0) {
    tname = demangled_name;
    std::free(demangled_name);
  }   
  return tname;
}

template <typename T>
void print_vector(size_t vid, int dim, const T *data, std::string name = "vec") {
  const int n = 8;
  //printf("DEBUG: vector dim: %d\n", dim);
  //std::cout << "DEBUG: vector type: " << type_name<T>() << "\n";
  assert(dim > 2*n);
  std::cout << name << "[" << vid << "] = [ ";
  for (int j = 0; j < n; j ++) {
    std::cout << data[vid*dim+j] << " ";
  }
  std::cout << "... ";
  for (int j = dim-n; j < dim; j ++) {
    std::cout << data[vid*dim+j] << " ";
  }
  std::cout << " ]\n";
}

template <typename T>
void load_vectors(size_t &num, int &dim, const char *filename, T *data) {
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    fprintf(stderr, "Error: cannot open file %s\n", filename);
    exit(EXIT_FAILURE);
  }
  in.read((char*)&dim, 4);
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t)ss;
  num = size_t(fsize / (dim + 1) / 4);
  //printf("DEBUG: num = %ld, dim = %d\n", num, dim);
  //data = new T[uint64_t(num) * uint64_t(dim)];
  in.seekg(0, std::ios::beg);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    in.read((char*)(data + i * dim), dim * 4);
  }
  in.close();
}

template <typename T = float>
T compute_distance(int dim, const T* __restrict__ a, const T* __restrict__ b) {
  T ans = 0.;
  #pragma omp simd
  for(int i = 0;i < dim; ++ i)
    ans += (a[i] - b[i]) * (a[i] - b[i]);
  return std::sqrt(ans);
}

template <typename T = float>
T compute_distance_squared(int dim, const T* __restrict__ a, const T* __restrict__ b) {
  T ans = 0.;
  #pragma omp simd
  for(int i = 0;i < dim; ++ i)
    ans += (a[i] - b[i]) * (a[i] - b[i]);
  return ans;
}

template <typename T = float>
T compute_distance_squared_early_stop(int dim, const T* __restrict__ a, const T* __restrict__ b, T const cutoff) {
  T ans = 0.;
  for(int i = 0;i < dim; ++ i)
    if((ans += (a[i] - b[i]) * (a[i] - b[i])) > cutoff)
      return cutoff;
  return ans;
}

// SIMD version
template <typename T = float>
T compute_distance_avx2(int dim, const T* __restrict__ a, const T* __restrict__ b) {
  T result = 0;
#define AVX_L2SQR(addr1, addr2, dest, tmp1, tmp2) \
      tmp1 = _mm256_loadu_ps(addr1);\
      tmp2 = _mm256_loadu_ps(addr2);\
      tmp1 = _mm256_sub_ps(tmp1, tmp2); \
      tmp1 = _mm256_mul_ps(tmp1, tmp1); \
      dest = _mm256_add_ps(dest, tmp1);

  __m256 sum;
  __m256 l0, l1;
  __m256 r0, r1;
  unsigned D = (dim + 7) & ~7U;
  unsigned DR = D % 16;
  unsigned DD = D - DR;
  const T* l = a;
  const T* r = b;
  const T* e_l = l + DD;
  const T* e_r = r + DD;
  T unpack[8] __attribute__ ((aligned (32))) = {0, 0, 0, 0, 0, 0, 0, 0};

  sum = _mm256_load_ps(unpack);
  if(DR) { AVX_L2SQR(e_l, e_r, sum, l0, r0); }

  for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
    AVX_L2SQR(l, r, sum, l0, r0);
    AVX_L2SQR(l + 8, r + 8, sum, l1, r1);
  }
  _mm256_store_ps(unpack, sum);
  result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];

  return result;
}

class StopW {
	std::chrono::steady_clock::time_point time_begin;
public:
	StopW() {
		time_begin = std::chrono::steady_clock::now();
	}
	double getElapsedTimeMicro() {
		std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
		return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
	}
	void reset() {
		time_begin = std::chrono::steady_clock::now();
	}
};

/**
* Returns the peak (maximum so far) resident set size (physical
* memory use) measured in bytes, or zero if the value cannot be
* determined on this OS.
*/
static inline size_t getPeakRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
        return (size_t)0L;      /* Can't open? */
    if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo))
    {
        close(fd);
        return (size_t)0L;      /* Can't read? */
    }
    close(fd);
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t) (rusage.ru_maxrss * 1024L);
#endif

#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L;          /* Unsupported. */
#endif
}

/**
* Returns the current resident set size (physical memory use) measured
* in bytes, or zero if the value cannot be determined on this OS.
*/
static inline size_t getCurrentRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
        (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L;      /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t) 0L;      /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t) 0L;      /* Can't read? */
    }
    fclose(fp);
    return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L;          /* Unsupported. */
#endif
}

} // end namespace
