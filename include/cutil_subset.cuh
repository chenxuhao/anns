#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus) {                                                    \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
            exit( cudaStatus );                                                             \
        }                                                                                   \
    }

#  define CUDA_SAFE_CALL_NO_SYNC( call) {                                      \
    cudaError err = call;                                                      \
    if( cudaSuccess != err) {                                                  \
        fprintf(stderr, "error %d: Cuda error in file '%s' in line %i : %s.\n",\
                err, __FILE__, __LINE__, cudaGetErrorString( err) );           \
        exit(EXIT_FAILURE);                                                    \
    } }

#  define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);              \

#  define CUDA_SAFE_THREAD_SYNC( ) {                                           \
    cudaError err = CUT_DEVICE_SYNCHRONIZE();                                  \
    if ( cudaSuccess != err) {                                                 \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",          \
                __FILE__, __LINE__, cudaGetErrorString( err) );                \
    } }

#if __CUDACC_VER_MAJOR__ >= 9
#define SHFL_DOWN(a,b) __shfl_down_sync(0xFFFFFFFF,a,b)
#define SHFL(a,b) __shfl_sync(0xFFFFFFFF,a,b)
#else
#define SHFL_DOWN(a,b) __shfl_down(a,b)
#define SHFL(a,b) __shfl(a,b)
#endif

namespace cutils {

// you must first call the cudaGetDeviceProperties() function, then pass
// the devProp structure returned to this function:
inline int getSPcores(cudaDeviceProp devProp) {
  int cores = 0;
  int mp = devProp.multiProcessorCount;
  switch (devProp.major){
    case 2: // Fermi
      if (devProp.minor == 1) cores = mp * 48;
      else cores = mp * 32;
      break;
    case 3: // Kepler
      cores = mp * 192;
      break;
    case 5: // Maxwell
      cores = mp * 128;
      break;
    case 6: // Pascal
      if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
      else if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
    case 7: // Volta and Turing
      if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
    case 8: // Ampere
      if (devProp.minor == 0) cores = mp * 64;
      else if (devProp.minor == 6) cores = mp * 128;
      else if (devProp.minor == 9) cores = mp * 128; // ada lovelace
      else printf("Unknown device type\n");
      break;
    case 9: // Hopper
      if (devProp.minor == 0) cores = mp * 128;
      else printf("Unknown device type\n");
      break;
    default:
      printf("Unknown device type\n");
      break;
  }
  return cores;
}

static size_t print_device_info(int print_num, bool disable = false) {
  int deviceCount = 0;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));
  if (!disable) printf("Found %d devices\n", deviceCount);
  // Another way to get the # of cores: #include <helper_cuda.h> in this link:
  // https://github.com/NVIDIA/cuda-samples/blob/6be514679b201c8a0f0cda050bc7c01c8cda32ec/Common/helper_cuda.h
  //int CUDACores = _ConvertSMVer2Cores(props.major, props.minor) * props.multiProcessorCount;
  size_t mem_size = 0;
  for (int device = 0; device < deviceCount; device++) {
    cudaDeviceProp prop;
    CUDA_SAFE_CALL(cudaSetDevice(device));
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, device));
    if (device == 0) mem_size = prop.totalGlobalMem;
    if (disable) break;
    printf("  Device[%d]: %s\n", device, prop.name);
    if (device == 0 || print_num > 0) {
      printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
      printf("  Warp size: %d\n", prop.warpSize);
      printf("  Total # SM: %d\n", prop.multiProcessorCount);
      printf("  Total # CUDA cores: %d\n", getSPcores(prop));
      printf("  Total amount of shared memory per block: %lu bytes\n", prop.sharedMemPerBlock);
      printf("  Total # registers per block: %d\n", prop.regsPerBlock);
      printf("  Total amount of constant memory: %lu bytes\n", prop.totalConstMem);
      printf("  Total global memory: %.1f GB\n", float(prop.totalGlobalMem)/float(1024*1024*1024));
      printf("  Memory Clock Rate: %.2f GHz\n", float(prop.memoryClockRate)/float(1024*1024));
      printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
      //printf("  Maximum memory pitch: %u\n", prop.memPitch);
      printf("  Peak Memory Bandwidth: %.2f GB/s\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
  }
  return mem_size;
}

static size_t get_gpu_mem_size(int device = 0) {
  cudaDeviceProp prop;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, device));
  return prop.totalGlobalMem;
}

template <typename NumeratorT, typename DenominatorT>
__host__ __device__ __forceinline__ constexpr NumeratorT
DivideAndRoundUp(NumeratorT n, DenominatorT d) {
  // Static cast to undo integral promotion.
  return static_cast<NumeratorT>(n / d + (n % d != 0 ? 1 : 0));
}

template <typename T>
void allocate_gpu_buffer(size_t n, T*& ptr) {
  //std::cout << "allocating GPU memory: size = " << n << "\n";
  size_t total_buffer_size = n * sizeof(T);
  std::cout << "Allocated memory for buffers: " << float(total_buffer_size)/float(1024*1024) << " MB\n";
  CUDA_SAFE_CALL(cudaMalloc((void**)&ptr, total_buffer_size));
}
 
template <typename T = float>
__device__ __forceinline__ T compute_distance(int dim, const T* a, const T* b) {
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  T val = 0.;
  for(int i = thread_lane; i < dim; i += WARP_SIZE)
    val += (a[i] - b[i]) * (a[i] - b[i]);
  T sum = val;
  sum += SHFL_DOWN(sum, 16);
  sum += SHFL_DOWN(sum, 8);
  sum += SHFL_DOWN(sum, 4);
  sum += SHFL_DOWN(sum, 2);
  sum += SHFL_DOWN(sum, 1);
  sum  = SHFL(sum, 0);
  return sum;
}

template <typename T>
__device__ __forceinline__ void memmove(T* new_ptr, T* old_ptr, T num) {
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  for (auto i = thread_lane; i < num; i += WARP_SIZE)
    new_ptr[i] = old_ptr[i];
}

template <typename T>
__device__ __forceinline__ T* upper_bound(T* begin, T* end, T bound) {
  int thread_lane = threadIdx.x & (WARP_SIZE-1);            // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;                // warp index within the CTA
  __shared__ int loc[WARPS_PER_BLOCK];
  if (thread_lane == 0) loc[warp_lane] = 0;
  __syncwarp();
  int l = 0;
  int r = end - begin;
  for (auto i = thread_lane + l; i < r; i += WARP_SIZE) {
    int found = 0;
    if (begin[i] < bound) found = 1;
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, found);
    if (thread_lane == 0) loc[warp_lane] += __popc(mask);
    __syncwarp(active);
    if (mask != FULL_MASK) break;
  }
  return begin + loc[warp_lane];
}

template <typename KeyT = float, typename ValueT = vidType>
__device__ __forceinline__ int upper_bound(int length, KeyT* keys, ValueT* values, KeyT bound) {
  int thread_lane = threadIdx.x & (WARP_SIZE-1);   // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;       // warp index within the CTA
  __shared__ int location[WARPS_PER_BLOCK];
  if (thread_lane == 0) location[warp_lane] = 0;
  __syncwarp();
  int l = 0;
  int r = length;
  for (auto i = thread_lane + l; i < r; i += WARP_SIZE) {
    int found = 0;
    if (keys[i] < bound) found = 1;
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, found);
    if (thread_lane == 0) location[warp_lane] += __popc(mask);
    __syncwarp(active);
    if (mask != FULL_MASK) break;
  }
  return location[warp_lane];
}

template <typename KeyT = float, typename ValueT = vidType>
__device__ __forceinline__ int upper_bound_cta(int length, KeyT* keys, ValueT* values, KeyT bound) {
  return 0;
}

template <typename KeyT = float>
__device__ __forceinline__ int upper_bound_cta(int length, KeyT* keys, KeyT bound) {
  assert(length >= 0);
  if (length == 0) return 0;
  //int thread_lane = threadIdx.x & (WARP_SIZE-1);   // thread index within the warp
  //int warp_lane   = threadIdx.x / WARP_SIZE;       // warp index within the CTA
  typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;
  __shared__ BlockScan::TempStorage temp_storage;
  __shared__ int location;
  if (threadIdx.x == 0) location = 0;
  __syncthreads();

  int round = (length - 1) / BLOCK_SIZE + 1;
  for (vidType i = threadIdx.x; i < round * BLOCK_SIZE; i += BLOCK_SIZE) {
    int found = 0;
    if (i < length && keys[i] < bound) found = 1;
    int position = 0, total_num = 0;
    BlockScan(temp_storage).ExclusiveSum(found, position, total_num);
    if (threadIdx.x == 0) location += total_num;
    __syncthreads();
    if (total_num != BLOCK_SIZE) break;
  }
  //if (location > length && (threadIdx.x < 128 || threadIdx.x == 127)) printf("tid=%d, overflow: location=%d, length=%d\n", threadIdx.x, location, length);
  //assert(location <= length);
  int pos = location == length? location - 1 : location;
  //if (location == length) return location - 1;
  return pos;
}

template <typename T>
__forceinline__ __device__ bool binary_search_2phase(T *list, T *cache, T key, int size) {
  int p = (threadIdx.x / WARP_SIZE) * WARP_SIZE;
  int mid = 0;
  // phase 1: search in the cache
  int bottom = 0;
  int top = WARP_SIZE;
  while (top > bottom + 1) {
    mid = (top + bottom) / 2;
    auto y = cache[p + mid];
    if (key == y) return true;
    if (key < y) top = mid;
    if (key > y) bottom = mid;
  }

  //phase 2: search in global memory
  bottom = bottom * size / WARP_SIZE;
  top = top * size / WARP_SIZE - 1;
  while (top >= bottom) {
    mid = (top + bottom) / 2;
    auto y = list[mid];
    if (key == y) return true;
    if (key < y) top = mid - 1;
    else bottom = mid + 1;
  }
  return false;
}

template <typename T = vidType>
__forceinline__ __device__ bool binary_search_2phase_cta(T *list, T *cache, T key, int size) {
  vidType y = 0;
  int mid = 0;
  // phase 1: cache
  int bottom = 0;
  int top = BLOCK_SIZE;
  while (top > bottom + 1) {
    mid = (top + bottom) / 2;
    y = cache[mid];
    if (key == y) return true;
    if (key < y) top = mid;
    if (key > y) bottom = mid;
  }
  //phase 2
  bottom = bottom * size / BLOCK_SIZE;
  top = top * size / BLOCK_SIZE - 1;
  while (top >= bottom) {
    mid = (top + bottom) / 2;
    y = list[mid];
    if (key == y) return true;
    if (key < y) top = mid - 1;
    else bottom = mid + 1;
  }
  return false;
}

template <typename KeyT = vidType, typename ValueT = float>
__forceinline__ __device__ int set_difference_warp(int size_a, KeyT* key_a, ValueT* val_a, 
                                                   int size_b, KeyT* key_b, ValueT* val_b,
                                                   KeyT* key_c, ValueT* val_c) {
  int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;     // warp index within the CTA
  __shared__ int count[WARPS_PER_BLOCK];
  __shared__ KeyT cache[BLOCK_SIZE];
  cache[warp_lane * WARP_SIZE + thread_lane] = key_b[thread_lane * size_b / WARP_SIZE];
  __syncwarp();

  if (thread_lane == 0) count[warp_lane] = 0;
  for (auto i = thread_lane; i < size_a; i += WARP_SIZE) {
    unsigned active = __activemask();
    __syncwarp(active);
    auto key = key_a[i]; // each thread picks a vertex as the key
    auto val = val_a[i];
    int found = 0;
    if (!binary_search_2phase(key_b, cache, key, size_b))
      found = 1;
    unsigned mask = __ballot_sync(active, found);
    auto idx = __popc(mask << (WARP_SIZE-thread_lane-1));
    if (found) {
      auto position = count[warp_lane]+idx-1;
      key_c[position] = key;
      val_c[position] = val;
    }
    if (thread_lane == 0) count[warp_lane] += __popc(mask);
  }
  return count[warp_lane];
}

template <typename KeyT = vidType, typename ValueT = float>
__forceinline__ __device__ int set_difference_cta(int size_a, KeyT* key_a, ValueT* val_a, 
                                                  int size_b, KeyT* key_b, ValueT* val_b,
                                                  KeyT* key_c, ValueT* val_c) {
  typedef cub::BlockScan<int, BLOCK_SIZE> BlockScan;
  __shared__ BlockScan::TempStorage temp_storage;

  __shared__ int count;
  __shared__ KeyT cache[BLOCK_SIZE];
  cache[threadIdx.x] = key_b[threadIdx.x * size_b / BLOCK_SIZE];
  if (threadIdx.x == 0) count = 0;
  __syncthreads();
  for (vidType i = threadIdx.x; i < size_a; i += BLOCK_SIZE) {
    auto key = key_a[i];
    auto val = val_a[i];
    int found = 0;
    if (binary_search_2phase_cta(key_b, cache, key, size_b))
      found = 1;
    // TODO: PrefixSum
    int position = 0, total_num = 0;
    BlockScan(temp_storage).ExclusiveSum(found, position, total_num);
    if (found) {
      if (threadIdx.x == 0) count += 1;
      key_c[position] = key;
      val_c[position] = val;
    }
  }
  return count;
}

template <typename KeyT = vidType, typename ValueT = float>
__forceinline__ __device__ int set_union(int size_a, KeyT* key_a, ValueT* val_a, 
                                         int size_b, KeyT* key_b, ValueT* val_b,
                                         KeyT* key_c, ValueT* val_c) {
  //int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  int warp_lane   = threadIdx.x / WARP_SIZE;       // warp index within the CTA
  __shared__ int count[WARPS_PER_BLOCK];
  return count[warp_lane];
} 

template <typename KeyT = vidType, typename ValueT = float>
__forceinline__ __device__ int set_union_cta(int size_a, KeyT* key_a, ValueT* val_a, 
                                             int size_b, KeyT* key_b, ValueT* val_b,
                                             KeyT* key_c, ValueT* val_c) {
  //int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
  //int warp_lane   = threadIdx.x / WARP_SIZE;       // warp index within the CTA
  __shared__ int count;
  return count;
} 

} // end namespace
