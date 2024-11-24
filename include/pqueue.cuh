#include "cutil_subset.cuh"
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>

#define INIT_QUEUE_LENGTH 512

template <typename T>
class PQ_GPU {
private:
  //int n_workers;
  int* next_idx;
  int* queue_size;
  int queue_capacity;

  T* vid_queues;
  float* dis_queues;
  uint8_t* exp_queues;

public:
  PQ_GPU(int nw, int L) {
    //n_workers = nw;
    queue_capacity = L;
    CUDA_SAFE_CALL(cudaMalloc((void **)&next_idx,   nw * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&queue_size, nw * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&vid_queues,  nw * L * sizeof(T)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&dis_queues,  nw * L * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&exp_queues,  nw * L * sizeof(uint8_t)));
    // zero-initialize 
    CUDA_SAFE_CALL(cudaMemset((void *)exp_queues, 0, nw * L * sizeof(uint8_t)));
    CUDA_SAFE_CALL(cudaMemset((void *)next_idx, 0, nw * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset((void *)queue_size, 0, nw * sizeof(int)));
  }
	
  inline __device__ __host__ int get_capacity() const { return queue_capacity; }
  inline __device__ int get_next_index(int wid) const { return next_idx[wid]; }
  inline __device__ float get_tail_dist(int wid) const { return size(wid) == 0 ? FLT_MAX : dis_queues[wid*queue_capacity+size(wid)-1]; }
  inline __device__ T get_vid(int wid, size_t idx) const { return vid_queues[wid*queue_capacity+idx]; }
  inline __device__ float get_dist(int wid, int idx) const { return dis_queues[wid*queue_capacity+idx]; }
  inline __device__ bool is_expanded(int wid, int idx) const { return exp_queues[wid*queue_capacity+idx] == 1 ? true : false; }
  inline __device__ int size(int wid) const { return queue_size[wid]; }
  inline __device__ bool is_full(int wid) const { return queue_size[wid] == queue_capacity; }

  inline __device__ void set_vid(int wid, int idx, T vid) { vid_queues[wid*queue_capacity+idx] = vid; }
  inline __device__ void set_dist(int wid, int idx, float dist) { dis_queues[wid*queue_capacity+idx] = dist; }
  inline __device__ void set_expanded(int wid, int idx) { exp_queues[wid*queue_capacity+idx] = 1; }
  inline __device__ void set_unexpanded(int wid, int idx) { exp_queues[wid*queue_capacity+idx] = 0; }
  inline __device__ void clear(int wid) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
    if (thread_lane == 0) {
      next_idx[wid] = 0;
      queue_size[wid] = 0;
    }
    __syncwarp();
  }

  inline __device__ void insert(int wid, T vid, float dist, int loc) {
    //int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
    //if (thread_id == 0) printf("thread 0 wid %d, insert vid %d, distance %f, location %d\n", wid, vid, dist, loc);
    auto offset = wid*queue_capacity;
    auto vid_queue = vid_queues + offset;
    auto distances = dis_queues + offset;
    auto expanded  = exp_queues + offset;
    //atomicAdd(queue_size, 1);
    queue_size[wid] ++;
    vid_queue[loc] = vid;
    distances[loc] = dist;
    expanded[loc] = 0;
  }
 
  // push a single element into the queue using a warp
  inline __device__ int push(int wid, T vid, float dist) {
    int thread_lane = threadIdx.x & (WARP_SIZE-1); // thread index within the warp
    auto offset = wid*queue_capacity;
    auto vid_queue = vid_queues + offset;
    auto distances = dis_queues + offset;
    auto expanded  = exp_queues + offset;
 
    if (size(wid) == 0) {
      if (thread_lane == 0) insert(wid, vid, dist, 0);
      __syncwarp();
      return 0;
    }
    // Find the insert location
    auto queue_start = &distances[0];
    const auto loc = cutils::lower_bound(queue_start, queue_start + size(wid), dist);
    auto insert_loc = loc - queue_start;
    if (insert_loc != size(wid)) {
      if (vid == vid_queue[insert_loc]) { // Duplicate, skip
        return -2;
      }
      if (size(wid) >= queue_capacity) { // Queue is full
        //atomicSub(queue_size, 1);
        if (thread_lane == 0) queue_size[wid] --;
        __syncwarp();
      }
    } else { // insert_loc == queue_size, insert at the end?
      if (size(wid) >= queue_capacity) return -1;
      if (thread_lane == 0) insert(wid, vid, dist, size(wid));
      __syncwarp();
      return insert_loc;
    }
    // Add into queue
    auto num = size(wid) - insert_loc;
    cutils::memmove<T>(&vid_queue[insert_loc + 1], &vid_queue[insert_loc], num);
    cutils::memmove<float>(&distances[insert_loc + 1], &distances[insert_loc], num);
    cutils::memmove<uint8_t>(&expanded[insert_loc + 1], &expanded[insert_loc], num);
    if (thread_lane == 0) insert(wid, vid, dist, insert_loc);
    __syncwarp();
    if (insert_loc < get_next_index(wid)) {
      if (thread_lane == 0) *next_idx = insert_loc;
      __syncwarp();
    }
    return insert_loc;
  }

  // sort the queue in acsending distance order, using a thread block
  inline __device__ void reorder(int wid) {
    auto offset = wid*queue_capacity;
    auto vid_queue = vid_queues + offset;
    auto distances = dis_queues + offset;
    auto expanded  = exp_queues + offset;

    enum { ITEMS_PER_THREAD = INIT_QUEUE_LENGTH / BLOCK_SIZE };
    
    // Specialize BlockLoad type for our thread block (uses warp-striped loads for coalescing, then transposes in shared memory to a blocked arrangement)
    typedef cub::BlockLoad<float, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadKey;
    typedef cub::BlockLoad<T, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadVal;

    // Specialize BlockRadixSort for a 1D block of 256 threads owning 2 integer items each
    typedef cub::BlockRadixSort<float, BLOCK_SIZE, ITEMS_PER_THREAD, T> BlockRadixSort;

    // Allocate shared memory
    //__shared__ typename BlockRadixSort::TempStorage temp_storage;
    __shared__ union TempStorage
    {
        typename BlockLoadKey::TempStorage      loadK;
        typename BlockLoadVal::TempStorage      loadV;
        typename BlockRadixSort::TempStorage    sort;
    } temp_storage;

    // Obtain a segment of consecutive items that are blocked across threads
    float thread_keys[ITEMS_PER_THREAD];
    T   thread_values[ITEMS_PER_THREAD];

    // Load items into a blocked arrangement
    BlockLoadKey(temp_storage.loadK).Load(distances, thread_keys);
    BlockLoadVal(temp_storage.loadV).Load(vid_queue, thread_values);

    // Collectively sort the keys
    BlockRadixSort(temp_storage.sort).Sort(thread_keys, thread_values);

    // Store output in striped fashion
    cub::StoreDirectStriped<BLOCK_SIZE>(threadIdx.x, distances, thread_keys);
    cub::StoreDirectStriped<BLOCK_SIZE>(threadIdx.x, vid_queue, thread_values);
  }
};

