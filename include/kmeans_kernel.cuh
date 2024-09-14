#include <cub/cub.cuh>
#include <cub/block/block_reduce.cuh>
#include "common.hpp"

#define THREADS_PER_DIM 16
#define BLOCKS_PER_DIM 16
#define THREADS_PER_BLOCK THREADS_PER_DIM*THREADS_PER_DIM

__global__ void invert_mapping(float *input, float *output, int n, int dim) {
  int point_id = threadIdx.x + blockDim.x*blockIdx.x;
  if (point_id < n) {
    for (int i=0; i<dim; i++)
      output[point_id + n*i] = input[point_id*dim+i];
  }
}

typedef cub::BlockReduce<gpu_long_t, BLOCK_SIZE> BlockReduce;

__global__ void find_closest_center(int dim, int npoints, int nclusters, float *features, int *membership, const float *centroids, gpu_long_t *delta) {
  int block_id = gridDim.x*blockIdx.y+blockIdx.x;
  int point_id = block_id*blockDim.x*blockDim.y + threadIdx.x;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int index = -1;
  gpu_long_t count = 0;
  if (point_id < npoints) {
    float min_dist = FLT_MAX;
    for (int i = 0; i < nclusters; i++) {
      int cluster_base_index = i*dim;
      float dist = 0.0;
      for (int j = 0; j < dim; j++) {
        float diff = features[point_id + j*npoints] - centroids[cluster_base_index + j];
        dist += diff*diff;/* sum of squares */
      }
      if (dist < min_dist) {
        min_dist = dist;
        index    = i;
      }
    }
    count = (membership[point_id] != index) ? 1 : 0;
    membership[point_id] = index;
  }
  auto block_delta = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) atomicAdd(delta, block_delta);
}

__global__ void update_new_centers(int dim, int npoints, int nclusters, float *features_flipped, int *membership, float *block_clusters) {
  int block_id = gridDim.x*blockIdx.y+blockIdx.x;
  int point_id = block_id*blockDim.x*blockDim.y + threadIdx.x;
  int center_id = threadIdx.x / dim;    
  int dim_id = threadIdx.x - dim*center_id;
  __shared__ int new_center_ids[THREADS_PER_BLOCK];

  if (point_id < npoints) {
    new_center_ids[threadIdx.x] = membership[point_id];
    __syncthreads();
    // determine which dimension calculte the sum for mapping of threads is center0[dim0,dim1,dim2,...]center1[dim0,dim1,dim2,...]...
    int new_base_index = (point_id - threadIdx.x)*dim + dim_id;
    float accumulator = 0.f;
    if (threadIdx.x < dim * nclusters) {
      // accumulate over all the elements of this threadblock 
      for (int i = 0; i< (THREADS_PER_BLOCK); i++) {
        float val = features_flipped[new_base_index+i*dim];
        if (new_center_ids[i] == center_id) 
          accumulator += val;
      }
      // now store the sum for this threadblock
      block_clusters[(blockIdx.y*gridDim.x + blockIdx.x) * nclusters * dim + threadIdx.x] = accumulator;
    }
  }
}

