#pragma once

#include <vector>
#include <cfloat>

// vector id type
using vid_t = unsigned;
typedef unsigned long long gpu_long_t;

#define FULL_MASK 0xffffffff
#define BLOCK_SIZE    256
#define WARP_SIZE     32
#define LOG_WARP_SIZE 5
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define NUM_BUCKETS 128
#define BUCKET_SIZE 1024
#define ADJ_SIZE_THREASHOLD 1024

//#define FLT_MAX 3.40282347e+38
