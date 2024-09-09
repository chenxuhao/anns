#pragma once

#include <vector>
#include <cfloat>
#include <stdint.h>

// vector id type
using vid_t = unsigned;
using vlabel_t = unsigned;
typedef unsigned long long gpu_long_t;
typedef uint32_t vidType;  // vertex ID type
typedef int64_t eidType;   // edge ID type
typedef std::vector<vidType> VertexList; // vertex ID list

#define FULL_MASK 0xffffffff
#define BLOCK_SIZE    128
#define WARP_SIZE     32
#define LOG_WARP_SIZE 5
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
//#define FLT_MAX 3.40282347e+38
#define MAX_DEG 32
#define MASTER_QUEUE_SIZE BLOCK_SIZE * 2 // queue length
