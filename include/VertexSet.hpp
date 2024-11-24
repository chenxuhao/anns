#pragma once

#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

#include <limits>
#include <vector>
#include <fstream>
#include <sstream>
#include <cassert>
#include <iostream>
#include <iterator>
#include <algorithm>

#include "common.hpp"

constexpr vidType VID_MIN = 0;
constexpr vidType VID_MAX = std::numeric_limits<vidType>::max();

class VertexSet {
private: // memory managed regions for per-thread intermediates
  vidType *ptr;
  vidType set_size, vid;
  const bool pooled;
  static thread_local std::vector<vidType*> buffers_exist, buffers_avail;

public:
  static void release_buffers();
  static vidType MAX_DEGREE;

  VertexSet() : VertexSet(vidType(-1)) {}
  VertexSet(vidType v) : set_size(0), vid(v), pooled(true) {
    if(buffers_avail.size() == 0) { 
      vidType *p = new vidType[MAX_DEGREE];
      buffers_exist.push_back(p);
      buffers_avail.push_back(p);
    }
    ptr = buffers_avail.back();
    buffers_avail.pop_back();
  }
  VertexSet(vidType *p, vidType s, vidType id) : 
    ptr(p), set_size(s), vid(id), pooled(false) {}
  VertexSet(const VertexSet&)=delete;
  VertexSet& operator=(const VertexSet&)=delete;
  VertexSet(VertexSet&&)=default;
  VertexSet& operator=(VertexSet&&)=default;

  void duplicate(const VertexSet &other) {
    ptr = other.ptr;
    set_size = other.set_size;
    vid = other.vid;
  }

  ~VertexSet() {
    if(pooled) {
      buffers_avail.push_back(ptr);
    }
  }

  vidType size() const { return set_size; }
  void adjust_size(vidType s) { set_size = s; }
  vidType get_vid() const { return vid; }
  vidType *data() const { return ptr; }
  const vidType *begin() const { return ptr; }
  const vidType *end() const { return ptr+set_size; }
  void add(vidType v) { ptr[set_size++] = v; }
  void clear() { set_size = 0; }
  vidType& operator[](size_t i) { return ptr[i]; }
  const vidType& operator[](size_t i) const { return ptr[i]; }
  void sort() { std::sort(ptr, ptr+set_size); }

  VertexSet operator &(const VertexSet &other) const {
    VertexSet out;
    vidType idx_l = 0, idx_r = 0;
    while(idx_l < set_size && idx_r < other.set_size) {
      vidType left = ptr[idx_l];
      vidType right = other.ptr[idx_r];
      if(left <= right) idx_l++;
      if(right <= left) idx_r++;
      if(left == right) out.ptr[out.set_size++] = left;
    }
    return out;
  }
  uint32_t get_intersect_num(const VertexSet &other) const {
    uint32_t num = 0;
    vidType idx_l = 0, idx_r = 0;
    while(idx_l < set_size && idx_r < other.set_size) {
      vidType left = ptr[idx_l];
      vidType right = other.ptr[idx_r];
      if(left <= right) idx_l++;
      if(right <= left) idx_r++;
      if(left == right) num++;
    }
    return num;
  }

  void print() const {
    std::copy(ptr, ptr+set_size, std::ostream_iterator<vidType>(std::cout, " "));
  }

  VertexSet intersect(const VertexSet &other, vidType upper) const {
    VertexSet out;
    vidType idx_l = 0, idx_r = 0;
    while(idx_l < set_size && idx_r < other.set_size) {
      vidType left = ptr[idx_l];
      vidType right = other.ptr[idx_r];
      if(left >= upper) break;
      if(right >= upper) break;
      if(left <= right) idx_l++;
      if(right <= left) idx_r++;
      if(left == right) out.ptr[out.set_size++] = left;
    }
    return out;
  }

  vidType intersect_ns(const VertexSet &other, vidType upper) const {
    vidType idx_l = 0, idx_r = 0, idx_out = 0;
    while(idx_l < set_size && idx_r < other.set_size) {
      vidType left = ptr[idx_l];
      vidType right = other.ptr[idx_r];
      if(left >= upper) break;
      if(right >= upper) break;
      if(left <= right) idx_l++;
      if(right <= left) idx_r++;
      if(left == right) idx_out++;
    }
    return idx_out;
  }
};

