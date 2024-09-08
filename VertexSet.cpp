#include "VertexSet.hpp"
#include <cassert>

// VertexSet static members
thread_local std::vector<vidType*> VertexSet::buffers_exist(0);
thread_local std::vector<vidType*> VertexSet::buffers_avail(0);
vidType VertexSet::MAX_DEGREE = 1;

void VertexSet::release_buffers() {
  buffers_avail.clear();
  while(buffers_exist.size() > 0) {
    delete[] buffers_exist.back();
    buffers_exist.pop_back();
  }
}

