#pragma once

#include <vector>
#include <stdint.h>
#include <string.h>
#include "common.hpp"

template <typename T>
class pqueue_t {
private:
  int queue_size;
  int queue_capacity;
  int next_idx;
  std::vector<T> vid_queue;
  std::vector<float> distances;
  std::vector<uint8_t> expanded;

public:
  pqueue_t(int L) {
    next_idx = 0;
    queue_size = 0;
    queue_capacity = L;
    //vid_queue = new T[queue_capacity];
    //distances = new float[queue_capacity];
    vid_queue.resize(L);
    distances.resize(L);
    expanded.resize(L);
    std::fill(expanded.begin(), expanded.end(), 0);
  }
	/*
  ~pqueue_t() {
    delete [] vid_queue;
    delete [] distances;
  }
	*/
  T& operator[](size_t i) { return vid_queue[i]; }
  const T& operator[](size_t i) const { return vid_queue[i]; }
  int get_capacity() const { return queue_capacity; }
  int get_next_index() const { return next_idx; }
  float get_tail_dist() const { return queue_size == 0 ? FLT_MAX : distances[queue_size-1]; }
  float get_dist(int idx) const { return distances[idx]; }
  bool is_expanded(int idx) const { return expanded[idx] == 1 ? true : false; }
  int size() { return queue_size; }
  void clear() { queue_size = 0; next_idx = 0; }

  void set_expanded(int idx) { expanded[idx] = 1; }
  void set_unexpanded(int idx) { expanded[idx] = 0; }

  bool split_queue(std::vector<pqueue_t> &lqs) {
    if (queue_size == 0) return false;
    int num_tasks = 0;
    size_t j = 0;
    for (int i = next_idx; i < queue_size; i++) {
      if (expanded[i]) continue;
      //auto loc = lqs[j].push(vid_queue[i], distances[i]);
      auto loc = lqs[j].insert(vid_queue[i], distances[i]);
      if (loc >= 0) {
        set_expanded(i);
        num_tasks ++;
      } else break;
      j ++; // round robin assignment
      if (j == lqs.size()) j = 0;
    }
    if (num_tasks == 0) return false;
    return true;
  }

  int merge_queues(std::vector<pqueue_t> &lqs) {
    // linear insertion
    // TODO: do 'merge' instead to improve performance
    for (size_t i = 0; i < lqs.size(); i++) {
      for (int j = 0; j < lqs[i].size(); j++) {
        auto loc = push(lqs[i][j], lqs[i].get_dist(j));
        if (loc == -2) { // Duplicate
          if (!lqs[i].is_expanded(j) && is_expanded(loc)) {
            set_unexpanded(loc);
            if (loc < next_idx) next_idx = loc;
          }
        } else if (loc >= 0) {
          if (lqs[i].is_expanded(j)) {
            set_expanded(loc);
            if (loc < next_idx) next_idx++;
          } else if (loc < next_idx) next_idx = loc;
        }
      }
    }
    return next_idx;
  }

  int insert(T vid, float dist) {
    if (queue_size == queue_capacity) return -1;
    vid_queue[queue_size] = vid;
    distances[queue_size] = dist;
    expanded[queue_size] = 0;
    return queue_size ++;
  }
  int push(std::pair<float, T> element) {
    auto vid = element.first;
    auto dist = element.second;
    return push(vid, dist);
  }
  int push(T vid, float dist) {
    //printf("pushing %d with dist %f into the queue\n", vid, dist);
    if (0 == queue_size) {
      vid_queue[0] = vid;
      queue_size = 1;
      distances[0] = dist;
      expanded[0] = 0; // not expanded yet
      return 0;
    }
    // Find the insert location
    auto queue_start = &distances[0];
    const auto loc = std::lower_bound(queue_start, queue_start + queue_size, dist);
    auto insert_loc = loc - queue_start;
    if (insert_loc != queue_size) {
      if (vid == vid_queue[insert_loc]) { // Duplicate, skip
        return -2;
      }
      if (queue_size >= queue_capacity) { // Queue is full
        --queue_size;
      }
    } else { // insert_loc == queue_size, insert at the end?
      if (queue_size < queue_capacity) { // Queue is not full
        // Insert at the end
        vid_queue[insert_loc] = vid;
        distances[insert_loc] = dist;
        expanded[insert_loc] = 0; // not expanded yet
        ++queue_size;
        return insert_loc;
      } else { // Queue is full, skip
        return -1;
      }
    }
    // Add into queue
    auto num = queue_size - insert_loc;
    memmove(reinterpret_cast<char *>(&vid_queue[insert_loc + 1]),
            reinterpret_cast<char *>(&vid_queue[insert_loc]),
            num * sizeof(T));
    vid_queue[insert_loc] = vid;
    memmove(reinterpret_cast<char *>(&distances[insert_loc + 1]),
            reinterpret_cast<char *>(&distances[insert_loc]),
            num * sizeof(float));
    distances[insert_loc] = dist;
    memmove(reinterpret_cast<char *>(&expanded[insert_loc + 1]),
            reinterpret_cast<char *>(&expanded[insert_loc]),
            num * sizeof(uint8_t));
    expanded[insert_loc] = 0; // not expanded yet
    ++queue_size;
    if (insert_loc < next_idx) next_idx = insert_loc;
    return insert_loc;
  }

  T* fetch_unexpanded_nodes(int P) {
    T* nodes = new T[P];
    int count = 0;
    for (int i = 0; i < queue_size; i++) {
      if (expanded[i]) nodes[count++] = i;
      if (count == P) break;
    }
    return nodes;
  }
};

