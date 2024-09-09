#pragma once

#include <vector>
#include <stdint.h>
#include <string.h>
#include "common.hpp"

template <typename T>
class hash_filter {
private:
  int bits;
  int capacity;
  std::vector<T> seen;
  std::hash<T> hasher;
/*
parlay uses:
inline uint64_t hash64_2(uint64_t x) {
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  x = x ^ (x >> 31);
  return x;
}
*/
public:
  hash_filter(int L) {
    bits = std::__lg(L)+1; // round up to power of 2
    L = (1 << bits);
    capacity = L;
    seen.resize(L);
    fill(seen.begin(),seen.end(),-1); // note: -1 should not be a possible index
  }
  int get_hash(T thing) {
    //return hasher(thing) & ((1 << bits)-1);
    return thing & ((1 << bits)-1);
  }
  bool contains(T thing) {
    int h = get_hash(thing);
    return (seen[h] == thing);
  }
  bool add(T thing) { // returns true if successfully added
    int h = get_hash(thing);
    if (seen[h] == thing) return false;
    seen[h] = thing;
    return true;
  }
};
