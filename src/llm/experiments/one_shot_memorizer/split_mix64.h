#pragma once

#include <cassert>
#include <cstdint>

namespace pluto::llm::one_shot_memorizer::internal {

// Reproducible integer generator: arithmetic deliberately wraps modulo 2^64.
// This is not a cryptographic generator. A bounded draw requires bound > 0.
class SplitMix64 {
 public:
  explicit SplitMix64(uint64_t seed) : state_(seed) {}

  uint64_t Bounded(uint64_t bound) {
    assert(bound > 0);
    // Rejecting the first 2^64 % bound values leaves an exact multiple of
    // bound.
    const uint64_t threshold = (uint64_t{0} - bound) % bound;
    uint64_t value;
    do {
      value = Next();
    } while (value < threshold);
    return value % bound;
  }

 private:
  uint64_t Next() {
    uint64_t value = (state_ += UINT64_C(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
  }

  uint64_t state_;
};

}  // namespace pluto::llm::one_shot_memorizer::internal
