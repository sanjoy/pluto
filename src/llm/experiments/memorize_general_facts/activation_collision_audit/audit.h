#pragma once

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/types/span.h"

namespace pluto::llm::activation_collision_audit {

// One scored corpus position. The target is the token predicted FROM this
// activation, not the input token whose activation was recorded.
struct Occurrence {
  int sample = 0;    // Zero-based corpus sample index.
  int position = 0;  // Zero-based input position within that sample.
  int target = 0;    // Required next-token ID, including EOS when appropriate.
  bool operator==(const Occurrence&) const = default;
};

// Distinct observations of exactly the same complete BF16 activation vector.
struct CollisionGroup {
  std::vector<uint16_t> bits;           // Raw BF16 words in channel order.
  std::vector<Occurrence> occurrences;  // Sorted by sample, position, target.
  bool operator==(const CollisionGroup&) const = default;
};

struct Summary {
  int64_t total_rows = 0;       // Number of accepted observations.
  int64_t unique_vectors = 0;   // Number of bitwise-distinct complete vectors.
  int64_t repeated_groups = 0;  // Vectors observed at least twice.
  int64_t conflicting_groups = 0;  // Vectors having more than one target ID.
  int64_t conflicting_rows = 0;    // All rows belonging to conflicting groups.
  // Sum(group size - largest target frequency). This is the minimum number of
  // mistakes for ANY deterministic vector-only classifier on these rows; a
  // classifier that also reads earlier positions is not subject to this bound.
  int64_t minimum_errors = 0;
  std::vector<CollisionGroup> conflicts;  // Sorted lexicographically by bits.
  bool operator==(const Summary&) const = default;
};

// CPU-only audit of exact vector collisions, with no numeric tolerance or
// quantization beyond the BF16 representation supplied by the caller. Positive
// and negative zero remain distinct. Each successful Add counts one observation
// even if its metadata duplicates an earlier observation.
class Audit {
 public:
  // Width must be positive; Add returns InvalidArgument otherwise.
  explicit Audit(int width) : width_(width) {}

  // Copies the complete vector and occurrence. Invalid widths, negative
  // metadata, and nonfinite BF16 values are rejected without changing the
  // audit.
  absl::Status Add(absl::Span<const uint16_t> bits, Occurrence occurrence);

  // Reports all scored rows and retains complete conflicting groups, with
  // deterministic ordering independent of hash-table and insertion order.
  Summary Summarize() const;

 private:
  int width_;
  int64_t total_rows_ = 0;
  absl::flat_hash_map<std::vector<uint16_t>, std::vector<Occurrence>> groups_;
};

}  // namespace pluto::llm::activation_collision_audit
