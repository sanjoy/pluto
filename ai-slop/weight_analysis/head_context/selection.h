#pragma once

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::weight_analysis::head_context {

// These scopes partition the queries WITHIN ONE selected sequence. All other
// sequences and heads remain byte-identical. "Other" includes every position
// except query_position, including later/padded positions; a causal tail must
// prevent those later positions from influencing the selected query's logits.
enum class QueryScope { kSelectedQuery, kOtherQueries, kAllQueries };

// Explicit geometry keeps the CPU operation independent of CUDA/the recipe.
// A native GPT-2 replay must separately require its actual recipe geometry.
struct Geometry {
  int blocks;
  int context_length;
  int heads;
  int head_dimension;
};

struct Selection {
  int block;
  int head;
  int sequence;
  int query_position;
  QueryScope scope;
  float scale;
};

// Validates dimensions, signed-int kernel element limits and the exact doses
// 0, 0.5, 1. No device allocation, CUDA context, or model execution occurs.
absl::Status ValidateSelection(const Geometry& geometry,
                               const Selection& selection, int total_rows);

// Exact finite-BF16 scaling, including subnormal halfway ties. Zero writes
// positive zero; identity and half preserve the sign of zero. No CPU BF16
// extension is needed. NaN/infinity and undeclared doses are rejected.
absl::StatusOr<uint16_t> ScaleBf16(uint16_t bits, float scale);

// Copy the entire [total_rows, heads * head_dimension] context, scaling only
// the declared sequence/query-scope/head. Input and destination must be
// disjoint. All validation finishes before destination is written, so errors
// leave it unchanged. The caller may supply page-locked destination storage
// for a subsequent H2D transfer; this helper itself performs no GPU transfer.
//
// These are CONTEXT interventions after attention, not one-source-V edits,
// key masking/renormalization, or changes to the head's projection weights.
// A native caller must rerun the real downstream tail even at identity dose,
// and require exact clean all-row/padded-logit parity before interpreting it.
absl::Status ScaleContext(absl::Span<const uint16_t> original,
                          absl::Span<uint16_t> destination, int total_rows,
                          const Geometry& geometry, const Selection& selection);

}  // namespace pluto::weight_analysis::head_context
