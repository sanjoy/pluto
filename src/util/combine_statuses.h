#pragma once

#include "absl/status/status.h"
#include "absl/types/span.h"

namespace pluto::util {

// Combines independent operations' results without discarding later failures.
// Returns OK for an empty span or when every status is OK. Otherwise the result
// has the first failed status's code and a message listing every failed status,
// in input order, with its original zero-based index and complete ToString().
// OK entries are omitted, but do not renumber the failed entries. For example:
//
//   Failures:
//     [1] INVALID_ARGUMENT: first
//     [3] NOT_FOUND: other
//
// The input statuses are unchanged. Their codes and messages are preserved in
// the combined message; their structured payloads are not copied to the result.
absl::Status CombineStatuses(absl::Span<const absl::Status> statuses);

}  // namespace pluto::util
