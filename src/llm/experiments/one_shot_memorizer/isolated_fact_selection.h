#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::llm::one_shot_memorizer {

// A small training corpus selected by original corpus line identity. Identical
// text on different lines remains distinct; selection never deduplicates text.
struct IsolatedFactSelection {
  // Zero-based original corpus indices, in increasing original line order.
  std::vector<size_t> corpus_indices;
  // Selected line bytes joined by LF, with no additional final newline.
  // A line's trailing CR is preserved for the dataset's CRLF handling.
  std::string text;

  // Maps a selected-dataset sample index back to the original corpus index.
  // Returns OutOfRange rather than silently confusing local/global indices.
  absl::StatusOr<size_t> CorpusIndex(size_t local_sample_index) const;
};

// Parses a nonempty list of distinct 1-based decimal line numbers, then emits
// the selected lines in original corpus order, not argument order. Numbers
// contain ASCII digits only; zero, overflow, duplicates (including alternate
// leading-zero spellings), and out-of-corpus indices are invalid. Selected
// blank/whitespace-only lines and embedded LF are rejected, never filtered:
// one selected entry must remain exactly one sample. Other line bytes are
// copied verbatim, including trailing CR and leading/trailing spaces.
absl::StatusOr<IsolatedFactSelection> SelectIsolatedFacts(
    absl::Span<const std::string> corpus_lines,
    absl::Span<const std::string> line_numbers_1based);

}  // namespace pluto::llm::one_shot_memorizer
