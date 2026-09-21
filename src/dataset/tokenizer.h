#pragma once

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"

namespace pluto::tokenizer {

// Converts text into vocabulary IDs without coupling datasets to a particular
// vocabulary or tokenization algorithm. Concrete implementations define which
// byte strings they accept (for example, GPT-2 requires valid UTF-8).
//
// IDs are signed ints in [0, vocab_size()), matching the dataset token schema.
// Encoding does not automatically prepend or append special tokens. For an
// accepted input, decoding with the matching vocabulary preserves its bytes.
class Tokenizer {
 public:
  virtual ~Tokenizer();

  // Returns CPU-readable IDs in executor-owned page-locked storage. The
  // executor must outlive the returned array and its copies. Uploads and frees
  // must be ordered through that executor; see PageLockedHostArray.
  virtual absl::StatusOr<cuda::PageLockedHostArray<int>> Encode(
      cuda::Executor& executor, absl::string_view text) const = 0;

  // Encodes using only permitted IDs in this tokenizer's ORIGINAL vocabulary.
  // The mask has vocab_size() entries; zero forbids an ID, nonzero allows it.
  // The default validates the ordinary encoding without modifying its storage.
  // Subword tokenizers can override this to choose another exact segmentation
  // when their preferred tokens are unavailable. GPT-2 does so while preserving
  // the ordinary encoding whenever it already uses only permitted tokens.
  virtual absl::StatusOr<cuda::PageLockedHostArray<int>> EncodeWithVocabulary(
      cuda::Executor& executor, absl::string_view text,
      absl::Span<const uint8_t> token_is_allowed) const;

  virtual int vocab_size() const = 0;
};

}  // namespace pluto::tokenizer
