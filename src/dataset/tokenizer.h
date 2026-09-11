#pragma once

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
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

  virtual int vocab_size() const = 0;
};

}  // namespace pluto::tokenizer
