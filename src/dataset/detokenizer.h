#pragma once

#include <string>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace pluto::tokenizer {

// Converts token IDs back to their exact vocabulary bytes. This interface is
// separate from Tokenizer so callers that only decode need no CUDA executor
// or encoding machinery. A byte tokenizer may implement both interfaces.
//
// The caller supplies CPU-readable IDs; this interface performs no CUDA copy.
// A decoded token sequence need not be valid UTF-8: individual subword tokens
// can split a multibyte character.
class Detokenizer {
 public:
  virtual ~Detokenizer();

  // Returns InvalidArgument for any ID outside [0, vocab_size()). Empty input
  // produces an empty string; embedded NUL bytes are preserved.
  virtual absl::StatusOr<std::string> Decode(
      absl::Span<const int> token_ids) const = 0;

  virtual int vocab_size() const = 0;
};

}  // namespace pluto::tokenizer
