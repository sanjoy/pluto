#pragma once

#include <filesystem>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace pluto::llm::qwen {

// Evaluate one expression, or open a line-editing REPL if expression is empty.
// Loads only the tokenizer and input embedding table, never decoder weights.
// Piped input also works; failed lines are reported without discarding later
// lines, and any failure makes the noninteractive session return an error.
absl::Status RunEmbeddingAlgebra(const std::filesystem::path& checkpoint,
                                 absl::string_view expression);

}  // namespace pluto::llm::qwen
