#pragma once

#include <iosfwd>
#include <string>

#include "absl/status/status.h"
#include "src/cuda/executor.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/llm/gpt2.h"

namespace pluto::llm::memorize_general_facts {

// A focused reproduction of the frozen-third-attention MLP-readout puzzle.
// Source dimensions describe the checkpoint, not the replacement MLP.
struct PuzzleOptions {
  std::string checkpoint;  // Complete, already-memorized GPT-2 checkpoint.
  std::string corpus;      // The 1,024 original facts, one per line.
  std::string output_directory;  // Must be new; reports never overwrite a run.
  Gpt2Config model_config;  // Vocabulary size is resolved from the tokenizer.
  bool compact_vocabulary = true;  // Load the checkpoint's saved ID mapping.
  bool train_mlp = false;  // Fit one residual MLP; exclusive with below.
  bool train_stacked_mlp = false;  // Fit five residual 10 -> 150 -> 10 MLPs.
  int mlp_width = 150;  // Single-MLP minimum expansion; ignored for the stack.
  int steps = 300000;   // Exact update budget; zero evaluates initialization.
  int eval_every = 1000;        // Full-corpus statistics cadence.
  int batch_size = 32;          // Facts per batch, not flattened token rows.
  int seed = 3;                 // Fresh MLP initialization and shuffle seed.
  float learning_rate = 0.01f;  // Cosine decay to one tenth of this rate.
};

// Capture the post-attention residual of transformer block 3 (one-based),
// verify that exact states cannot have different scored targets, and write a
// standalone coordinate-pair plot for every position. Only suffix/EOS targets
// after five-token prompts enter the audit and training; padding is labeled
// explicitly in plots. Optional fitting freezes the source and tied head and
// trains either one fresh residual MLP or the fixed five-MLP stack, each with
// its own input LN, plus one final LN using cached A3 vectors. With neither
// training flag set, only capture, audit, and plotting run. All GPU work and
// pinned-memory transfers use the supplied executor.
absl::Status RunPuzzle(cuda::Executor& executor,
                       const tokenizer::Gpt2Tokenizer& base_tokenizer,
                       const PuzzleOptions& options, std::ostream& output);

}  // namespace pluto::llm::memorize_general_facts
