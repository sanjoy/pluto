#pragma once

#include <filesystem>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/gpt2.h"
#include "src/llm/vocabulary_readout.h"

namespace pluto::llm::kvq_explorer {

// Checkpoints store Q, K, V in this order, each with model_width columns.
// The CLI deliberately labels/reorders them as K, V, Q for display.
enum Projection { kQuery = 0, kKey = 1, kValue = 2 };

using TopThree = TopThreeTokens;

// Stable temperature-one softmax over all vocabulary entries, returning only
// the top three distinct IDs per row. Ties prefer lower IDs. Any nonfinite
// logit invalidates its entire row (all IDs -1 and probabilities NaN).
// Input must be an exact FP32 [rows, vocab_size] buffer on executor; vocab
// >= 3. The result is a device buffer of rows TopThree records, not yet
// CPU-ready.
absl::StatusOr<cuda::Buffer> ReadTopThree(cuda::Executor& executor,
                                          const cuda::Buffer& logits, int rows,
                                          int vocab_size);

// Defaults match the GPT-2 recipe. Small dimensions enable self-contained
// numerical tests. Embedding rows are padded to a multiple of 16 on disk;
// projection widths are unpadded. Block count may select a checkpoint prefix.
struct Dimensions {
  int vocab_size = kGpt2VocabularySize;
  int model_width = kGpt2ModelWidth;
  int block_count = kGpt2TransformerBlockCount;
};

// An isolated FP32 projection diagnostic, not contextual model inference:
//   x = E[token]
//   [q, k, v] = x W_qkv(block) + b_qkv(block)
//   logits_s = s E_logical^T, for s in {q, k, v}.
//
// Uses all concatenated heads, with no positions, LayerNorm, attention,
// output projection, residual, or preceding blocks. Q/K/V coordinates are
// not trained as residual-stream vocabulary logits: this is a chosen readout
// basis, not a prediction of the model's next token or attention weights.
// Executor must outlive the Readout and all returned host arrays.
class Readout final {
 public:
  // Reads E from weight_0.bin, then W_qkv and b_qkv from 4+12*b and 5+12*b.
  // Requires exact file sizes and finite FP32 values, ignores unrelated files,
  // and never mutates a training model. All transfers use page-locked memory.
  static absl::StatusOr<std::unique_ptr<Readout>> Load(
      cuda::Executor& executor, const std::filesystem::path& checkpoint,
      const Dimensions& dimensions = {});

  // One result per token, block and projection, in that order:
  // result[(token_position * block_count + block) * 3 + Projection].
  // Repeated input tokens remain repeated. Empty input returns an empty array.
  // Tokens must belong to executor and be in the logical vocabulary.
  // Processes at most 16 tokens at a time to bound temporary GPU memory;
  // returns only CPU-ready top-three records, never full logits.
  absl::StatusOr<cuda::PageLockedHostArray<TopThree>> Explore(
      cuda::Executor& executor,
      const cuda::PageLockedHostArray<int>& tokens) const;

 private:
  struct BlockWeights {
    cuda::Buffer matrix;
    cuda::Buffer bias;
  };
  Readout(Dimensions dimensions, cuda::Buffer embedding,
          std::vector<BlockWeights> blocks)
      : dimensions_(dimensions),
        embedding_(std::move(embedding)),
        blocks_(std::move(blocks)) {}

  Dimensions dimensions_;
  cuda::Buffer embedding_;
  std::vector<BlockWeights> blocks_;
};

}  // namespace pluto::llm::kvq_explorer
