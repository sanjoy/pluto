#pragma once

#include <filesystem>
#include <functional>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/llm/layer_hooks.h"
#include "src/llm/qwen/checkpoint.h"

namespace pluto::llm::qwen {

struct InferenceOptions {
  // Bounds both the full-attention KV cache and the accepted token history.
  int context_length = 512;
  // Receives the number of loaded decoder blocks and the total block count.
  std::function<void(int, int)> load_progress;
};

// Stateful, batch-one Qwen3.8 text inference through a composed Pluto Layer
// graph. Decoder blocks use ordinary residual, parallel, and composed layers.
// FP8 weights stay compressed on the GPU; no gradients or optimizer storage is
// allocated. Vision, MTP, training and batched prefill are not implemented.
// The executor must outlive this object and every returned logits buffer.
class Model final {
 public:
  static absl::StatusOr<std::unique_ptr<Model>> Load(
      cuda::Executor& executor, const std::filesystem::path& checkpoint,
      const InferenceOptions& options = {});
  ~Model();

  // Consumes one token, advancing every recurrent/KV cache. Call for each
  // prompt token, then each generated token. Reset is required after an error.
  // Hooks see all layers, including projections and the named decoder blocks.
  // Calls on the same model must not execute concurrently (caches are mutable).
  absl::Status Step(int token, LayerHooks* hooks = nullptr);
  // Returns device FP32 logits for the token following the consumed history.
  // Calling this does not advance the history. At least one Step is required.
  absl::StatusOr<cuda::Buffer> Logits(LayerHooks* hooks = nullptr);
  absl::Status Reset();
  const Config& config() const;
  int position() const;
  size_t weight_bytes() const;

 private:
  struct Impl;
  explicit Model(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace pluto::llm::qwen
