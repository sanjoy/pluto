#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/gpt2_model.h"

namespace pluto::tokenizer {

// Fast GPT-2 byte-level BPE encoder.
//
// Load() accepts a directory at runtime instead of baking a machine-specific
// dataset location into the library. The directory must contain the
// tokenizer.json produced by Hugging Face's save_pretrained(). Instances are
// safe for concurrent Encode() calls; a bounded cache amortizes BPE work for
// repeated words without allowing memory use to grow with the corpus.
class Gpt2Tokenizer final {
 public:
  static absl::StatusOr<std::unique_ptr<Gpt2Tokenizer>> Load(
      const std::filesystem::path& directory);

  // Returns page-locked storage so callers can upload token IDs with a true
  // asynchronous CUDA transfer and no hidden pageable-memory staging copy.
  absl::StatusOr<cuda::PageLockedHostArray<int>> Encode(
      absl::string_view text) const;

  int vocab_size() const { return model_->vocab_size(); }
  int eos_token_id() const { return model_->eos_token_id(); }
  absl::string_view eos_token() const { return model_->eos_token(); }

 private:
  explicit Gpt2Tokenizer(std::shared_ptr<const internal::Gpt2Model> model)
      : model_(std::move(model)) {}

  absl::Status EncodeOrdinary(absl::string_view text,
                              std::vector<int>* output) const;
  absl::StatusOr<std::vector<int>> ApplyBpe(std::string token) const;

  std::shared_ptr<const internal::Gpt2Model> model_;
  mutable absl::Mutex cache_mutex_;
  mutable absl::flat_hash_map<std::string, std::vector<int>> cache_
      ABSL_GUARDED_BY(cache_mutex_);
};

}  // namespace pluto::tokenizer
