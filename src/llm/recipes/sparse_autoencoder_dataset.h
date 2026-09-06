#pragma once

#include <filesystem>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/cuda/executor.h"
#include "src/dataset/dataset.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// One batch of hidden states suitable as sparse-autoencoder input.
// activations contains batch_size rows; the row width and element type are
// defined by the activation generator used to create the iterator.
struct ActivationBatch {
  Buffer activations;
  int batch_size;
};

// Lazily transforms token batches into hidden-state batches.
//
// Create() first restores activation_generator from checkpoint_directory.
// ReadFromDirectory() accepts a complete-model checkpoint when the generator's
// weights form its prefix, which is the normal use with
// CreateActivationGenerator(). Each Next() then passes the source batch's token
// buffer through the generator. Next-token targets are deliberately ignored:
// the resulting activations themselves are the examples consumed by an SAE.
//
// The iterator borrows executor, activation_generator, and source. All three
// must outlive it. The generator is used only for forward inference; its tape
// is discarded after each batch and no gradients are computed.
class SparseAutoEncoderDataSetIterator final {
 public:
  static absl::StatusOr<std::unique_ptr<SparseAutoEncoderDataSetIterator>>
  Create(cuda::Executor& executor, Layer& activation_generator,
         DataSetIterator& source,
         const std::filesystem::path& checkpoint_directory);

  absl::StatusOr<ActivationBatch> Next();

  // Restores the wrapped source's original sequence.
  absl::Status Reset();

 private:
  SparseAutoEncoderDataSetIterator(cuda::Executor& executor,
                                   const Layer& activation_generator,
                                   DataSetIterator& source)
      : executor_(executor),
        activation_generator_(activation_generator),
        source_(source) {}

  cuda::Executor& executor_;
  const Layer& activation_generator_;
  DataSetIterator& source_;
};

}  // namespace pluto::llm
