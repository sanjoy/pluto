#pragma once

#include <filesystem>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "src/cuda/executor.h"
#include "src/dataset/dataset.h"
#include "src/llm/layer.h"

namespace pluto::llm {

// Lazily transforms token batches into hidden-state batches.
//
// Create() first restores activation_generator from checkpoint_directory.
// ReadFromDirectory() accepts a complete-model checkpoint when the generator's
// weights form its prefix, which is the normal use with
// CreateActivationGenerator(). The source must use
// InMemoryDataSetIterator's packed next-token schema. Each Next() passes its
// input-token half through the generator; the target-token half is deliberately
// ignored because the resulting activations themselves are the SAE examples.
//
// The iterator borrows executor, activation_generator, and source. All three
// must outlive it. The generator is used only for forward inference; its tape
// is discarded after each batch and no gradients are computed. Returned
// DataBatch::data buffers contain batch_size * sequence_length hidden-state
// rows. Both sample dimensions are preserved from the source; row width and
// element type are defined by activation_generator. Its sequence-width contract
// is checked before running it; fixed-context attention/position generators
// reject shorter samples even when their total row count is divisible by the
// configured context width.
class SparseAutoEncoderDataSetIterator final : public DataSetIterator {
 public:
  static absl::StatusOr<std::unique_ptr<SparseAutoEncoderDataSetIterator>>
  Create(cuda::Executor& executor, Layer& activation_generator,
         DataSetIterator& source,
         const std::filesystem::path& checkpoint_directory);

  absl::StatusOr<DataBatch> Next() override;

  // Restores the wrapped source's original sequence.
  absl::Status Reset() override;

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
