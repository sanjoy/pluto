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
// CreateActivationGenerator(). The source must provide separate int32 input
// and target token buffers with batch_size * sequence_length elements each,
// as InMemoryDataSetIterator does. Each Next() forwards the input buffer to the
// generator without copying it. Targets are validated but otherwise unused
// because the resulting activations themselves are the SAE examples.
//
// The iterator borrows executor, activation_generator, and source. All three
// must outlive it. The generator is used only for forward inference; its state
// is discarded after each batch and no gradients are computed. Returned
// DataBatch inputs and targets share the generator's single output allocation
// without copying it. Each contains batch_size * sequence_length hidden-state
// rows. The handles keep that allocation alive, but a generator that reuses
// storage may overwrite its contents on the next call. Both sample dimensions
// are preserved from the source; row width and element type are defined by
// activation_generator. Its declared input and output ActivationTypes must
// preserve [batch, sequence_length] axes; signatures are checked before fwd
// and returned buffer sizes/executors afterward. In particular, matching total
// row counts cannot disguise different sample boundaries. Because raw Buffer
// bytes have no dtype tag, the source and generator must honor their declared
// element types.
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
