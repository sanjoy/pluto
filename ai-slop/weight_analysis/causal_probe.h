#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layer.h"

namespace pluto::weight_analysis {

// Validation-only utilities. These deliberately do not expose an optimizer,
// backward pass, checkpoint writer, or mechanism for choosing passages by loss.
struct ProbeArm {
  std::string name;
  std::vector<int> weight_indices;
  float scale;
};

// The order is part of the frozen experiment: two clean replays, independent
// half/zero interventions on each output branch, then a final clean replay.
std::vector<ProbeArm> CausalArms();
std::vector<size_t> Gpt2WeightByteSizes();

struct CheckpointFileInfo {
  std::filesystem::path path;
  uintmax_t bytes;
  std::filesystem::file_time_type modified;
};

// Accepts the exact physical GPT-2 weight layout and, optionally, patch.json
// written by paired_weight_patch.py. The metadata is not a weight and does not
// relax any weight count, canonical filename, or byte-size requirement. Rejects
// symlinks at the supplied directory and all entries, and rejects other files
// or subdirectories. This is a layout check, not validation of the JSON
// payload. The returned snapshot lists weights by index, followed by optional
// metadata.
absl::StatusOr<std::vector<CheckpointFileInfo>> InspectGpt2CheckpointFiles(
    const std::filesystem::path& directory);

// Rechecks the complete allowed layout, including optional metadata, against
// the snapshot. These are stat checks, not cryptographic integrity checks;
// evidence runners must still hash input files before and after a probe.
absl::Status VerifyCheckpointFilesUnchanged(
    const std::vector<CheckpointFileInfo>& files);

// Matches checkpoint traversal: the tied embedding is counted only at its
// first occurrence. Returned handles SHARE storage; replacing one would not
// replace the handles cached inside composed layers. Mutate bytes instead.
absl::StatusOr<std::vector<cuda::Buffer>> UniqueWeights(
    cuda::Executor& executor, absl::Span<const cuda::Buffer> weights);
absl::Status ValidateGpt2Weights(absl::Span<const cuda::Buffer> weights);

struct PackedBatch {
  cuda::PageLockedHostArray<int32_t> tokens;
  int context_length;
  int passage_count;
  absl::Span<const int32_t> inputs(int first, int count) const;
  absl::Span<const int32_t> targets(int first, int count) const;
};

// Headerless little-endian int32 [all inputs][all targets]. Each half is
// passage-major. This owns pinned storage for every later asynchronous H2D.
// The supplied executor owns its host-memory pool and must outlive the batch.
// Tokens must be logical vocabulary IDs; adjacent input/target IDs must agree
// with next-token prediction within each independently selected passage.
absl::StatusOr<PackedBatch> LoadPackedBatch(cuda::Executor& executor,
                                            const std::filesystem::path& path,
                                            int context_length, int vocab_size);

// Refuses existing directories, files, and symlinks, including dangling ones.
absl::Status CreateNewOutputDirectory(const std::filesystem::path& path);
std::string JsonQuote(const std::string& text);

// Owns a distinct device snapshot plus pinned original bytes. Apply changes
// the actual allocations, never the composed layer's copied Buffer handles.
// Every normal arm must call RestoreAndVerify explicitly. The destructor is
// only a best-effort safety net on an error path and cannot certify a run.
class WeightIntervention final {
 public:
  static absl::StatusOr<std::unique_ptr<WeightIntervention>> Capture(
      cuda::Executor& executor, absl::Span<const cuda::Buffer> unique_weights,
      absl::Span<const int> indices);
  ~WeightIntervention();
  absl::Status Apply(float scale);
  absl::Status RestoreAndVerify();

 private:
  struct Snapshot {
    cuda::Buffer target;
    cuda::Buffer backup;
    cuda::PageLockedHostArray<float> original;
    cuda::PageLockedHostArray<float> staging;
  };
  explicit WeightIntervention(cuda::Executor& executor) : executor_(executor) {}
  cuda::Executor& executor_;
  std::vector<Snapshot> snapshots_;
  bool dirty_ = false;
};

// A validation-only intervention on selected neuron rows of a row-major FP32
// MLP output matrix W2[input_feature, output]. Bias is intentionally absent:
// changing a neuron's output contribution must not change the branch bias.
// The target Buffer is retained by reference count, so all layer aliases see
// the same bytes even after the caller releases its original handle.
//
// Apply always derives its dose from the pristine snapshot, not the previous
// dose; repeated half/zero/one calls are allowed. Only selected rows are ever
// written. RestoreAndVerify checks the ENTIRE matrix against its snapshot,
// detecting unrelated row mutations rather than silently overwriting them.
// The caller must keep the Executor alive, avoid concurrent matrix mutation,
// and explicitly check restoration before declaring a measurement complete.
// The destructor's best-effort restore is only an error-path safety net.
class MlpRowIntervention final {
 public:
  static absl::StatusOr<std::unique_ptr<MlpRowIntervention>> Capture(
      cuda::Executor& executor, const cuda::Buffer& output_weight,
      int input_features, int output_width, absl::Span<const int> feature_ids);
  ~MlpRowIntervention();
  MlpRowIntervention(const MlpRowIntervention&) = delete;
  MlpRowIntervention& operator=(const MlpRowIntervention&) = delete;

  // Exactly 0, 0.5, or 1. A dose of 1 performs verified byte restoration,
  // including signed zeros, instead of floating-point multiplication.
  absl::Status Apply(float scale);
  absl::Status RestoreAndVerify();

 private:
  MlpRowIntervention(cuda::Executor& executor, cuda::Buffer target,
                     cuda::Buffer backup,
                     cuda::PageLockedHostArray<float> original,
                     cuda::PageLockedHostArray<float> staging,
                     std::vector<int> feature_ids, size_t output_width);
  cuda::Executor& executor_;
  cuda::Buffer target_;
  cuda::Buffer backup_;
  cuda::PageLockedHostArray<float> original_;
  cuda::PageLockedHostArray<float> staging_;
  std::vector<int> feature_ids_;
  size_t output_width_;
  bool dirty_ = false;
};

struct Measurements {
  cuda::PageLockedHostArray<float> losses;
  cuda::PageLockedHostArray<int32_t> argmax;
};

// Runs the real forward/loss interfaces, retaining per-token values rather
// than averaging across passages. Microbatching changes execution grouping
// only: every passage starts at position zero and output order is unchanged.
// Logits are FP32 even when model weights are cast to BF16 for forward work.
absl::StatusOr<Measurements> EvaluatePassages(
    cuda::Executor& executor, const llm::Layer& model,
    const llm::Layer& loss_layer, const PackedBatch& batch, int batch_sequences,
    int vocab_size, int padded_vocab_size);

// Exclusive creation prevents accidental replacement even inside a new run
// directory. Values are already validated and native byte order must be LE.
absl::Status WriteExclusive(const std::filesystem::path& path,
                            const void* bytes, size_t size);

}  // namespace pluto::weight_analysis
