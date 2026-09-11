#include "ai-slop/weight_analysis/causal_probe.h"

#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <system_error>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "ai-slop/weight_analysis/token_argmax.h"
#include "src/util/status_macros.h"

namespace pluto::weight_analysis {
namespace {
absl::Status Transfer(cuda::Executor& executor, void* destination,
                      const void* source, size_t size, cudaMemcpyKind kind) {
  return cuda::CudaStatus(
      cudaMemcpyAsync(destination, source, size, kind, executor.stream()),
      "causal probe cudaMemcpyAsync");
}
}  // namespace

std::vector<ProbeArm> CausalArms() {
  std::vector<ProbeArm> arms{{"clean_before", {}, 1}, {"clean_repeat", {}, 1}};
  for (int block = 0; block < 8; ++block) {
    for (bool mlp : {false, true}) {
      int first = 12 * block + (mlp ? 12 : 6);
      const std::string prefix =
          absl::StrCat("block", block, mlp ? "_mlp_" : "_attention_");
      arms.push_back({prefix + "half", {first, first + 1}, 0.5f});
      arms.push_back({prefix + "zero", {first, first + 1}, 0.0f});
    }
  }
  arms.push_back({"clean_after", {}, 1});
  return arms;
}

std::vector<size_t> Gpt2WeightByteSizes() {
  // Explicit physical dimensions, including embedding padding. These are
  // checked against the compiled recipe, not used to reinterpret raw weights
  // from arbitrary architectures (in particular, an SAE checkpoint fails).
  std::vector<size_t> sizes{50272ULL * 512 * 4, 1024ULL * 512 * 4};
  for (int block = 0; block < 8; ++block) {
    for (size_t count :
         {512ULL, 512ULL, 512ULL * 1536, 1536ULL, 512ULL * 512, 512ULL, 512ULL,
          512ULL, 512ULL * 2048, 2048ULL, 2048ULL * 512, 512ULL}) {
      sizes.push_back(count * sizeof(float));
    }
  }
  sizes.push_back(512 * sizeof(float));
  sizes.push_back(512 * sizeof(float));
  return sizes;
}

absl::StatusOr<std::vector<CheckpointFileInfo>> InspectGpt2CheckpointFiles(
    const std::filesystem::path& directory) {
  namespace fs = std::filesystem;
  // Filesystem failures are recoverable input errors. Use error_code overloads
  // explicitly: the build disables exceptions, so throwing overloads cannot be
  // caught here (and allocation failures intentionally terminate the process).
  std::error_code error;
  const auto inspection_error = [&] {
    return absl::InvalidArgumentError(
        absl::StrCat("cannot inspect checkpoint: ", error.message()));
  };
  const auto directory_status = fs::symlink_status(directory, error);
  if (error)
    return inspection_error();
  if (!fs::is_directory(directory_status)) {
    return absl::InvalidArgumentError(
        "checkpoint must be a directory, not a symlink");
  }
  const auto sizes = Gpt2WeightByteSizes();
  absl::flat_hash_set<std::string> expected_names;
  for (size_t i = 0; i < sizes.size(); ++i)
    expected_names.insert(absl::StrCat("weight_", i, ".bin"));
  bool has_metadata = false;
  fs::directory_iterator entry(directory, error);
  if (error)
    return inspection_error();
  const fs::directory_iterator end;
  while (entry != end) {
    const auto status = entry->symlink_status(error);
    if (error)
      return inspection_error();
    if (!fs::is_regular_file(status)) {
      return absl::InvalidArgumentError(
          "checkpoint contains a symlink or non-regular entry");
    }
    const auto name = entry->path().filename().string();
    if (name == "patch.json") {
      has_metadata = true;
    } else if (expected_names.erase(name) != 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("unexpected checkpoint entry: ", name));
    }
    entry.increment(error);
    if (error)
      return inspection_error();
  }
  if (!expected_names.empty()) {
    return absl::InvalidArgumentError(
        "checkpoint must contain all 100 canonical GPT-2 weight files");
  }
  std::vector<CheckpointFileInfo> files;
  files.reserve(sizes.size() + has_metadata);
  // Recheck each file while collecting its metadata, since the directory may
  // have changed since enumeration. Reject symlinks as before.
  for (size_t i = 0; i < sizes.size() + has_metadata; ++i) {
    const bool metadata = i == sizes.size();
    const auto path =
        directory /
        (metadata ? "patch.json" : absl::StrCat("weight_", i, ".bin"));
    const auto status = fs::symlink_status(path, error);
    if (error)
      return inspection_error();
    if (!fs::is_regular_file(status)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "checkpoint contains a non-regular file: ", path.string()));
    }
    const auto bytes = fs::file_size(path, error);
    if (error)
      return inspection_error();
    if (!metadata && bytes != sizes[i]) {
      return absl::InvalidArgumentError(
          absl::StrCat("checkpoint weight layout mismatch: ", path.string()));
    }
    const auto modified = fs::last_write_time(path, error);
    if (error)
      return inspection_error();
    files.push_back({path, bytes, modified});
  }
  return files;
}

absl::Status VerifyCheckpointFilesUnchanged(
    const std::vector<CheckpointFileInfo>& files) {
  if (files.empty())
    return absl::InvalidArgumentError("empty checkpoint stat snapshot");
  const auto current =
      InspectGpt2CheckpointFiles(files.front().path.parent_path());
  if (!current.ok() || current->size() != files.size())
    return absl::DataLossError("checkpoint layout changed during probe");
  for (size_t i = 0; i < files.size(); ++i) {
    if ((*current)[i].path != files[i].path ||
        (*current)[i].bytes != files[i].bytes ||
        (*current)[i].modified != files[i].modified) {
      return absl::DataLossError("checkpoint file stat changed during probe");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<cuda::Buffer>> UniqueWeights(
    cuda::Executor& executor, absl::Span<const cuda::Buffer> weights) {
  std::vector<cuda::Buffer> unique;
  absl::flat_hash_set<const void*> seen;
  for (const auto& weight : weights) {
    if (&weight.executor() != &executor || weight.size_bytes() == 0)
      return absl::InvalidArgumentError("wrong executor or empty weight");
    if (seen.insert(weight.data()).second)
      unique.push_back(weight);
  }
  return unique;
}

absl::Status ValidateGpt2Weights(absl::Span<const cuda::Buffer> weights) {
  const auto sizes = Gpt2WeightByteSizes();
  if (weights.size() != sizes.size())
    return absl::InvalidArgumentError("expected 100 unique GPT-2 weights");
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (weights[i].size_bytes() != sizes[i]) {
      return absl::InvalidArgumentError(
          absl::StrCat("weight size mismatch: ", i));
    }
  }
  return absl::OkStatus();
}

absl::Span<const int32_t> PackedBatch::inputs(int first, int count) const {
  return tokens.span().subspan(static_cast<size_t>(first) * context_length,
                               static_cast<size_t>(count) * context_length);
}
absl::Span<const int32_t> PackedBatch::targets(int first, int count) const {
  return tokens.span().subspan(
      static_cast<size_t>(passage_count + first) * context_length,
      static_cast<size_t>(count) * context_length);
}

absl::StatusOr<PackedBatch> LoadPackedBatch(cuda::Executor& executor,
                                            const std::filesystem::path& path,
                                            int context_length,
                                            int vocab_size) {
  static_assert(sizeof(float) == 4 && sizeof(int32_t) == 4);
  static_assert(std::numeric_limits<float>::is_iec559);
  if constexpr (std::endian::native != std::endian::little)
    return absl::UnimplementedError("probe requires a little-endian host");
  if (context_length <= 0 || vocab_size <= 0)
    return absl::InvalidArgumentError("invalid input shape or vocabulary");
  std::error_code error;
  const uintmax_t bytes = std::filesystem::file_size(path, error);
  if (error)
    return absl::InvalidArgumentError("cannot stat token batch");
  const uint64_t passage_bytes = uint64_t{2} * context_length * sizeof(int32_t);
  if (!bytes || bytes % passage_bytes ||
      bytes / (2 * sizeof(int32_t)) > std::numeric_limits<int>::max()) {
    return absl::InvalidArgumentError("invalid token batch byte size");
  }
  ASSIGN_OR_RETURN(auto tokens, cuda::PageLockedHostArray<int32_t>::Allocate(
                                    executor, bytes / 4));
  std::ifstream input(path, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(tokens.data()), bytes) ||
      input.peek() != std::ifstream::traits_type::eof()) {
    return absl::DataLossError("short or changed token batch");
  }
  for (int32_t token : tokens)
    if (token < 0 || token >= vocab_size)
      return absl::InvalidArgumentError("token outside logical vocabulary");
  PackedBatch batch{std::move(tokens), context_length,
                    static_cast<int>(bytes / passage_bytes)};
  for (int i = 0; i < batch.passage_count; ++i) {
    auto x = batch.inputs(i, 1);
    auto y = batch.targets(i, 1);
    for (int j = 0; j + 1 < context_length; ++j)
      if (y[j] != x[j + 1])
        return absl::InvalidArgumentError("targets are not next-token shifted");
  }
  return batch;
}

absl::Status CreateNewOutputDirectory(const std::filesystem::path& path) {
  std::error_code error;
  // create_directory is atomic, and unlike exists() does not mistake a
  // dangling symlink for an available destination. Parent must already exist.
  if (!std::filesystem::create_directory(path, error)) {
    return absl::AlreadyExistsError(
        "output directory exists or cannot be created");
  }
  return absl::OkStatus();
}

std::string JsonQuote(const std::string& text) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string result = "\"";
  for (unsigned char c : text) {
    if (c == '"' || c == '\\') {
      result += '\\';
      result += c;
    } else if (c < 32) {
      result += "\\u00";
      result += hex[c >> 4];
      result += hex[c & 15];
    } else {
      result += c;
    }
  }
  return result + '"';
}

absl::StatusOr<std::unique_ptr<WeightIntervention>> WeightIntervention::Capture(
    cuda::Executor& executor, absl::Span<const cuda::Buffer> weights,
    absl::Span<const int> indices) {
  if (indices.empty())
    return absl::InvalidArgumentError("empty intervention");
  auto result = absl::WrapUnique(new WeightIntervention(executor));
  absl::flat_hash_set<const void*> seen;
  for (int index : indices) {
    if (index < 0 || static_cast<size_t>(index) >= weights.size() ||
        &weights[index].executor() != &executor ||
        weights[index].size_bytes() == 0 ||
        weights[index].size_bytes() % sizeof(float) ||
        !seen.insert(weights[index].data()).second) {
      return absl::InvalidArgumentError(
          "invalid or aliased intervention weight");
    }
    const auto& weight = weights[index];
    ASSIGN_OR_RETURN(auto backup,
                     cuda::Buffer::Allocate(executor, weight.size_bytes()));
    ASSIGN_OR_RETURN(auto original,
                     cuda::PageLockedHostArray<float>::Allocate(
                         executor, weight.size_bytes() / sizeof(float)));
    ASSIGN_OR_RETURN(auto staging,
                     cuda::PageLockedHostArray<float>::Allocate(
                         executor, weight.size_bytes() / sizeof(float)));
    // Keep every transfer endpoint owned before issuing asynchronous work.
    result->snapshots_.push_back(
        {weight, std::move(backup), std::move(original), std::move(staging)});
    auto& snapshot = result->snapshots_.back();
    RETURN_IF_ERROR(Transfer(executor, snapshot.backup.data(), weight.data(),
                             weight.size_bytes(), cudaMemcpyDeviceToDevice));
    RETURN_IF_ERROR(Transfer(executor, snapshot.original.data(), weight.data(),
                             weight.size_bytes(), cudaMemcpyDeviceToHost));
    RETURN_IF_ERROR(executor.Synchronize());
    for (float value : snapshot.original)
      if (!std::isfinite(value))
        return absl::InvalidArgumentError("nonfinite intervention weight");
  }
  return result;
}

WeightIntervention::~WeightIntervention() {
  if (dirty_) {
    const auto status = RestoreAndVerify();
    if (!status.ok())
      std::cerr << "Emergency weight restore failed: " << status << '\n';
  }
  // Also protect pinned snapshot lifetimes after a failed asynchronous copy.
  (void)executor_.Synchronize();
}

absl::Status WeightIntervention::Apply(float scale) {
  if (dirty_ || (scale != 0.0f && scale != 0.5f))
    return absl::InvalidArgumentError("restore first; scale must be 0 or 0.5");
  dirty_ = true;
  for (auto& snapshot : snapshots_) {
    if (scale == 0) {
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemsetAsync(snapshot.target.data(), 0,
                          snapshot.target.size_bytes(), executor_.stream()),
          "zero branch output weight"));
    } else {
      // Always scale the pristine FP32 originals, never a preceding dose.
      for (size_t i = 0; i < snapshot.original.size(); ++i)
        snapshot.staging[i] = snapshot.original[i] * scale;
      RETURN_IF_ERROR(
          Transfer(executor_, snapshot.target.data(), snapshot.staging.data(),
                   snapshot.target.size_bytes(), cudaMemcpyHostToDevice));
    }
  }
  return executor_.Synchronize();
}

absl::Status WeightIntervention::RestoreAndVerify() {
  for (auto& snapshot : snapshots_) {
    RETURN_IF_ERROR(
        Transfer(executor_, snapshot.target.data(), snapshot.backup.data(),
                 snapshot.target.size_bytes(), cudaMemcpyDeviceToDevice));
    RETURN_IF_ERROR(
        Transfer(executor_, snapshot.staging.data(), snapshot.target.data(),
                 snapshot.target.size_bytes(), cudaMemcpyDeviceToHost));
  }
  RETURN_IF_ERROR(executor_.Synchronize());
  for (const auto& snapshot : snapshots_) {
    if (std::memcmp(snapshot.original.data(), snapshot.staging.data(),
                    snapshot.target.size_bytes()) != 0) {
      return absl::DataLossError(
          "restored weight bytes differ from pristine snapshot");
    }
  }
  dirty_ = false;
  return absl::OkStatus();
}

MlpRowIntervention::MlpRowIntervention(
    cuda::Executor& executor, cuda::Buffer target, cuda::Buffer backup,
    cuda::PageLockedHostArray<float> original,
    cuda::PageLockedHostArray<float> staging, std::vector<int> feature_ids,
    size_t output_width)
    : executor_(executor),
      target_(std::move(target)),
      backup_(std::move(backup)),
      original_(std::move(original)),
      staging_(std::move(staging)),
      feature_ids_(std::move(feature_ids)),
      output_width_(output_width) {}

absl::StatusOr<std::unique_ptr<MlpRowIntervention>> MlpRowIntervention::Capture(
    cuda::Executor& executor, const cuda::Buffer& output_weight,
    int input_features, int output_width, absl::Span<const int> feature_ids) {
  if (input_features <= 0 || output_width <= 0 || feature_ids.empty() ||
      size_t(input_features) > std::numeric_limits<size_t>::max() /
                                   sizeof(float) / size_t(output_width) ||
      output_weight.size_bytes() !=
          size_t(input_features) * size_t(output_width) * sizeof(float) ||
      &output_weight.executor() != &executor) {
    return absl::InvalidArgumentError(
        "invalid FP32 MLP output matrix shape, executor, or empty row group");
  }
  absl::flat_hash_set<int> seen;
  for (int feature : feature_ids) {
    if (feature < 0 || feature >= input_features ||
        !seen.insert(feature).second) {
      return absl::InvalidArgumentError(
          "MLP feature IDs must be unique and in range");
    }
  }
  ASSIGN_OR_RETURN(auto backup,
                   cuda::Buffer::Allocate(executor, output_weight.size_bytes()));
  const size_t count = output_weight.size_bytes() / sizeof(float);
  ASSIGN_OR_RETURN(auto original,
                   cuda::PageLockedHostArray<float>::Allocate(executor, count));
  ASSIGN_OR_RETURN(auto staging,
                   cuda::PageLockedHostArray<float>::Allocate(executor, count));
  // Own every asynchronous endpoint before the first copy. Destruction also
  // synchronizes failed captures, so no pinned allocation can expire early.
  auto result = absl::WrapUnique(new MlpRowIntervention(
      executor, output_weight, std::move(backup), std::move(original),
      std::move(staging), {feature_ids.begin(), feature_ids.end()},
      size_t(output_width)));
  RETURN_IF_ERROR(Transfer(executor, result->backup_.data(),
                           output_weight.data(), output_weight.size_bytes(),
                           cudaMemcpyDeviceToDevice));
  RETURN_IF_ERROR(Transfer(executor, result->original_.data(),
                           output_weight.data(), output_weight.size_bytes(),
                           cudaMemcpyDeviceToHost));
  RETURN_IF_ERROR(executor.Synchronize());
  for (float value : result->original_)
    if (!std::isfinite(value))
      return absl::InvalidArgumentError("nonfinite MLP output weight");
  return result;
}

MlpRowIntervention::~MlpRowIntervention() {
  if (dirty_) {
    const auto status = RestoreAndVerify();
    if (!status.ok())
      std::cerr << "Emergency MLP row restore failed: " << status << '\n';
  }
  (void)executor_.Synchronize();
}

absl::Status MlpRowIntervention::Apply(float scale) {
  if (scale != 0.0f && scale != 0.5f && scale != 1.0f)
    return absl::InvalidArgumentError("MLP row scale must be 0, 0.5, or 1");
  if (scale == 1.0f)
    return RestoreAndVerify();
  dirty_ = true;
  const size_t row_bytes = output_width_ * sizeof(float);
  for (int feature : feature_ids_) {
    const size_t offset = size_t(feature) * output_width_;
    auto* target = static_cast<float*>(target_.data()) + offset;
    if (scale == 0.0f) {
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemsetAsync(target, 0, row_bytes, executor_.stream()),
          "zero selected MLP output row"));
    } else {
      for (size_t column = 0; column < output_width_; ++column)
        staging_[offset + column] = original_[offset + column] * scale;
      RETURN_IF_ERROR(Transfer(executor_, target, staging_.data() + offset,
                               row_bytes, cudaMemcpyHostToDevice));
    }
  }
  return executor_.Synchronize();
}

absl::Status MlpRowIntervention::RestoreAndVerify() {
  // Mark dirty before submitting work so a failed explicit restore is retried
  // on destruction. Never write unselected rows, even if verification fails.
  dirty_ = true;
  const size_t row_bytes = output_width_ * sizeof(float);
  for (int feature : feature_ids_) {
    const size_t offset = size_t(feature) * output_width_;
    RETURN_IF_ERROR(Transfer(executor_,
                             static_cast<float*>(target_.data()) + offset,
                             static_cast<const float*>(backup_.data()) + offset,
                             row_bytes, cudaMemcpyDeviceToDevice));
  }
  RETURN_IF_ERROR(Transfer(executor_, staging_.data(), target_.data(),
                           target_.size_bytes(), cudaMemcpyDeviceToHost));
  RETURN_IF_ERROR(executor_.Synchronize());
  if (std::memcmp(original_.data(), staging_.data(), target_.size_bytes()) !=
      0) {
    return absl::DataLossError(
        "MLP output matrix differs from pristine snapshot after row restore");
  }
  dirty_ = false;
  return absl::OkStatus();
}

absl::StatusOr<Measurements> EvaluatePassages(
    cuda::Executor& executor, const llm::Layer& model,
    const llm::Layer& loss_layer, const PackedBatch& batch, int batch_sequences,
    int vocab_size, int padded_vocab_size) {
  if (batch_sequences <= 0 || batch.context_length <= 0 ||
      batch.passage_count <= 0 || vocab_size <= 0 ||
      padded_vocab_size < vocab_size ||
      batch.passage_count >
          std::numeric_limits<int>::max() / batch.context_length ||
      batch.tokens.size() !=
          size_t{2} * batch.passage_count * batch.context_length) {
    return absl::InvalidArgumentError("invalid evaluation batch shape");
  }
  const size_t total = size_t(batch.passage_count) * batch.context_length;
  ASSIGN_OR_RETURN(auto losses,
                   cuda::PageLockedHostArray<float>::Allocate(executor, total));
  ASSIGN_OR_RETURN(auto argmax, cuda::PageLockedHostArray<int32_t>::Allocate(
                                    executor, total));
  // All host storage outlives outstanding work, even on an error return.
  struct SynchronizeOnExit {
    cuda::Executor& executor;
    ~SynchronizeOnExit() { (void)executor.Synchronize(); }
  } guard{executor};
  for (int first = 0; first < batch.passage_count;) {
    int count = std::min(batch_sequences, batch.passage_count - first);
    int rows = count * batch.context_length;
    size_t bytes = size_t(rows) * sizeof(int32_t);
    ASSIGN_OR_RETURN(auto inputs, cuda::Buffer::Allocate(executor, bytes));
    ASSIGN_OR_RETURN(auto targets, cuda::Buffer::Allocate(executor, bytes));
    RETURN_IF_ERROR(Transfer(executor, inputs.data(),
                             batch.inputs(first, count).data(), bytes,
                             cudaMemcpyHostToDevice));
    RETURN_IF_ERROR(Transfer(executor, targets.data(),
                             batch.targets(first, count).data(), bytes,
                             cudaMemcpyHostToDevice));
    llm::Tape model_tape;
    llm::Tape loss_tape;
    ASSIGN_OR_RETURN(auto logits, model.fwd(executor, {inputs}, &model_tape));
    if (logits.size_bytes() != size_t(rows) * padded_vocab_size * sizeof(float))
      return absl::DataLossError("model did not return expected FP32 logits");
    ASSIGN_OR_RETURN(auto loss,
                     loss_layer.fwd(executor, {logits, targets}, &loss_tape));
    if (loss.size_bytes() != bytes)
      return absl::DataLossError("unexpected loss shape");
    ASSIGN_OR_RETURN(auto ids, ArgmaxTokens(executor, logits, rows, vocab_size,
                                            padded_vocab_size));
    const size_t offset = size_t(first) * batch.context_length;
    RETURN_IF_ERROR(Transfer(executor, losses.data() + offset, loss.data(),
                             bytes, cudaMemcpyDeviceToHost));
    RETURN_IF_ERROR(Transfer(executor, argmax.data() + offset, ids.data(),
                             bytes, cudaMemcpyDeviceToHost));
    RETURN_IF_ERROR(executor.Synchronize());
    for (size_t i = offset; i < offset + rows; ++i) {
      if (!std::isfinite(losses[i]) || losses[i] < 0 || argmax[i] < 0 ||
          argmax[i] >= vocab_size) {
        return absl::DataLossError(
            "invalid loss or nonfinite/logically invalid argmax row");
      }
    }
    first += count;
  }
  return Measurements{std::move(losses), std::move(argmax)};
}

absl::Status WriteExclusive(const std::filesystem::path& path,
                            const void* bytes, size_t size) {
  const int fd =
      open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (fd < 0)
    return absl::AlreadyExistsError(
        absl::StrCat("cannot create ", path.string()));
  const char* current = static_cast<const char*>(bytes);
  while (size) {
    ssize_t written = write(fd, current, size);
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0) {
      close(fd);
      return absl::DataLossError("failed writing probe artifact");
    }
    current += written;
    size -= written;
  }
  if (close(fd))
    return absl::DataLossError("failed closing probe artifact");
  return absl::OkStatus();
}

}  // namespace pluto::weight_analysis
