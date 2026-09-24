// Isolate the earliest differing operation using an actual captured training
// batch. Reorder renamed logits/targets back to the original vocabulary BEFORE
// invoking the unchanged native cross-entropy layer, and check bitwise replay.
#include <cuda_runtime.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/experiments/memorize_general_facts/permutation_trace/trace_util.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/llm/layers/embedding.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, trace_dir, "",
          "Directory of a permutation_trace capture");

namespace pluto::llm::permutation_trace {
namespace {

constexpr int kVocabulary = 4475;
constexpr int kPadded = 4480;
constexpr int kRows = 32 * 27;

absl::StatusOr<std::string> ReadFile(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return absl::NotFoundError(absl::StrCat("cannot open ", path.string()));
  std::string result((std::istreambuf_iterator<char>(input)), {});
  if (input.bad())
    return absl::DataLossError("reading trace file failed");
  return result;
}

struct SavedTensor {
  TensorDescription description;
  std::vector<uint8_t> bytes;
};

absl::StatusOr<SavedTensor> ReadStage(const std::filesystem::path& directory,
                                      absl::string_view stage) {
  ASSIGN_OR_RETURN(auto manifest, ReadFile(directory / "tensors.tsv"));
  for (auto line : absl::StrSplit(manifest, '\n', absl::SkipEmpty())) {
    const std::vector<absl::string_view> fields = absl::StrSplit(line, '\t');
    if (fields.size() != 7 || fields[1] != stage)
      continue;
    TensorDescription description{
        std::string(stage), std::string(fields[2]), {}, std::string(fields[4])};
    for (auto dimension : absl::StrSplit(fields[3], ',')) {
      int64_t extent;
      if (!absl::SimpleAtoi(dimension, &extent))
        return absl::DataLossError("invalid shape in trace manifest");
      description.shape.push_back(extent);
    }
    const std::filesystem::path file{std::string(fields[5])};
    if (file.is_absolute() || file.has_parent_path())
      return absl::InvalidArgumentError("trace files must be local filenames");
    ASSIGN_OR_RETURN(auto bytes, ReadFile(directory / file));
    uint64_t expected_bytes;
    if (!absl::SimpleAtoi(fields[6], &expected_bytes) ||
        bytes.size() != expected_bytes)
      return absl::DataLossError("trace byte count mismatch");
    return SavedTensor{std::move(description), {bytes.begin(), bytes.end()}};
  }
  return absl::NotFoundError(absl::StrCat("missing stage ", stage));
}

absl::StatusOr<Buffer> Upload(cuda::Executor& executor,
                              absl::Span<const uint8_t> bytes) {
  ASSIGN_OR_RETURN(
      auto host, cuda::PageLockedHostArray<uint8_t>::CopyFrom(executor, bytes));
  ASSIGN_OR_RETURN(auto device, Buffer::Allocate(executor, bytes.size()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(device.data(), host.data(), bytes.size(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload CE replay"));
  return device;
}

absl::StatusOr<std::vector<uint8_t>> Download(cuda::Executor& executor,
                                              const Buffer& buffer) {
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<uint8_t>::Allocate(
                                  executor, buffer.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host.data(), buffer.data(), buffer.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "download CE replay"));
  RETURN_IF_ERROR(executor.Synchronize());
  return std::vector<uint8_t>(host.begin(), host.end());
}

struct ReplayOutput {
  std::vector<uint8_t> loss;
  std::vector<uint8_t> gradient;
};

// Native LM-head backward from the actual saved pre-update table and hidden
// activation. No reference GEMM or replacement reduction is introduced here.
absl::StatusOr<std::vector<uint8_t>> ReplayHead(cuda::Executor& executor,
                                                const SavedTensor& embedding,
                                                const SavedTensor& hidden,
                                                const SavedTensor& dlogits) {
  if (embedding.description.dtype != "fp32" ||
      embedding.description.shape != std::vector<int64_t>{kVocabulary, 10} ||
      hidden.description.dtype != "bf16" ||
      hidden.description.shape != std::vector<int64_t>{32, 27, 10} ||
      dlogits.description.dtype != "fp32" ||
      dlogits.description.shape != std::vector<int64_t>{32, 27, kPadded})
    return absl::InvalidArgumentError(
        "unexpected LM-head replay tensor shape/type");
  for (const auto* tensor : {&embedding, &hidden, &dlogits}) {
    ASSIGN_OR_RETURN(auto checked, Compare(tensor->description, tensor->bytes,
                                           tensor->bytes, {}));
    (void)checked;
  }
  ASSIGN_OR_RETURN(auto table,
                   EmbeddingLookupLayer::Create(executor, kVocabulary, 10,
                                                DataType::BF16, 27, false));
  ASSIGN_OR_RETURN(auto device_embedding, Upload(executor, embedding.bytes));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(table->weights()[0].data(), device_embedding.data(),
                      embedding.bytes.size(), cudaMemcpyDeviceToDevice,
                      executor.stream()),
      "set LM-head replay weights"));
  ASSIGN_OR_RETURN(auto head, LanguageModelingHeadLayer::Create(table.get()));
  ASSIGN_OR_RETURN(auto device_hidden, Upload(executor, hidden.bytes));
  ASSIGN_OR_RETURN(auto device_dlogits, Upload(executor, dlogits.bytes));
  ASSIGN_OR_RETURN(auto forward, head->fwd(executor, {device_hidden}));
  ASSIGN_OR_RETURN(auto gradient, head->bwd(executor, {device_dlogits},
                                            std::move(forward.state)));
  return Download(executor, gradient[0]);
}

absl::StatusOr<ReplayOutput> Replay(cuda::Executor& executor,
                                    const SavedTensor& logits,
                                    const SavedTensor& targets) {
  if (logits.description.dtype != "fp32" ||
      targets.description.dtype != "int32" ||
      logits.description.shape != std::vector<int64_t>{32, 27, kPadded} ||
      targets.description.shape != std::vector<int64_t>{32, 27})
    return absl::InvalidArgumentError(
        "unexpected captured logits/target types or shapes");
  // Validate the byte-size/product before uploading any untrusted capture.
  ASSIGN_OR_RETURN(auto checked_logits,
                   Compare(logits.description, logits.bytes, logits.bytes, {}));
  ASSIGN_OR_RETURN(
      auto checked_targets,
      Compare(targets.description, targets.bytes, targets.bytes, {}));
  (void)checked_logits;
  (void)checked_targets;
  ASSIGN_OR_RETURN(auto device_logits, Upload(executor, logits.bytes));
  ASSIGN_OR_RETURN(auto device_targets, Upload(executor, targets.bytes));
  ASSIGN_OR_RETURN(auto loss, CrossEntropyLossLayer::Create(
                                  executor, kVocabulary, DataType::BF16, 27));
  ASSIGN_OR_RETURN(auto forward,
                   loss->fwd(executor, {device_logits, device_targets}));
  ASSIGN_OR_RETURN(auto gradients,
                   loss->bwd(executor, {}, std::move(forward.state)));
  ASSIGN_OR_RETURN(auto loss_bytes, Download(executor, forward.outputs[0]));
  ASSIGN_OR_RETURN(auto gradient_bytes, Download(executor, gradients[0]));
  return ReplayOutput{std::move(loss_bytes), std::move(gradient_bytes)};
}

absl::Status PrintComparison(absl::string_view name,
                             const TensorDescription& description,
                             absl::Span<const uint8_t> baseline,
                             absl::Span<const uint8_t> candidate,
                             absl::Span<const int> permutation,
                             bool must_match) {
  ASSIGN_OR_RETURN(auto difference,
                   Compare(description, baseline, candidate, permutation));
  std::cout << name << '\t' << difference.mismatches << '\t'
            << difference.max_abs << '\t' << difference.l2 << '\n';
  if (must_match && difference.mismatches != 0)
    return absl::DataLossError(
        absl::StrCat("required bitwise replay failed: ", name));
  return absl::OkStatus();
}

absl::Status Run() {
  const std::filesystem::path root(absl::GetFlag(FLAGS_trace_dir));
  if (root.empty())
    return absl::InvalidArgumentError("--trace_dir is required");
  ASSIGN_OR_RETURN(auto text, ReadFile(root / "permutation.tsv"));
  ASSIGN_OR_RETURN(auto permutation,
                   ParsePermutation(text, kVocabulary, kVocabulary - 1));
  ASSIGN_OR_RETURN(
      auto baseline_logits,
      ReadStage(root / "baseline", "fwd/gpt2/LanguageModelingHeadLayer/0"));
  ASSIGN_OR_RETURN(
      auto renamed_logits,
      ReadStage(root / "renamed", "fwd/gpt2/LanguageModelingHeadLayer/0"));
  ASSIGN_OR_RETURN(auto baseline_targets,
                   ReadStage(root / "baseline", "targets"));
  ASSIGN_OR_RETURN(auto renamed_targets, ReadStage(root / "renamed", "targets"));
  ASSIGN_OR_RETURN(auto baseline_loss, ReadStage(root / "baseline", "loss/fwd"));
  ASSIGN_OR_RETURN(auto renamed_loss, ReadStage(root / "renamed", "loss/fwd"));
  ASSIGN_OR_RETURN(auto baseline_gradient,
                   ReadStage(root / "baseline", "loss/dlogits"));
  ASSIGN_OR_RETURN(auto renamed_gradient,
                   ReadStage(root / "renamed", "loss/dlogits"));
  std::cout << std::setprecision(17) << "comparison\tmismatches\tmax_abs\tl2\n";
  RETURN_IF_ERROR(PrintComparison(
      "aligned_input_logits", baseline_logits.description,
      baseline_logits.bytes, renamed_logits.bytes, permutation, true));
  RETURN_IF_ERROR(PrintComparison(
      "aligned_input_targets", baseline_targets.description,
      baseline_targets.bytes, renamed_targets.bytes, permutation, true));
  // Check the fixed replay dimensions before indexing the capture's byte
  // arrays below. Matching byte counts alone do not establish this layout.
  for (const auto* tensor : {&baseline_logits, &renamed_logits})
    if (tensor->description.dtype != "fp32" ||
        tensor->description.shape != std::vector<int64_t>{32, 27, kPadded})
      return absl::InvalidArgumentError("unexpected logits replay layout");
  for (const auto* tensor : {&baseline_targets, &renamed_targets})
    if (tensor->description.dtype != "int32" ||
        tensor->description.shape != std::vector<int64_t>{32, 27})
      return absl::InvalidArgumentError("unexpected targets replay layout");
  // Canonicalization is an offline byte permutation, not a new GPU kernel.
  // Preserve every padding lane and masked target. The resulting CE inputs
  // must exactly equal the baseline, isolating CE's vocabulary reduction order.
  SavedTensor canonical_logits = renamed_logits;
  SavedTensor canonical_targets = renamed_targets;
  std::vector<int> inverse(kVocabulary);
  for (int old = 0; old < kVocabulary; ++old)
    inverse[permutation[old]] = old;
  for (int row = 0; row < kRows; ++row) {
    for (int old = 0; old < kVocabulary; ++old)
      std::memcpy(
          canonical_logits.bytes.data() + (row * kPadded + old) * sizeof(float),
          renamed_logits.bytes.data() +
              (row * kPadded + permutation[old]) * sizeof(float),
          sizeof(float));
    int target;
    std::memcpy(&target, renamed_targets.bytes.data() + row * sizeof(int32_t),
                sizeof(int32_t));
    if (target != -1) {
      if (target < 0 || target >= kVocabulary)
        return absl::DataLossError("invalid replay target");
      target = inverse[target];
    }
    std::memcpy(canonical_targets.bytes.data() + row * sizeof(int32_t), &target,
                sizeof(int32_t));
  }
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto baseline,
                   Replay(*executor, baseline_logits, baseline_targets));
  ASSIGN_OR_RETURN(auto repeat,
                   Replay(*executor, baseline_logits, baseline_targets));
  ASSIGN_OR_RETURN(auto renamed,
                   Replay(*executor, renamed_logits, renamed_targets));
  ASSIGN_OR_RETURN(auto canonical,
                   Replay(*executor, canonical_logits, canonical_targets));
  for (bool gradient : {false, true}) {
    const auto& d =
        gradient ? baseline_gradient.description : baseline_loss.description;
    const auto& base_bytes = gradient ? baseline.gradient : baseline.loss;
    const auto& repeat_bytes = gradient ? repeat.gradient : repeat.loss;
    const auto& rename_bytes = gradient ? renamed.gradient : renamed.loss;
    const auto& canonical_bytes =
        gradient ? canonical.gradient : canonical.loss;
    const auto& saved_base =
        gradient ? baseline_gradient.bytes : baseline_loss.bytes;
    const auto& saved_rename =
        gradient ? renamed_gradient.bytes : renamed_loss.bytes;
    const std::string suffix = gradient ? "/dlogits" : "/loss";
    RETURN_IF_ERROR(PrintComparison("baseline_vs_saved" + suffix, d, saved_base,
                                    base_bytes, {}, true));
    RETURN_IF_ERROR(PrintComparison("renamed_vs_saved" + suffix, d,
                                    saved_rename, rename_bytes, {}, true));
    RETURN_IF_ERROR(PrintComparison("repeat" + suffix, d, base_bytes,
                                    repeat_bytes, {}, true));
    RETURN_IF_ERROR(PrintComparison("renamed_aligned" + suffix, d, base_bytes,
                                    rename_bytes, permutation, false));
    RETURN_IF_ERROR(PrintComparison("canonical_input_order" + suffix, d,
                                    base_bytes, canonical_bytes, {}, true));
  }
  ASSIGN_OR_RETURN(
      auto base_embedding,
      ReadStage(root / "baseline", "weights_before/token_embedding"));
  ASSIGN_OR_RETURN(
      auto renamed_embedding,
      ReadStage(root / "renamed", "weights_before/token_embedding"));
  ASSIGN_OR_RETURN(auto base_hidden,
                   ReadStage(root / "baseline", "fwd/gpt2/LayerNormLayer/0"));
  ASSIGN_OR_RETURN(auto renamed_hidden,
                   ReadStage(root / "renamed", "fwd/gpt2/LayerNormLayer/0"));
  ASSIGN_OR_RETURN(auto base_dhidden,
                   ReadStage(root / "baseline", "bwd/gpt2/LayerNormLayer/0"));
  ASSIGN_OR_RETURN(auto renamed_dhidden,
                   ReadStage(root / "renamed", "bwd/gpt2/LayerNormLayer/0"));
  RETURN_IF_ERROR(PrintComparison(
      "aligned_head_embedding", base_embedding.description,
      base_embedding.bytes, renamed_embedding.bytes, permutation, true));
  RETURN_IF_ERROR(PrintComparison("head_hidden", base_hidden.description,
                                  base_hidden.bytes, renamed_hidden.bytes, {},
                                  true));
  if (base_embedding.description.shape !=
          std::vector<int64_t>{kVocabulary, 10} ||
      renamed_embedding.description.shape !=
          std::vector<int64_t>{kVocabulary, 10} ||
      base_embedding.description.dtype != "fp32" ||
      renamed_embedding.description.dtype != "fp32")
    return absl::InvalidArgumentError("unexpected embedding replay layout");
  SavedTensor canonical_embedding = renamed_embedding;
  SavedTensor canonical_gradient = renamed_gradient;
  for (int old = 0; old < kVocabulary; ++old)
    std::memcpy(
        canonical_embedding.bytes.data() + old * 10 * sizeof(float),
        renamed_embedding.bytes.data() + permutation[old] * 10 * sizeof(float),
        10 * sizeof(float));
  for (int row = 0; row < kRows; ++row)
    for (int old = 0; old < kVocabulary; ++old)
      std::memcpy(canonical_gradient.bytes.data() +
                      (row * kPadded + old) * sizeof(float),
                  renamed_gradient.bytes.data() +
                      (row * kPadded + permutation[old]) * sizeof(float),
                  sizeof(float));
  ASSIGN_OR_RETURN(
      auto native_base_dhidden,
      ReplayHead(*executor, base_embedding, base_hidden, baseline_gradient));
  ASSIGN_OR_RETURN(
      auto native_repeat_dhidden,
      ReplayHead(*executor, base_embedding, base_hidden, baseline_gradient));
  ASSIGN_OR_RETURN(auto native_renamed_dhidden,
                   ReplayHead(*executor, renamed_embedding, renamed_hidden,
                              renamed_gradient));
  ASSIGN_OR_RETURN(auto native_canonical_dhidden,
                   ReplayHead(*executor, canonical_embedding, renamed_hidden,
                              canonical_gradient));
  const auto& d = base_dhidden.description;
  RETURN_IF_ERROR(PrintComparison("head_baseline_vs_saved", d,
                                  base_dhidden.bytes, native_base_dhidden, {},
                                  true));
  RETURN_IF_ERROR(PrintComparison("head_renamed_vs_saved", d,
                                  renamed_dhidden.bytes, native_renamed_dhidden,
                                  {}, true));
  RETURN_IF_ERROR(PrintComparison("head_repeat", d, native_base_dhidden,
                                  native_repeat_dhidden, {}, true));
  RETURN_IF_ERROR(PrintComparison("head_renamed", d, native_base_dhidden,
                                  native_renamed_dhidden, {}, false));
  RETURN_IF_ERROR(PrintComparison("head_canonical_input_order", d,
                                  native_base_dhidden, native_canonical_dhidden,
                                  {}, true));
  return absl::OkStatus();
}

}  // namespace
}  // namespace pluto::llm::permutation_trace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  auto status = pluto::llm::permutation_trace::Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
