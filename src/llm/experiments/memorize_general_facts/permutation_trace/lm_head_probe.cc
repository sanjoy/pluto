// Isolate vocabulary-order sensitivity in the existing BF16 LM-head backward.
// No training, checkpoint, or replacement GPU kernel is involved. Identical
// embedding rows imply that renaming targets cannot change dLoss/dHidden in
// real arithmetic. Native tensor-core accumulation can nevertheless depend on
// where the exceptional target gradient lies in its vocabulary reduction.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layer.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/llm/layers/embedding.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

constexpr int kVocabulary = 4475;
constexpr int kWidth = 10;
constexpr int kSequence = 27;
constexpr int kTarget = 2154;
constexpr int kRenamedTarget = 2457;

// Host conversion is solely for deterministic inputs and a mathematical
// reference. The actual head uses its unmodified cuTile BF16 MMA kernels.
uint16_t Bf16Bits(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fffU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>(bits >> 16);
}

double RoundedBf16(float value) {
  return std::bit_cast<float>(static_cast<uint32_t>(Bf16Bits(value)) << 16);
}

template <class T>
absl::StatusOr<Buffer> CopyH2D(cuda::Executor& executor,
                               absl::Span<const T> values) {
  ASSIGN_OR_RETURN(auto host,
                   cuda::PageLockedHostArray<T>::CopyFrom(executor, values));
  ASSIGN_OR_RETURN(auto device, Buffer::Allocate(executor, host.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(device.data(), host.data(), host.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "probe copy H2D"));
  RETURN_IF_ERROR(executor.Synchronize());
  return device;
}

template <class T>
absl::StatusOr<cuda::PageLockedHostArray<T>> CopyD2H(cuda::Executor& executor,
                                                     const Buffer& device) {
  if (device.size_bytes() % sizeof(T) != 0)
    return absl::InvalidArgumentError(
        "probe buffer has incompatible element size");
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<T>::Allocate(
                                  executor, device.size_bytes() / sizeof(T)));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host.data(), device.data(), host.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "probe copy D2H"));
  RETURN_IF_ERROR(executor.Synchronize());
  return host;
}

// Captured arrays contain the exact FP32 values produced by native kernels.
struct Capture {
  cuda::PageLockedHostArray<float> logits;
  cuda::PageLockedHostArray<float> logit_gradient;
  cuda::PageLockedHostArray<float> hidden_gradient;
};

absl::StatusOr<Capture> RunNative(cuda::Executor& executor,
                                  LanguageModelingHeadLayer& head,
                                  CrossEntropyLossLayer& loss,
                                  const Buffer& hidden, const Buffer& targets) {
  ASSIGN_OR_RETURN(auto forward, head.fwd(executor, {hidden}));
  ASSIGN_OR_RETURN(auto loss_forward,
                   loss.fwd(executor, {forward.outputs[0], targets}));
  ASSIGN_OR_RETURN(auto logit_gradient,
                   loss.bwd(executor, {}, std::move(loss_forward.state)));
  ASSIGN_OR_RETURN(auto hidden_gradient,
                   head.bwd(executor, logit_gradient, std::move(forward.state)));
  ASSIGN_OR_RETURN(auto logits, CopyD2H<float>(executor, forward.outputs[0]));
  ASSIGN_OR_RETURN(auto dlogits, CopyD2H<float>(executor, logit_gradient[0]));
  ASSIGN_OR_RETURN(auto dhidden, CopyD2H<float>(executor, hidden_gradient[0]));
  return Capture{std::move(logits), std::move(dlogits), std::move(dhidden)};
}

// An exact permutation only reindexes vocabulary columns. Padding columns do
// not correspond to tokens and must retain their original positions.
void UndoRename(cuda::PageLockedHostArray<float>& values, int rows,
                int stride) {
  for (int row = 0; row < rows; ++row)
    std::swap(values[row * stride + kTarget],
              values[row * stride + kRenamedTarget]);
}

struct Difference {
  size_t changed = 0;
  double max_absolute = 0;
  double squared_l2 = 0;
};

Difference Compare(absl::Span<const float> first,
                   absl::Span<const float> second) {
  Difference difference;
  for (size_t i = 0; i < first.size(); ++i) {
    // Masked logits are -infinity. Equal bit patterns contribute zero error,
    // not the NaN that subtracting two equal infinities would produce.
    if (std::bit_cast<uint32_t>(first[i]) == std::bit_cast<uint32_t>(second[i]))
      continue;
    ++difference.changed;
    const double error = static_cast<double>(first[i]) - second[i];
    difference.max_absolute =
        std::max(difference.max_absolute, std::abs(error));
    difference.squared_l2 += error * error;
  }
  return difference;
}

void PrintDifference(const std::string& label, const Difference& difference,
                     size_t size) {
  std::cout << label << '\t' << difference.changed << '/' << size << '\t'
            << difference.max_absolute << '\t'
            << std::sqrt(difference.squared_l2) << '\n';
}

absl::Status RequireSame(const std::string& label,
                         absl::Span<const float> first,
                         absl::Span<const float> second) {
  if (first.size() != second.size())
    return absl::InternalError(label + ": incompatible capture sizes");
  const auto difference = Compare(first, second);
  PrintDifference(label, difference, first.size());
  if (difference.changed)
    return absl::InternalError(label + ": expected bitwise equality");
  return absl::OkStatus();
}

absl::Status RunCase(cuda::Executor& executor, int batch_size,
                     bool one_scored_row) {
  const int rows = batch_size * kSequence;
  ASSIGN_OR_RETURN(auto embedding,
                   EmbeddingLookupLayer::Create(executor, kVocabulary, kWidth,
                                                DataType::BF16, kSequence,
                                                /*pad_vocabulary=*/false));
  RETURN_IF_ERROR(embedding->InitializeNormal(0.02f, 1337, true));
  ASSIGN_OR_RETURN(auto head,
                   LanguageModelingHeadLayer::Create(embedding.get()));
  ASSIGN_OR_RETURN(auto loss,
                   CrossEntropyLossLayer::Create(executor, kVocabulary,
                                                 DataType::BF16, kSequence));
  const int stride = embedding->padded_vocab_size();
  ASSIGN_OR_RETURN(
      auto hidden_values,
      cuda::PageLockedHostArray<uint16_t>::Allocate(executor, rows * kWidth));
  ASSIGN_OR_RETURN(auto targets,
                   cuda::PageLockedHostArray<int32_t>::Allocate(executor, rows));
  for (int row = 0; row < rows; ++row) {
    // Small dyadic values are exactly representable as BF16, and vary by row
    // and channel without requiring a platform-dependent random generator.
    for (int column = 0; column < kWidth; ++column)
      hidden_values[row * kWidth + column] =
          Bf16Bits(((row * 7 + column * 3) % 31 - 15) / 16.0f);
    targets[row] = !one_scored_row || row == 0 ? kTarget : -1;
  }
  ASSIGN_OR_RETURN(auto hidden,
                   CopyH2D<uint16_t>(executor, hidden_values.span()));
  ASSIGN_OR_RETURN(auto baseline_targets,
                   CopyH2D<int32_t>(executor, targets.span()));
  ASSIGN_OR_RETURN(auto baseline,
                   RunNative(executor, *head, *loss, hidden, baseline_targets));
  ASSIGN_OR_RETURN(auto repeat,
                   RunNative(executor, *head, *loss, hidden, baseline_targets));
  for (auto& target : targets)
    if (target == kTarget)
      target = kRenamedTarget;
  ASSIGN_OR_RETURN(auto renamed_targets,
                   CopyH2D<int32_t>(executor, targets.span()));
  ASSIGN_OR_RETURN(auto renamed,
                   RunNative(executor, *head, *loss, hidden, renamed_targets));

  std::cout << "\ncase batch_size=" << batch_size
            << " sequence_length=" << kSequence
            << " scored_rows=" << (one_scored_row ? 1 : rows) << '\n';
  std::cout << "comparison\tchanged/total\tmax_absolute\tl2\n";
  RETURN_IF_ERROR(RequireSame("baseline_repeat_logits", baseline.logits.span(),
                              repeat.logits.span()));
  RETURN_IF_ERROR(RequireSame("baseline_repeat_dlogits",
                              baseline.logit_gradient.span(),
                              repeat.logit_gradient.span()));
  RETURN_IF_ERROR(RequireSame("baseline_repeat_dhidden",
                              baseline.hidden_gradient.span(),
                              repeat.hidden_gradient.span()));
  RETURN_IF_ERROR(RequireSame("renamed_logits", baseline.logits.span(),
                              renamed.logits.span()));
  PrintDifference(
      "renamed_dhidden",
      Compare(baseline.hidden_gradient.span(), renamed.hidden_gradient.span()),
      baseline.hidden_gradient.size());

  // Reindex only dLogits into the original order. Embeddings need no copy:
  // all their rows are identical. This preserves the trained kernel and its
  // arithmetic while undoing the placement of its negative target gradient.
  UndoRename(renamed.logit_gradient, rows, stride);
  RETURN_IF_ERROR(RequireSame("inverse_permuted_dlogits",
                              baseline.logit_gradient.span(),
                              renamed.logit_gradient.span()));
  ASSIGN_OR_RETURN(auto canonical_dlogits,
                   CopyH2D<float>(executor, renamed.logit_gradient.span()));
  ASSIGN_OR_RETURN(auto replay_forward, head->fwd(executor, {hidden}));
  ASSIGN_OR_RETURN(
      auto replay_gradient,
      head->bwd(executor, {canonical_dlogits}, std::move(replay_forward.state)));
  ASSIGN_OR_RETURN(auto replay, CopyD2H<float>(executor, replay_gradient[0]));
  RETURN_IF_ERROR(RequireSame("canonical_order_replay_dhidden",
                              baseline.hidden_gradient.span(), replay.span()));

  ASSIGN_OR_RETURN(auto weights, CopyD2H<float>(executor, embedding->weight()));
  // FP64 sums of the *BF16-rounded* MMA operands isolate accumulation error
  // from operand quantization. Raw FP32 dLogits sum is printed separately;
  // BF16 rounding means that the sum need not remain exactly zero.
  double fp32_sum = 0;
  double bf16_sum = 0;
  for (int token = 0; token < kVocabulary; ++token) {
    fp32_sum += baseline.logit_gradient[token];
    bf16_sum += RoundedBf16(baseline.logit_gradient[token]);
  }
  std::cout << "row0_dlogits_sum_fp64=" << fp32_sum
            << " row0_bf16_dlogits_sum_fp64=" << bf16_sum << '\n';
  std::cout << "row0_channel\tbaseline_dhidden\trenamed_dhidden\t"
               "fp64_sum_of_bf16_products\tbaseline_error\trenamed_error\n";
  for (int column = 0; column < kWidth; ++column) {
    const double reference = bf16_sum * RoundedBf16(weights[column]);
    std::cout << column << '\t' << baseline.hidden_gradient[column] << '\t'
              << renamed.hidden_gradient[column] << '\t' << reference << '\t'
              << baseline.hidden_gradient[column] - reference << '\t'
              << renamed.hidden_gradient[column] - reference << '\n';
  }
  return absl::OkStatus();
}

absl::Status Run() {
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  std::cout
      << std::setprecision(17)
      << "LM head vocabulary permutation diagnostic; native BF16 kernels\n"
      << "vocabulary=" << kVocabulary << " width=" << kWidth
      << " target=" << kTarget << " renamed_target=" << kRenamedTarget
      << " identical_embedding_seed=1337\n";
  RETURN_IF_ERROR(RunCase(*executor, 1, true));
  RETURN_IF_ERROR(RunCase(*executor, 1, false));
  RETURN_IF_ERROR(RunCase(*executor, 32, false));
  return executor->Synchronize();
}

}  // namespace
}  // namespace pluto::llm

int main() {
  const auto status = pluto::llm::Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
