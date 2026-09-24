#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/adamw_optimizer.h"
#include "src/llm/gpt2.h"
#include "src/llm/layer.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

// Capture FP32 bit patterns, not approximately equal numerical values: even
// the smallest rounding difference can grow after many optimizer updates.
// Deduplicating shared allocations also checks the tied embedding only once.
struct Snapshot {
  std::vector<size_t> offsets{0};
  std::vector<uint32_t> bits;
};

absl::StatusOr<Snapshot> CopyD2H(cuda::Executor& executor,
                                 absl::Span<const Buffer> buffers) {
  Snapshot snapshot;
  std::vector<const Buffer*> unique;
  for (const Buffer& buffer : buffers) {
    if (std::any_of(unique.begin(), unique.end(), [&](const Buffer* previous) {
          return previous->data() == buffer.data();
        }))
      continue;
    if (buffer.size_bytes() % sizeof(uint32_t) != 0)
      return absl::InternalError("training snapshot expected FP32 buffers");
    unique.push_back(&buffer);
    snapshot.offsets.push_back(snapshot.offsets.back() +
                               buffer.size_bytes() / sizeof(uint32_t));
  }
  ASSIGN_OR_RETURN(auto staging, cuda::PageLockedHostArray<uint32_t>::Allocate(
                                     executor, snapshot.offsets.back()));
  for (size_t tensor = 0; tensor < unique.size(); ++tensor) {
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(staging.data() + snapshot.offsets[tensor],
                        unique[tensor]->data(), unique[tensor]->size_bytes(),
                        cudaMemcpyDeviceToHost, executor.stream()),
        "copy vocabulary-order test snapshot"));
  }
  RETURN_IF_ERROR(executor.Synchronize());
  snapshot.bits.assign(staging.begin(), staging.end());
  for (uint32_t bits : snapshot.bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    if (!std::isfinite(value))
      return absl::InternalError("training snapshot contains nonfinite data");
  }
  return snapshot;
}

template <typename T>
absl::StatusOr<Buffer> CopyH2D(cuda::Executor& executor,
                               absl::Span<const T> values) {
  ASSIGN_OR_RETURN(auto staging,
                   cuda::PageLockedHostArray<T>::CopyFrom(executor, values));
  ASSIGN_OR_RETURN(auto buffer,
                   Buffer::Allocate(executor, staging.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(buffer.data(), staging.data(), staging.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload vocabulary-order test input"));
  return buffer;
}

enum class VocabularyAxis { kNone, kColumns, kFirstTensorRows };

// Physical ID order changes, while the logical tensor does not. Only logits
// and dLogits permute columns, and only the first parameter tensor (the token
// embedding) permutes rows. Padding and every transformer parameter stay put.
testing::AssertionResult EqualAligned(const Snapshot& baseline,
                                      const Snapshot& renamed,
                                      absl::Span<const int32_t> order,
                                      VocabularyAxis axis, int width,
                                      int logit_stride) {
  if (baseline.offsets != renamed.offsets ||
      baseline.bits.size() != renamed.bits.size())
    return testing::AssertionFailure() << "snapshot shapes differ";
  for (size_t tensor = 0; tensor + 1 < baseline.offsets.size(); ++tensor) {
    const size_t begin = baseline.offsets[tensor];
    const size_t elements = baseline.offsets[tensor + 1] - begin;
    for (size_t index = 0; index < elements; ++index) {
      size_t mapped = index;
      if (axis == VocabularyAxis::kColumns) {
        const size_t token = index % logit_stride;
        if (token < order.size())
          mapped = index - token + order[token];
      } else if (axis == VocabularyAxis::kFirstTensorRows && tensor == 0) {
        const size_t token = index / width;
        if (token < order.size())
          mapped = static_cast<size_t>(order[token]) * width + index % width;
      }
      if (baseline.bits[begin + index] != renamed.bits[begin + mapped])
        return testing::AssertionFailure()
               << "tensor " << tensor << ", canonical element " << index
               << ", physical element " << mapped << ": FP32 bits "
               << baseline.bits[begin + index]
               << " != " << renamed.bits[begin + mapped];
    }
  }
  return testing::AssertionSuccess();
}

class TokenOrderTrainingTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }

  void TearDown() override {
    if (executor_ == nullptr)
      return;
    EXPECT_TRUE(executor_->Synchronize().ok());
  }

  void Run(DataType type, int vocabulary_size, bool pad_vocabulary, int steps) {
    SCOPED_TRACE(testing::Message() << "compute=" << static_cast<int>(type)
                                    << ", vocabulary=" << vocabulary_size
                                    << ", padded_table=" << pad_vocabulary);
    constexpr int kWidth = 10;
    constexpr int kContext = 27;
    constexpr int kBatchSize = 2;
    constexpr int kRows = kBatchSize * kContext;
    const int logit_stride = (vocabulary_size + 15) / 16 * 16;
    Gpt2Config config;
    config.transformer_block_count = 4;
    config.model_width = kWidth;
    config.attention_heads = 1;
    config.feed_forward_width = 20;
    config.vocabulary_size = vocabulary_size;
    config.pad_vocabulary = pad_vocabulary;
    config.context_length = kContext;

    // An affine permutation crosses 16/64-column reduction tiles and is not
    // its own inverse. This catches accidentally reversing the API mapping.
    // Both tested vocabulary sizes are coprime to 37.
    std::vector<int32_t> order(vocabulary_size);
    for (int token = 0; token < vocabulary_size; ++token)
      order[token] = (37 * token + 17) % vocabulary_size;

    auto baseline = CreateGpt2(*executor_, type, 1337, config);
    auto renamed = CreateGpt2(*executor_, type, 1337, config, order);
    ASSERT_TRUE(baseline.ok()) << baseline.status();
    ASSERT_TRUE(renamed.ok()) << renamed.status();
    auto baseline_loss = CrossEntropyLossLayer::Create(
        *executor_, vocabulary_size, type, kContext);
    auto renamed_loss = CrossEntropyLossLayer::Create(
        *executor_, vocabulary_size, type, kContext, order);
    ASSERT_TRUE(baseline_loss.ok()) << baseline_loss.status();
    ASSERT_TRUE(renamed_loss.ok()) << renamed_loss.status();

    // A change of labels must also relabel the initial embedding rows. Merely
    // using the same initialization seed would give different logical models.
    // Distinct random rows make this stronger than identical-row
    // initialization.
    auto initial = CopyD2H(*executor_, (*baseline)->weights());
    ASSERT_TRUE(initial.ok()) << initial.status();
    const size_t embedding_elements = initial->offsets[1];
    ASSERT_EQ(
        embedding_elements,
        static_cast<size_t>(pad_vocabulary ? logit_stride : vocabulary_size) *
            kWidth);
    auto permuted_embedding = cuda::PageLockedHostArray<uint32_t>::Allocate(
        *executor_, embedding_elements);
    ASSERT_TRUE(permuted_embedding.ok()) << permuted_embedding.status();
    std::copy_n(initial->bits.begin(), embedding_elements,
                permuted_embedding->begin());
    for (int token = 0; token < vocabulary_size; ++token)
      for (int column = 0; column < kWidth; ++column)
        (*permuted_embedding)[order[token] * kWidth + column] =
            initial->bits[token * kWidth + column];
    ASSERT_EQ(cudaMemcpyAsync((*renamed)->weights()[0].data(),
                              permuted_embedding->data(),
                              permuted_embedding->size_bytes(),
                              cudaMemcpyHostToDevice, executor_->stream()),
              cudaSuccess);
    auto renamed_initial = CopyD2H(*executor_, (*renamed)->weights());
    ASSERT_TRUE(renamed_initial.ok()) << renamed_initial.status();
    ASSERT_TRUE(EqualAligned(*initial, *renamed_initial, order,
                             VocabularyAxis::kFirstTensorRows, kWidth,
                             logit_stride));
    // Four complete blocks expose 52 unique parameter tensors, with the
    // embedding/head alias deduplicated rather than updated or counted twice.
    ASSERT_EQ(initial->offsets.size(), size_t{53});

    const AdamWConfig optimizer_config{.learning_rate = 1.2e-3f,
                                       .beta1 = 0.9f,
                                       .beta2 = 0.99f,
                                       .epsilon = 1e-8f,
                                       .weight_decay = 0.0f};
    auto baseline_optimizer =
        AdamWOptimizer::Create(*executor_, **baseline, optimizer_config);
    auto renamed_optimizer =
        AdamWOptimizer::Create(*executor_, **renamed, optimizer_config);
    ASSERT_TRUE(baseline_optimizer.ok()) << baseline_optimizer.status();
    ASSERT_TRUE(renamed_optimizer.ok()) << renamed_optimizer.status();
    ASSERT_TRUE((*baseline_optimizer)->ZeroGrad().ok());
    ASSERT_TRUE((*renamed_optimizer)->ZeroGrad().ok());

    for (int step = 0; step < steps; ++step) {
      SCOPED_TRACE(testing::Message() << "step=" << step);
      std::vector<int32_t> tokens(kRows), targets(kRows);
      std::vector<int32_t> renamed_tokens(kRows), renamed_targets(kRows);
      for (int row = 0; row < kRows; ++row) {
        // Repeated IDs exercise the input-embedding gradient gather. Other
        // rows spread across the entire vocabulary and its final partial tile.
        tokens[row] = row % 5 == 0
                          ? vocabulary_size - 1
                          : (row * 97 + step * 43 + row / 7) % vocabulary_size;
        renamed_tokens[row] = order[tokens[row]];
      }
      for (int row = 0; row < kRows; ++row) {
        targets[row] =
            tokens[row / kContext * kContext + (row % kContext + 1) % kContext];
        // Include fully supervised batches, prompt/right-padding masks, a
        // wholly ignored sample, and a wholly ignored batch. Ignored IDs are
        // sentinels, not vocabulary entries, and must never be permuted.
        if ((step % 4 == 1 &&
             (row % kContext < 4 || row % kContext >= kContext - 3)) ||
            (step % 4 == 2 && row < kContext) || step % 4 == 3)
          targets[row] = CrossEntropyLossLayer::kIgnoredTarget;
        renamed_targets[row] =
            targets[row] < 0 ? targets[row] : order[targets[row]];
      }
      auto input = CopyH2D<int32_t>(*executor_, tokens);
      auto renamed_input = CopyH2D<int32_t>(*executor_, renamed_tokens);
      auto target = CopyH2D<int32_t>(*executor_, targets);
      auto renamed_target = CopyH2D<int32_t>(*executor_, renamed_targets);
      ASSERT_TRUE(input.ok()) << input.status();
      ASSERT_TRUE(renamed_input.ok()) << renamed_input.status();
      ASSERT_TRUE(target.ok()) << target.status();
      ASSERT_TRUE(renamed_target.ok()) << renamed_target.status();

      auto forward = (*baseline)->fwd(*executor_, {*input});
      auto renamed_forward = (*renamed)->fwd(*executor_, {*renamed_input});
      ASSERT_TRUE(forward.ok()) << forward.status();
      ASSERT_TRUE(renamed_forward.ok()) << renamed_forward.status();
      auto logits = CopyD2H(*executor_, forward->outputs);
      auto renamed_logits = CopyD2H(*executor_, renamed_forward->outputs);
      ASSERT_TRUE(logits.ok()) << logits.status();
      ASSERT_TRUE(renamed_logits.ok()) << renamed_logits.status();
      ASSERT_TRUE(EqualAligned(*logits, *renamed_logits, order,
                               VocabularyAxis::kColumns, kWidth, logit_stride))
          << "forward logits";

      auto loss =
          (*baseline_loss)->fwd(*executor_, {forward->outputs[0], *target});
      auto loss_renamed =
          (*renamed_loss)
              ->fwd(*executor_, {renamed_forward->outputs[0], *renamed_target});
      ASSERT_TRUE(loss.ok()) << loss.status();
      ASSERT_TRUE(loss_renamed.ok()) << loss_renamed.status();
      auto loss_values = CopyD2H(*executor_, loss->outputs);
      auto renamed_loss_values = CopyD2H(*executor_, loss_renamed->outputs);
      ASSERT_TRUE(loss_values.ok()) << loss_values.status();
      ASSERT_TRUE(renamed_loss_values.ok()) << renamed_loss_values.status();
      ASSERT_TRUE(EqualAligned(*loss_values, *renamed_loss_values, order,
                               VocabularyAxis::kNone, kWidth, logit_stride))
          << "cross-entropy loss";

      auto d_logits =
          (*baseline_loss)->bwd(*executor_, {}, std::move(loss->state));
      auto renamed_d_logits =
          (*renamed_loss)->bwd(*executor_, {}, std::move(loss_renamed->state));
      ASSERT_TRUE(d_logits.ok()) << d_logits.status();
      ASSERT_TRUE(renamed_d_logits.ok()) << renamed_d_logits.status();
      auto gradient_values = CopyD2H(*executor_, *d_logits);
      auto renamed_gradient_values = CopyD2H(*executor_, *renamed_d_logits);
      ASSERT_TRUE(gradient_values.ok()) << gradient_values.status();
      ASSERT_TRUE(renamed_gradient_values.ok())
          << renamed_gradient_values.status();
      ASSERT_TRUE(EqualAligned(*gradient_values, *renamed_gradient_values,
                               order, VocabularyAxis::kColumns, kWidth,
                               logit_stride))
          << "cross-entropy logit gradients";

      auto backward =
          (*baseline)->bwd(*executor_, *d_logits, std::move(forward->state));
      auto renamed_backward = (*renamed)->bwd(
          *executor_, *renamed_d_logits, std::move(renamed_forward->state));
      ASSERT_TRUE(backward.ok()) << backward.status();
      ASSERT_TRUE(renamed_backward.ok()) << renamed_backward.status();
      auto gradients = CopyD2H(*executor_, (*baseline)->gradients());
      auto renamed_gradients = CopyD2H(*executor_, (*renamed)->gradients());
      ASSERT_TRUE(gradients.ok()) << gradients.status();
      ASSERT_TRUE(renamed_gradients.ok()) << renamed_gradients.status();
      ASSERT_TRUE(EqualAligned(*gradients, *renamed_gradients, order,
                               VocabularyAxis::kFirstTensorRows, kWidth,
                               logit_stride))
          << "all unique parameter gradients, before AdamW clears them";

      // Match the original experiment's warmup and optimizer sensitivity.
      const float learning_rate = 0.0012 * (step + 1) / 100;
      ASSERT_TRUE((*baseline_optimizer)->SetLearningRate(learning_rate).ok());
      ASSERT_TRUE((*renamed_optimizer)->SetLearningRate(learning_rate).ok());
      ASSERT_TRUE((*baseline_optimizer)->ApplyStep().ok());
      ASSERT_TRUE((*renamed_optimizer)->ApplyStep().ok());
      auto weights = CopyD2H(*executor_, (*baseline)->weights());
      auto renamed_weights = CopyD2H(*executor_, (*renamed)->weights());
      ASSERT_TRUE(weights.ok()) << weights.status();
      ASSERT_TRUE(renamed_weights.ok()) << renamed_weights.status();
      ASSERT_TRUE(EqualAligned(*weights, *renamed_weights, order,
                               VocabularyAxis::kFirstTensorRows, kWidth,
                               logit_stride))
          << "all unique parameters after AdamW";
    }
  }

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(TokenOrderTrainingTest, Bf16TrainingCommutesWithTokenRenaming) {
  for (bool pad_vocabulary : {false, true})
    Run(DataType::BF16, 131, pad_vocabulary, 8);
}

TEST_F(TokenOrderTrainingTest,
       FloatActivationTrainingCommutesWithTokenRenaming) {
  // FP16 is the legacy compute policy with FP32 activation storage and FP16
  // matrix operands. FP32 and FP8 are not supported compute policies.
  for (bool pad_vocabulary : {false, true})
    Run(DataType::FP16, 131, pad_vocabulary, 8);
}

TEST_F(TokenOrderTrainingTest, ActualVocabularyShapeRemainsExactFor32Updates) {
  Run(DataType::BF16, 4475, false, 32);
}

}  // namespace
}  // namespace pluto::llm
