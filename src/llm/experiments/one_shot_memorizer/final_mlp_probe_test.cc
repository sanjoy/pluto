#include "src/llm/experiments/one_shot_memorizer/final_mlp_probe.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/padded_line_dataset.h"
#include "src/dataset/plain_text_tokenizer.h"
#include "src/llm/experiments/one_shot_memorizer/mlp_probe.h"
#include "src/llm/extract_top1_ids.h"
#include "src/llm/gpt2.h"
#include "src/llm/layer_hooks.h"
#include "src/llm/layers/embedding.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/norm.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

constexpr int kWidth = 16;
constexpr int kFeatures = 32;
constexpr int kVocab = 256;
constexpr float kGamma = 1.125f;
constexpr float kBeta = 0.0039067f;

template <typename T>
absl::Status Upload(cuda::Executor& executor, const Buffer& target,
                    absl::Span<const T> values) {
  if (target.size_bytes() != values.size() * sizeof(T))
    return absl::InvalidArgumentError("invalid final MLP test upload size");
  ASSIGN_OR_RETURN(auto host,
                   cuda::PageLockedHostArray<T>::CopyFrom(executor, values));
  return cuda::CudaStatus(
      cudaMemcpyAsync(target.data(), host.data(), host.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload final MLP fixture");
}

template <typename T>
absl::StatusOr<std::vector<T>> Download(cuda::Executor& executor,
                                        const Buffer& source) {
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<T>::Allocate(
                                  executor, source.size_bytes() / sizeof(T)));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host.data(), source.data(), source.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "read final MLP fixture"));
  RETURN_IF_ERROR(executor.Synchronize());
  return std::vector<T>(host.begin(), host.end());
}

absl::Status CloneWeights(cuda::Executor& executor, const Layer& source,
                          Layer& target) {
  const auto from = source.weights();
  const auto to = target.weights();
  if (from.size() != to.size())
    return absl::InvalidArgumentError("invalid final MLP test clone arity");
  for (size_t i = 0; i < from.size(); ++i) {
    if (from[i].size_bytes() != to[i].size_bytes())
      return absl::InvalidArgumentError("invalid final MLP test clone shape");
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(to[i].data(), from[i].data(), from[i].size_bytes(),
                        cudaMemcpyDeviceToDevice, executor.stream()),
        "clone test weights"));
  }
  return absl::OkStatus();
}

const BackwardState* Find(const BackwardState& state, absl::string_view name) {
  if (state.layer && state.layer->name() == name)
    return &state;
  for (const auto& child : state.children)
    if (const auto* found = Find(child, name))
      return found;
  return nullptr;
}

struct Copies {
  std::unique_ptr<FullyConnectedLayer> projection;
  std::unique_ptr<LayerNormLayer> norm;
  std::unique_ptr<EmbeddingLookupLayer> embedding;
  std::unique_ptr<LanguageModelingHeadLayer> head;
};

class FinalMlpProbeTest : public testing::Test {
 protected:
  void SetUp() override {
    auto created = cuda::Executor::Create();
    ASSERT_TRUE(created.ok()) << created.status();
    executor_ = std::move(*created);
  }
  void TearDown() override {
    if (!executor_)
      return;
    EXPECT_TRUE(executor_->Synchronize().ok());
  }

  absl::StatusOr<std::unique_ptr<PaddedLineDataSetIterator>> Dataset() {
    tokenizer::PlainTextTokenizer tokenizer;
    return PaddedLineDataSetIterator::Create(
        *executor_, "bbaaa\nbbaaaa\nbbaaaa\n", tokenizer,
        {.batch_size = 2,
         .context_length = kGpt2ContextLength,
         .prompt_tokens = 2,
         .eos_token = 0,
         .shuffle = false});
  }

  absl::StatusOr<std::unique_ptr<ComposedLayer>> Model(int blocks = 1) {
    const Gpt2Config config{.transformer_block_count = blocks,
                            .model_width = kWidth,
                            .attention_heads = 1,
                            .feed_forward_width = kFeatures,
                            .vocabulary_size = kVocab,
                            .pad_vocabulary = false};
    ASSIGN_OR_RETURN(auto model,
                     CreateGpt2(*executor_, DataType::BF16, 7, config));
    auto weights = model->weights();
    // Controlled test fixture only: no attention/position effects, W1 embeds
    // the identity, W2 is zero, and each block adds 4 to residual channel 0.
    for (const auto& weight : weights)
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemsetAsync(weight.data(), 0, weight.size_bytes(),
                          executor_->stream()),
          "clear final MLP fixture"));
    std::vector<float> embedding(kVocab * kWidth, 0);
    embedding['a' * kWidth] = 1;
    embedding['b' * kWidth] = -1;
    RETURN_IF_ERROR(Upload<float>(*executor_, weights[0], embedding));
    for (int block = 0; block < blocks; ++block) {
      const int base = 2 + 12 * block;
      RETURN_IF_ERROR(Upload<float>(*executor_, weights[base],
                                    std::vector<float>(kWidth, 1)));
      RETURN_IF_ERROR(Upload<float>(*executor_, weights[base + 6],
                                    std::vector<float>(kWidth, 1)));
      std::vector<float> first(kWidth * kFeatures, 0), bias(kWidth, 0);
      for (int d = 0; d < kWidth; ++d)
        first[d * kFeatures + d] = 1;
      bias[0] = 4.0003f;
      RETURN_IF_ERROR(Upload<float>(*executor_, weights[base + 8], first));
      RETURN_IF_ERROR(Upload<float>(*executor_, weights[base + 11], bias));
    }
    RETURN_IF_ERROR(Upload<float>(*executor_, weights[2 + 12 * blocks],
                                  std::vector<float>(kWidth, kGamma)));
    RETURN_IF_ERROR(Upload<float>(*executor_, weights[3 + 12 * blocks],
                                  std::vector<float>(kWidth, kBeta)));
    return model;
  }

  absl::StatusOr<Copies> Clone(const Layer& model, const DataBatch& batch,
                               int last_block = 0) {
    ASSIGN_OR_RETURN(auto forward, model.fwd(*executor_, {batch.inputs}));
    const auto* block =
        Find(forward.state, "transformer_block_" + std::to_string(last_block));
    const auto* mlp = block ? Find(*block, "mlp") : nullptr;
    const BackwardState* final_norm = nullptr;
    const BackwardState* original_head = nullptr;
    for (const auto& child : forward.state.children) {
      if (child.layer->name() == "LayerNormLayer")
        final_norm = &child;
      if (child.layer->name() == "LanguageModelingHeadLayer")
        original_head = &child;
    }
    if (!mlp || mlp->children.empty() || !final_norm || !original_head)
      return absl::InternalError("missing original clone sites");
    Copies copies;
    ASSIGN_OR_RETURN(copies.projection, FullyConnectedLayer::Create(
                                            *executor_, kFeatures, kWidth,
                                            DataType::BF16, kGpt2ContextLength));
    ASSIGN_OR_RETURN(copies.norm,
                     LayerNormLayer::Create(*executor_, kWidth, 1e-5f,
                                            DataType::BF16, kGpt2ContextLength));
    ASSIGN_OR_RETURN(
        copies.embedding,
        EmbeddingLookupLayer::Create(*executor_, kVocab, kWidth, DataType::BF16,
                                     kGpt2ContextLength, false));
    ASSIGN_OR_RETURN(copies.head,
                     LanguageModelingHeadLayer::Create(copies.embedding.get()));
    RETURN_IF_ERROR(CloneWeights(*executor_, *mlp->children.back().layer,
                                 *copies.projection));
    RETURN_IF_ERROR(CloneWeights(*executor_, *final_norm->layer, *copies.norm));
    RETURN_IF_ERROR(
        CloneWeights(*executor_, *original_head->layer, *copies.head));
    return copies;
  }

  absl::StatusOr<std::vector<int>> Predictions(const Layer& model,
                                               const DataBatch& batch,
                                               LayerHooks* hooks = nullptr) {
    ASSIGN_OR_RETURN(auto forward, model.fwd(*executor_, {batch.inputs}, hooks));
    ASSIGN_OR_RETURN(auto ids, ExtractTop1Ids(*executor_, forward.outputs[0],
                                              batch.targets, kVocab));
    return Download<int>(*executor_, ids);
  }

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(FinalMlpProbeTest,
       CapturesOnlySupervisedRowsAndIncomingLastMlpResidual) {
  auto model = Model(2);
  auto dataset = Dataset();
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_TRUE(dataset.ok()) << dataset.status();
  auto batch = (*dataset)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  auto captured = CaptureFinalMlpBatch(*executor_, **model, *batch, kWidth,
                                       kFeatures, 2, kVocab);
  ASSERT_TRUE(captured.ok()) << captured.status();
  EXPECT_EQ(captured->width, kWidth);
  EXPECT_EQ(captured->feature_width, kFeatures);
  EXPECT_EQ(captured->labels,
            (std::vector<int>{'a', 'a', 'a', 0, 'a', 'a', 'a', 'a', 0}));
  EXPECT_EQ(captured->sample_indices,
            (std::vector<int>{0, 0, 0, 0, 1, 1, 1, 1, 1}));
  EXPECT_EQ(captured->positions, (std::vector<int>{1, 2, 3, 4, 1, 2, 3, 4, 5}));
  EXPECT_EQ(captured->original_predictions, std::vector<int>(9, 'a'));
  EXPECT_EQ(captured->finalnorm_gamma, std::vector<float>(kWidth, kGamma));
  EXPECT_EQ(captured->finalnorm_beta, std::vector<float>(kWidth, kBeta));
  ASSERT_EQ(captured->normalized_inputs.size(), 9u * kWidth);
  ASSERT_EQ(captured->features.size(), 9u * kFeatures);
  ASSERT_EQ(captured->residuals.size(), 9u * kWidth);
  for (size_t row = 0; row < 9; ++row) {
    // The earlier block has already added 4, but the last block has not.
    EXPECT_EQ(captured->residuals[row * kWidth],
              captured->positions[row] == 1 ? 3 : 5);
    EXPECT_GT(captured->normalized_inputs[row * kWidth], 3.8f);
    EXPECT_GT(captured->features[row * kFeatures], 3.8f);
    for (int d = 1; d < kWidth; ++d)
      EXPECT_EQ(captured->residuals[row * kWidth + d], 0);
  }
  ASSERT_TRUE(Upload<int>(*executor_, batch->targets,
                          std::vector<int>(2 * kGpt2ContextLength, -1))
                  .ok());
  batch->supervised_row_count = 0;
  captured = CaptureFinalMlpBatch(*executor_, **model, *batch, kWidth,
                                  kFeatures, 2, kVocab);
  ASSERT_TRUE(captured.ok()) << captured.status();
  EXPECT_TRUE(captured->features.empty());
  EXPECT_TRUE(captured->labels.empty());
  EXPECT_EQ(captured->finalnorm_beta.size(), static_cast<size_t>(kWidth));
}

TEST_F(FinalMlpProbeTest,
       ClonedNormHeadProjectionPreserveLogitsAndExistingEvaluators) {
  auto model = Model();
  auto dataset = Dataset();
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_TRUE(dataset.ok()) << dataset.status();
  auto batch = (*dataset)->Next();
  ASSERT_TRUE(batch.ok());
  auto copies = Clone(**model, *batch);
  ASSERT_TRUE(copies.ok()) << copies.status();
  auto original = (*model)->fwd(*executor_, {batch->inputs});
  ASSERT_TRUE(original.ok());
  auto expected = Download<float>(*executor_, original->outputs[0]);
  ASSERT_TRUE(expected.ok());
  for (const Layer* projection :
       {static_cast<const Layer*>(nullptr),
        static_cast<const Layer*>(copies->projection.get())}) {
    auto wrapper = CreateFinalMlpReplacement(**model, projection, *copies->norm,
                                             *copies->head, 0);
    ASSERT_TRUE(wrapper.ok()) << wrapper.status();
    EXPECT_TRUE((*wrapper)->weights().empty());
    auto forward = (*wrapper)->fwd(*executor_, {batch->inputs});
    ASSERT_TRUE(forward.ok()) << forward.status();
    auto actual = Download<float>(*executor_, forward->outputs[0]);
    ASSERT_TRUE(actual.ok());
    EXPECT_EQ(*actual, *expected);
    EXPECT_FALSE(
        (*wrapper)->bwd(*executor_, {}, std::move(forward->state)).ok());
    auto evaluation =
        EvaluateMlpReplacements(*executor_, **wrapper, **dataset, {}, kVocab);
    ASSERT_TRUE(evaluation.ok()) << evaluation.status();
    EXPECT_EQ(evaluation->correct_targets, 11);
    EXPECT_EQ(evaluation->targets, 14);
    auto greedy = VerifyMlpGreedyCompletions(*executor_, **wrapper, **dataset,
                                             {}, kVocab);
    ASSERT_TRUE(greedy.ok()) << greedy.status();
    EXPECT_EQ(greedy->exact_sentences, 0);
    EXPECT_EQ(greedy->generated_targets, 14);
    // Evaluation reuses the dataset's device buffers. Restore the first batch.
    ASSERT_TRUE((*dataset)->Reset().ok());
    batch = (*dataset)->Next();
    ASSERT_TRUE(batch.ok());
  }
}

TEST_F(FinalMlpProbeTest,
       FreshGeluReplacementChainsHooksAndLeavesOriginalUnchanged) {
  auto model = Model();
  auto dataset = Dataset();
  ASSERT_TRUE(model.ok());
  ASSERT_TRUE(dataset.ok());
  auto batch = (*dataset)->Next();
  ASSERT_TRUE(batch.ok());
  auto copies = Clone(**model, *batch);
  ASSERT_TRUE(copies.ok()) << copies.status();
  const auto before = Predictions(**model, *batch);
  ASSERT_TRUE(before.ok());
  // Identity initialization also clears the previously cloned +4.0003 bias.
  ASSERT_TRUE(copies->projection->InitializeIdentity(-1).ok());
  auto wrapper = CreateFinalMlpReplacement(**model, copies->projection.get(),
                                           *copies->norm, *copies->head, 0);
  ASSERT_TRUE(wrapper.ok()) << wrapper.status();
  auto changed = Predictions(**wrapper, *batch);
  ASSERT_TRUE(changed.ok()) << changed.status();
  EXPECT_EQ((*changed)[1], 'b');
  EXPECT_EQ((*changed)[2], 'b');
  EXPECT_NE(*changed, *before);

  // An outer hook changes the current GELU handle. The replacement must use
  // this new handle, not its pre-hook value or a previous pass's activation.
  auto zeros =
      Buffer::Allocate(*executor_, static_cast<size_t>(2) * kGpt2ContextLength *
                                       kFeatures * sizeof(uint16_t));
  ASSERT_TRUE(zeros.ok());
  ASSERT_TRUE(
      cuda::CudaStatus(cudaMemsetAsync(zeros->data(), 0, zeros->size_bytes(),
                                       executor_->stream()),
                       "clear GELU fixture")
          .ok());
  std::vector<std::string> scopes;
  int gelu_count = 0, norm_count = 0, head_count = 0;
  LayerHooks hooks;
  hooks.enter_combinator = [&](cuda::Executor&, absl::string_view name) {
    scopes.emplace_back(name);
    return absl::OkStatus();
  };
  hooks.exit_combinator = [&](cuda::Executor&) {
    if (scopes.empty())
      return absl::InternalError("test hook underflow");
    scopes.pop_back();
    return absl::OkStatus();
  };
  hooks.activation_hook = [&](cuda::Executor&, absl::string_view name,
                              absl::Span<const ActivationType>,
                              absl::Span<Buffer> outputs) {
    if (name == "GeluLayer") {
      ++gelu_count;
      outputs[0] = *zeros;
    }
    if (name == "LayerNormLayer")
      ++norm_count;
    if (name == "LanguageModelingHeadLayer")
      ++head_count;
    return absl::OkStatus();
  };
  auto intervened = Predictions(**wrapper, *batch, &hooks);
  ASSERT_TRUE(intervened.ok()) << intervened.status();
  EXPECT_EQ((*intervened)[1], 'b');
  EXPECT_EQ((*intervened)[2], 'a');
  EXPECT_TRUE(scopes.empty());
  EXPECT_EQ(gelu_count, 1);
  EXPECT_EQ(norm_count, 3);  // Replacement internals are intentionally hidden.
  EXPECT_EQ(head_count, 1);
  const auto repeated = Predictions(**wrapper, *batch);
  ASSERT_TRUE(repeated.ok());
  EXPECT_EQ(*repeated, *changed);
  const auto after = Predictions(**model, *batch);
  ASSERT_TRUE(after.ok());
  EXPECT_EQ(*after, *before);
}

TEST_F(FinalMlpProbeTest,
       ReplacedHeadAndNormUseFreshOutputsWithoutChangingEmbeddings) {
  auto model = Model();
  auto dataset = Dataset();
  ASSERT_TRUE(model.ok());
  ASSERT_TRUE(dataset.ok());
  auto batch = (*dataset)->Next();
  ASSERT_TRUE(batch.ok());
  auto copies = Clone(**model, *batch);
  ASSERT_TRUE(copies.ok());
  const auto original = Predictions(**model, *batch);
  ASSERT_TRUE(original.ok());
  auto table = Download<float>(*executor_, copies->embedding->weights()[0]);
  ASSERT_TRUE(table.ok());
  for (float& value : *table)
    value = -value;
  ASSERT_TRUE(
      Upload<float>(*executor_, copies->embedding->weights()[0], *table).ok());
  auto wrapper = CreateFinalMlpReplacement(**model, nullptr, *copies->norm,
                                           *copies->head, 0);
  ASSERT_TRUE(wrapper.ok());
  auto reversed = Predictions(**wrapper, *batch);
  ASSERT_TRUE(reversed.ok()) << reversed.status();
  EXPECT_EQ((*reversed)[1], 'b');
  EXPECT_EQ((*reversed)[2], 'b');
  // Reversing replacement gamma restores the first-channel ranking. This
  // catches a head that accidentally reads the old normalization output.
  ASSERT_TRUE(Upload<float>(*executor_, copies->norm->weights()[0],
                            std::vector<float>(kWidth, -kGamma))
                  .ok());
  auto restored = Predictions(**wrapper, *batch);
  ASSERT_TRUE(restored.ok());
  EXPECT_EQ(*restored, *original);
  auto untouched = Predictions(**model, *batch);
  ASSERT_TRUE(untouched.ok());
  EXPECT_EQ(*untouched, *original);
}

TEST_F(FinalMlpProbeTest,
       RejectsBadCaptureAndReplacementShapesAndUnwindsOuterErrors) {
  auto model = Model();
  auto dataset = Dataset();
  ASSERT_TRUE(model.ok());
  ASSERT_TRUE(dataset.ok());
  auto batch = (*dataset)->Next();
  ASSERT_TRUE(batch.ok());
  EXPECT_FALSE(
      CaptureFinalMlpBatch(*executor_, **model, *batch, 0, kFeatures, 1, kVocab)
          .ok());
  EXPECT_FALSE(CaptureFinalMlpBatch(*executor_, **model, *batch, kWidth - 1,
                                    kFeatures, 1, kVocab)
                   .ok());
  EXPECT_FALSE(CaptureFinalMlpBatch(*executor_, **model, *batch, kWidth,
                                    kFeatures + 1, 1, kVocab)
                   .ok());
  EXPECT_FALSE(CaptureFinalMlpBatch(*executor_, **model, *batch, kWidth,
                                    kFeatures, 2, kVocab)
                   .ok());
  auto wrong_count = *batch;
  wrong_count.supervised_row_count = 1;
  EXPECT_FALSE(CaptureFinalMlpBatch(*executor_, **model, wrong_count, kWidth,
                                    kFeatures, 1, kVocab)
                   .ok());
  auto copies = Clone(**model, *batch);
  ASSERT_TRUE(copies.ok());
  auto wrong = FullyConnectedLayer::Create(*executor_, kFeatures, kWidth + 1,
                                           DataType::BF16, kGpt2ContextLength);
  ASSERT_TRUE(wrong.ok());
  EXPECT_FALSE(CreateFinalMlpReplacement(**model, wrong->get(), *copies->norm,
                                         *copies->head, 0)
                   .ok());
  EXPECT_FALSE(CreateFinalMlpReplacement(**model, nullptr, *copies->norm,
                                         *copies->head, -1)
                   .ok());
  wrong = FullyConnectedLayer::Create(*executor_, kFeatures + 1, kWidth,
                                      DataType::BF16, kGpt2ContextLength);
  ASSERT_TRUE(wrong.ok());
  auto mismatched = CreateFinalMlpReplacement(**model, wrong->get(),
                                              *copies->norm, *copies->head, 0);
  ASSERT_TRUE(mismatched.ok());
  EXPECT_FALSE(Predictions(**mismatched, *batch).ok());
  auto wrapper = CreateFinalMlpReplacement(**model, nullptr, *copies->norm,
                                           *copies->head, 0);
  ASSERT_TRUE(wrapper.ok());
  int depth = 0;
  LayerHooks hooks;
  hooks.enter_combinator = [&](cuda::Executor&, absl::string_view) {
    ++depth;
    return absl::OkStatus();
  };
  hooks.exit_combinator = [&](cuda::Executor&) {
    --depth;
    return absl::OkStatus();
  };
  hooks.activation_hook = [](cuda::Executor&, absl::string_view name,
                             absl::Span<const ActivationType>,
                             absl::Span<Buffer>) {
    return name == "GeluLayer"
               ? absl::AbortedError("intentional outer-hook failure")
               : absl::OkStatus();
  };
  EXPECT_FALSE(Predictions(**wrapper, *batch, &hooks).ok());
  EXPECT_EQ(depth, 0);
  EXPECT_TRUE(Predictions(**wrapper, *batch).ok());
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
