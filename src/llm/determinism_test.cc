#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/dataset.h"
#include "src/llm/adamw_optimizer.h"
#include "src/llm/layer.h"
#include "src/llm/layers/attention.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/llm/layers/embedding.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/gelu.h"
#include "src/llm/layers/norm.h"
#include "src/llm/layers/reference_test_util.h"
#include "src/llm/layers/sparse_autoencoder.h"
#include "src/llm/sampling.h"
#include "src/llm/trainer.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

constexpr int kContext = 16;
constexpr int kWidth = 32;
// Deliberately not a tile multiple: the tied head must mask its padded lanes.
constexpr int kVocabulary = 47;
constexpr int kTrainingRows = 4 * kContext;
constexpr int kUpdates = 4;

// This is the production GPT-2 graph at a unit-test scale, not a fake layer:
// token/position embeddings, two pre-LN attention + MLP residual blocks,
// final LN, and a head sharing the token embedding's weight and gradient.
// Repeated corpus IDs below make embedding backward perform real reductions.
absl::StatusOr<std::unique_ptr<ComposedLayer>> MakeLanguageModel(
    cuda::Executor& executor, DataType type, uint64_t seed) {
  ComposedLayerBuilder model;
  RETURN_IF_ERROR(model.add(EmbeddingLookupLayer::Create(
      executor, kVocabulary, kWidth, type, kContext)));
  auto* embedding = static_cast<EmbeddingLookupLayer*>(model.back());
  RETURN_IF_ERROR(embedding->InitializeNormal(0.08f, seed));
  RETURN_IF_ERROR(model.add(
      PositionEmbeddingLayer::Create(executor, kContext, kWidth, type)));
  RETURN_IF_ERROR(static_cast<PositionEmbeddingLayer*>(model.back())
                      ->InitializeNormal(0.04f, seed + 1));

  for (int block = 0; block < 2; ++block) {
    const uint64_t block_seed = seed + 100 + 10 * block;
    ComposedLayerBuilder attention;
    RETURN_IF_ERROR(attention.add(
        LayerNormLayer::Create(executor, kWidth, 1e-5f, type, kContext)));
    RETURN_IF_ERROR(attention.add(FullyConnectedLayer::Create(
        executor, kWidth, 3 * kWidth, type, kContext)));
    RETURN_IF_ERROR(static_cast<FullyConnectedLayer*>(attention.back())
                        ->InitializeNormal(0.06f, block_seed));
    RETURN_IF_ERROR(attention.add(
        AttentionLayer::Create(executor, kContext, 2, kWidth, type)));
    RETURN_IF_ERROR(attention.add(
        FullyConnectedLayer::Create(executor, kWidth, kWidth, type, kContext)));
    RETURN_IF_ERROR(static_cast<FullyConnectedLayer*>(attention.back())
                        ->InitializeNormal(0.03f, block_seed + 1));
    ASSIGN_OR_RETURN(
        auto attention_branch,
        attention.create("block_" + std::to_string(block) + "_attention"));
    RETURN_IF_ERROR(
        model.add(ResidualLayer::Create(std::move(attention_branch))));

    ComposedLayerBuilder mlp;
    RETURN_IF_ERROR(mlp.add(
        LayerNormLayer::Create(executor, kWidth, 1e-5f, type, kContext)));
    RETURN_IF_ERROR(mlp.add(FullyConnectedLayer::Create(
        executor, kWidth, 2 * kWidth, type, kContext)));
    RETURN_IF_ERROR(static_cast<FullyConnectedLayer*>(mlp.back())
                        ->InitializeNormal(0.06f, block_seed + 2));
    RETURN_IF_ERROR(
        mlp.add(GeluLayer::Create(executor, 2 * kWidth, type, kContext)));
    RETURN_IF_ERROR(mlp.add(FullyConnectedLayer::Create(
        executor, 2 * kWidth, kWidth, type, kContext)));
    RETURN_IF_ERROR(static_cast<FullyConnectedLayer*>(mlp.back())
                        ->InitializeNormal(0.03f, block_seed + 3));
    ASSIGN_OR_RETURN(auto mlp_branch,
                     mlp.create("block_" + std::to_string(block) + "_mlp"));
    RETURN_IF_ERROR(model.add(ResidualLayer::Create(std::move(mlp_branch))));
  }
  RETURN_IF_ERROR(model.add(
      LayerNormLayer::Create(executor, kWidth, 1e-5f, type, kContext)));
  RETURN_IF_ERROR(model.add(LanguageModelingHeadLayer::Create(embedding)));
  return model.create("deterministic_language_model");
}

// All CUDA transfers use pinned staging. Compare raw bytes rather than floats:
// approximate equality would hide precisely the reduction-order regression
// these tests are intended to catch. The CPU string only retains the result
// after the asynchronous transfer and stream synchronization have completed.
absl::StatusOr<std::string> ReadBytes(cuda::Executor& executor,
                                      absl::Span<const Buffer> buffers) {
  size_t size = 0;
  for (const Buffer& buffer : buffers)
    size += buffer.size_bytes();
  ASSIGN_OR_RETURN(
      auto staging,
      cuda::PageLockedHostArray<unsigned char>::Allocate(executor, size));
  size_t offset = 0;
  for (const Buffer& buffer : buffers) {
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(staging.data() + offset, buffer.data(),
                        buffer.size_bytes(), cudaMemcpyDeviceToHost,
                        executor.stream()),
        "read exact replay snapshot"));
    offset += buffer.size_bytes();
  }
  RETURN_IF_ERROR(executor.Synchronize());
  return std::string(reinterpret_cast<const char*>(staging.data()), size);
}

absl::StatusOr<std::string> ReadMeanLoss(cuda::Executor& executor,
                                         const Layer& model,
                                         const Layer& loss_layer,
                                         DataSetIterator& data) {
  ASSIGN_OR_RETURN(
      auto loss,
      Evaluate(executor, model,
               EvaluationOptions{
                   .loss_layer = loss_layer, .eval_data = data, .batches = 2}));
  ASSIGN_OR_RETURN(auto bytes, ReadBytes(executor, {loss}));
  if (bytes.size() != sizeof(float))
    return absl::InternalError("Evaluate did not return one FP32 scalar");
  float value;
  std::memcpy(&value, bytes.data(), sizeof(value));
  if (!std::isfinite(value))
    return absl::InternalError("replay evaluation produced a nonfinite loss");
  return bytes;
}

struct Trajectory {
  // Index zero is initialization; later entries are completed AdamW updates.
  std::vector<std::string> weights;
  std::vector<std::string> gradients;
  std::vector<std::string> losses;
  std::vector<std::pair<int, uint64_t>> callback_losses;
  std::vector<std::string> outputs;
  std::vector<std::vector<int>> completions;
  std::vector<uint64_t> statistics;
};

uint64_t DoubleBits(double value) {
  uint64_t bits;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

absl::Status RecordTraining(cuda::Executor& executor, Layer& model,
                            Layer& loss_layer, DataSetIterator& training,
                            DataSetIterator& evaluation,
                            Trajectory& trajectory) {
  ASSIGN_OR_RETURN(auto optimizer,
                   AdamWOptimizer::Create(executor, model,
                                          AdamWConfig{.learning_rate = 1e-3f}));
  auto record = [&]() -> absl::Status {
    ASSIGN_OR_RETURN(auto weights, ReadBytes(executor, model.weights()));
    ASSIGN_OR_RETURN(auto loss,
                     ReadMeanLoss(executor, model, loss_layer, evaluation));
    trajectory.weights.push_back(std::move(weights));
    trajectory.losses.push_back(std::move(loss));
    return absl::OkStatus();
  };
  RETURN_IF_ERROR(record());
  // Inspect backward directly as well as its eventual weight update: rounding
  // in AdamW could otherwise conceal a tiny nondeterministic gradient change.
  // Resetting both iterator and accumulators replays the very same batch.
  for (int repetition = 0; repetition < 3; ++repetition) {
    RETURN_IF_ERROR(training.Reset());
    RETURN_IF_ERROR(optimizer->ZeroGrad());
    ASSIGN_OR_RETURN(auto batch, training.Next());
    ASSIGN_OR_RETURN(auto model_fwd, model.fwd(executor, {batch.inputs}));
    BufferVec loss_inputs = model_fwd.outputs;
    loss_inputs.push_back(batch.targets);
    ASSIGN_OR_RETURN(auto loss_fwd, loss_layer.fwd(executor, loss_inputs));
    ASSIGN_OR_RETURN(auto output_gradients,
                     loss_layer.bwd(executor, {}, std::move(loss_fwd.state)));
    if (output_gradients.size() != model_fwd.outputs.size())
      return absl::InternalError("loss returned an unexpected gradient count");
    ASSIGN_OR_RETURN(
        auto input_gradients,
        model.bwd(executor, output_gradients, std::move(model_fwd.state)));
    (void)input_gradients;
    ASSIGN_OR_RETURN(auto gradients, ReadBytes(executor, model.gradients()));
    trajectory.gradients.push_back(std::move(gradients));
  }
  TrainingOptions options{.loss_layer = loss_layer,
                          .optimizer = *optimizer,
                          .training_data = training};
  options.max_steps = kUpdates;
  options.evaluation_interval = 1;
  options.evaluation_batches = 2;
  options.evaluation_data = &evaluation;
  options.step_callback = [&](int) { return record(); };
  options.evaluation_callback = [&](int step, double loss) {
    trajectory.callback_losses.emplace_back(step, DoubleBits(loss));
  };
  ASSIGN_OR_RETURN(auto result, Train(executor, model, options));
  if (result.steps_completed != kUpdates || optimizer->step() != kUpdates)
    return absl::InternalError("replay did not perform all requested updates");
  return absl::OkStatus();
}

// Retaining extra initialized allocations in the second run changes addresses
// and allocator history without changing model inputs or RNG state. A correct
// implementation must not depend on either, or on a particular CUDA stream ID.
absl::StatusOr<Buffer> PerturbAllocations(cuda::Executor& executor,
                                          bool perturb) {
  ASSIGN_OR_RETURN(auto buffer,
                   Buffer::Allocate(executor, perturb ? 32768 : 16));
  RETURN_IF_ERROR(
      cuda::CudaStatus(cudaMemsetAsync(buffer.data(), 0xa5, buffer.size_bytes(),
                                       executor.stream()),
                       "initialize replay allocation perturbation"));
  return buffer;
}

absl::Status RecordCompletions(cuda::Executor& executor, const Layer& model,
                               Trajectory& trajectory) {
  ASSIGN_OR_RETURN(auto input,
                   cuda::PageLockedHostArray<int>::Allocate(executor, kContext));
  ASSIGN_OR_RETURN(auto device_input,
                   Buffer::Allocate(executor, input.size_bytes()));
  for (double temperature : {0.0, 0.8, 1.5}) {
    std::fill(input.begin(), input.end(), 0);
    input[0] = 3;
    input[1] = 1;
    input[2] = 3;
    input[3] = 7;
    std::mt19937 random(918273);
    std::vector<int> completion;
    for (int step = 0; step < 8; ++step) {
      // ReadDeviceFloats below synchronizes before the next iteration changes
      // input, so this pinned upload cannot race its CPU producer.
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpyAsync(device_input.data(), input.data(), input.size_bytes(),
                          cudaMemcpyHostToDevice, executor.stream()),
          "upload deterministic inference context"));

      ASSIGN_OR_RETURN(auto logits_fwd, model.fwd(executor, {device_input}));
      auto logits = std::move(logits_fwd.outputs[0]);

      ASSIGN_OR_RETURN(auto host_logits, ReadDeviceFloats(executor, logits));
      const int padded_vocabulary = host_logits.size() / kContext;
      const int prediction_row = 3 + step;
      ASSIGN_OR_RETURN(
          int next,
          SelectNextToken(host_logits.span().subspan(
                              prediction_row * padded_vocabulary, kVocabulary),
                          temperature, random));
      completion.push_back(next);
      input[4 + step] = next;
      trajectory.outputs.emplace_back(
          reinterpret_cast<const char*>(host_logits.data()),
          host_logits.size_bytes());
    }
    trajectory.completions.push_back(std::move(completion));
  }
  return absl::OkStatus();
}

absl::StatusOr<Trajectory> RunLanguageModel(DataType type, bool perturb) {
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto allocation_noise,
                   PerturbAllocations(*executor, perturb));
  ASSIGN_OR_RETURN(auto model, MakeLanguageModel(*executor, type, 193));
  ASSIGN_OR_RETURN(auto loss, CrossEntropyLossLayer::Create(
                                  *executor, kVocabulary, type, kContext));
  ASSIGN_OR_RETURN(auto corpus,
                   cuda::PageLockedHostArray<int>::Allocate(*executor, 257));
  for (size_t index = 0; index < corpus.size(); ++index) {
    // Most IDs repeat several times per batch, with nonuniform frequencies.
    corpus[index] = (index * index + 7 * index + index / 9) % 13;
  }
  ASSIGN_OR_RETURN(
      auto training,
      InMemoryDataSetIterator::Create(
          *executor, corpus,
          InMemoryDataSetOptions{.batch_size = kTrainingRows / kContext,
                                 .context_length = kContext,
                                 .order = InMemoryDataSetOrder::kRandom,
                                 .seed = 817263}));
  ASSIGN_OR_RETURN(
      auto evaluation,
      InMemoryDataSetIterator::Create(
          *executor, corpus,
          InMemoryDataSetOptions{.batch_size = kTrainingRows / kContext,
                                 .context_length = kContext,
                                 .order = InMemoryDataSetOrder::kSequential}));
  Trajectory trajectory;
  RETURN_IF_ERROR(RecordTraining(*executor, *model, *loss, *training,
                                 *evaluation, trajectory));
  RETURN_IF_ERROR(RecordCompletions(*executor, *model, trajectory));
  RETURN_IF_ERROR(executor->Synchronize());
  return trajectory;
}

class FixedActivationDataSet final : public DataSetIterator {
 public:
  explicit FixedActivationDataSet(DataBatch batch) : batch_(std::move(batch)) {}
  absl::StatusOr<DataBatch> Next() override { return batch_; }
  absl::Status Reset() override { return absl::OkStatus(); }

 private:
  DataBatch batch_;
};

absl::StatusOr<Trajectory> RunSparseAutoEncoder(
    DataType type, SparseAutoEncoderLayer::Mode mode, bool perturb) {
  // SAE requires 16-row tiles; three tiles avoid a power-of-two batch shape.
  constexpr int kRows = 48;
  constexpr int kFeatures = 96;
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto allocation_noise,
                   PerturbAllocations(*executor, perturb));
  ASSIGN_OR_RETURN(auto model,
                   SparseAutoEncoderLayer::Create(*executor, kWidth, kFeatures,
                                                  type, mode, kRows / 3));
  RETURN_IF_ERROR(model->InitializeNormal(0.1f, 817263));
  ASSIGN_OR_RETURN(auto loss,
                   SparseAutoEncoderLossLayer::Create(
                       *executor, kWidth, kFeatures, 0.5f, type, kRows / 3));
  std::vector<float> values(kRows * kWidth);
  for (size_t index = 0; index < values.size(); ++index)
    values[index] = (static_cast<int>(index % 29) - 14) * 0.0625f;
  ASSIGN_OR_RETURN(auto activation,
                   MakeActivationBufferPair(*executor, values, type));
  // Keep the token count unchanged while exercising multi-token samples.
  FixedActivationDataSet training(
      {activation.device, activation.device, 3, kRows / 3});
  FixedActivationDataSet evaluation(
      {activation.device, activation.device, 3, kRows / 3});
  Trajectory trajectory;
  RETURN_IF_ERROR(RecordTraining(*executor, *model, *loss, training, evaluation,
                                 trajectory));

  ASSIGN_OR_RETURN(auto reconstruction_fwd,
                   model->fwd(*executor, {activation.device}));
  auto reconstruction = std::move(reconstruction_fwd.outputs[0]);

  auto latents = reconstruction_fwd.outputs[1];
  ASSIGN_OR_RETURN(auto outputs,
                   ReadBytes(*executor, {reconstruction, latents}));
  trajectory.outputs.push_back(std::move(outputs));
  if (mode == SparseAutoEncoderLayer::Mode::kCollectStatistics) {
    ASSIGN_OR_RETURN(
        auto stats, model->ReadZStatistics(*executor, reconstruction_fwd.state));
    trajectory.statistics = {static_cast<uint64_t>(stats.rows),
                             static_cast<uint64_t>(stats.feature_dim),
                             static_cast<uint64_t>(stats.active_count),
                             DoubleBits(stats.mean),
                             DoubleBits(stats.standard_deviation),
                             DoubleBits(stats.maximum)};
  }
  RETURN_IF_ERROR(executor->Synchronize());
  return trajectory;
}

// Report the first differing byte rather than dumping an entire weight table
// into test output. Unlike EXPECT_FLOAT_EQ, this also detects signed-zero and
// single-ULP differences, as well as different NaN payloads.
testing::AssertionResult IdenticalBytes(const std::string& expected,
                                        const std::string& actual) {
  if (expected.size() != actual.size()) {
    return testing::AssertionFailure()
           << "different byte counts: " << expected.size() << " vs "
           << actual.size();
  }
  for (size_t index = 0; index < expected.size(); ++index) {
    if (expected[index] != actual[index]) {
      return testing::AssertionFailure()
             << "first difference at byte " << index << ": "
             << static_cast<unsigned int>(
                    static_cast<unsigned char>(expected[index]))
             << " vs "
             << static_cast<unsigned int>(
                    static_cast<unsigned char>(actual[index]));
    }
  }
  return testing::AssertionSuccess();
}

void ExpectIdenticalTrajectory(const Trajectory& expected,
                               const Trajectory& actual) {
  ASSERT_EQ(expected.weights.size(), static_cast<size_t>(kUpdates + 1));
  ASSERT_EQ(actual.weights.size(), expected.weights.size());
  ASSERT_EQ(actual.losses.size(), expected.losses.size());
  ASSERT_EQ(expected.gradients.size(), 3u);
  ASSERT_EQ(actual.gradients.size(), expected.gradients.size());
  ASSERT_EQ(actual.outputs.size(), expected.outputs.size());
  for (size_t step = 0; step < expected.weights.size(); ++step) {
    SCOPED_TRACE(testing::Message() << "completed step " << step);
    EXPECT_TRUE(IdenticalBytes(expected.weights[step], actual.weights[step]));
    EXPECT_TRUE(IdenticalBytes(expected.losses[step], actual.losses[step]));
  }
  ASSERT_EQ(expected.callback_losses.size(), static_cast<size_t>(kUpdates));
  for (size_t repetition = 0; repetition < expected.gradients.size();
       ++repetition) {
    SCOPED_TRACE(testing::Message() << "backward repetition " << repetition);
    EXPECT_TRUE(IdenticalBytes(expected.gradients.front(),
                               expected.gradients[repetition]));
    EXPECT_TRUE(IdenticalBytes(expected.gradients[repetition],
                               actual.gradients[repetition]));
  }
  EXPECT_EQ(actual.callback_losses, expected.callback_losses);
  for (size_t index = 0; index < expected.outputs.size(); ++index) {
    SCOPED_TRACE(testing::Message() << "inference output " << index);
    EXPECT_TRUE(IdenticalBytes(expected.outputs[index], actual.outputs[index]));
  }
  EXPECT_EQ(actual.completions, expected.completions);
  EXPECT_EQ(actual.statistics, expected.statistics);
  // Reproducibility alone is insufficient: a no-op optimizer also replays.
  EXPECT_FALSE(expected.weights.front() == expected.weights.back());
}

class DeterministicTrainingTest : public testing::TestWithParam<DataType> {};

TEST_P(DeterministicTrainingTest, TransformerTrainingAndSamplingReplayExactly) {
  auto first = RunLanguageModel(GetParam(), false);
  ASSERT_TRUE(first.ok()) << first.status();
  auto second = RunLanguageModel(GetParam(), true);
  ASSERT_TRUE(second.ok()) << second.status();
  ExpectIdenticalTrajectory(*first, *second);
  ASSERT_EQ(first->completions.size(), 3u);
  for (const auto& completion : first->completions)
    EXPECT_EQ(completion.size(), 8u);

  // A different seed must really change the starting state. This guards
  // against accidentally obtaining reproducibility by ignoring the seed.
  auto executor = cuda::Executor::Create();
  ASSERT_TRUE(executor.ok()) << executor.status();
  auto different_model = MakeLanguageModel(**executor, GetParam(), 194);
  ASSERT_TRUE(different_model.ok()) << different_model.status();
  auto different_weights = ReadBytes(**executor, (*different_model)->weights());
  ASSERT_TRUE(different_weights.ok()) << different_weights.status();
  EXPECT_FALSE(first->weights.front() == *different_weights);
}

TEST_P(DeterministicTrainingTest, SparseTrainingAndStatisticsReplayExactly) {
  using Mode = SparseAutoEncoderLayer::Mode;
  auto first = RunSparseAutoEncoder(GetParam(), Mode::kDefault, false);
  ASSERT_TRUE(first.ok()) << first.status();
  auto second = RunSparseAutoEncoder(GetParam(), Mode::kDefault, true);
  ASSERT_TRUE(second.ok()) << second.status();
  ExpectIdenticalTrajectory(*first, *second);

  auto stats_first =
      RunSparseAutoEncoder(GetParam(), Mode::kCollectStatistics, false);
  ASSERT_TRUE(stats_first.ok()) << stats_first.status();
  auto stats_second =
      RunSparseAutoEncoder(GetParam(), Mode::kCollectStatistics, true);
  ASSERT_TRUE(stats_second.ok()) << stats_second.status();
  ASSERT_FALSE(stats_first->statistics.empty());
  ExpectIdenticalTrajectory(*stats_first, *stats_second);

  // The extra statistics kernels must not change the model or its updates.
  stats_first->statistics.clear();
  ExpectIdenticalTrajectory(*first, *stats_first);
}

INSTANTIATE_TEST_SUITE_P(SupportedComputeTypes, DeterministicTrainingTest,
                         testing::Values(DataType::FP16, DataType::BF16),
                         [](const testing::TestParamInfo<DataType>& info) {
                           return info.param == DataType::FP16 ? "FP16"
                                                               : "BF16";
                         });

}  // namespace
}  // namespace pluto::llm
