#include "src/llm/trainer.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/dataset.h"
#include "src/llm/layer.h"
#include "src/llm/layers/attention.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/embedding.h"
#include "src/llm/layers/sparse_autoencoder.h"
#include "src/llm/optimizer.h"

namespace pluto::llm {
namespace {

class FakeModel final : public Layer {
 public:
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return DataType::FP16; }

  mutable int forward_calls = 0;
  int backward_calls = 0;

 private:
  absl::StatusOr<FwdResult> fwd_impl(
      cuda::Executor& executor,
      absl::Span<const Buffer> inputs) const override {
    BackwardState state;
    ++forward_calls;
    return FwdResult{std::move(inputs[0]), std::move(state)};
  }

  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state) override {
    ++backward_calls;
    return BufferVec{};
  }
};

class FakeLoss final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<FakeLoss>> Create(
      cuda::Executor& executor, absl::Span<const float> losses) {
    auto device_losses =
        Buffer::Allocate(executor, losses.size() * sizeof(float));
    if (!device_losses.ok())
      return device_losses.status();
    auto gradient = Buffer::Allocate(executor, losses.size() * sizeof(float));
    if (!gradient.ok())
      return gradient.status();
    if (cudaMemcpyAsync(device_losses->data(), losses.data(),
                        device_losses->size_bytes(), cudaMemcpyHostToDevice,
                        executor.stream()) != cudaSuccess) {
      return absl::Status(absl::StatusCode::kInternal,
                          "failed to initialize fake losses");
    }
    const absl::Status synchronized = executor.Synchronize();
    if (!synchronized.ok())
      return synchronized;
    return absl::WrapUnique(
        new FakeLoss(std::move(*device_losses), std::move(*gradient)));
  }

  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return DataType::FP16; }

  mutable int forward_calls = 0;
  int backward_calls = 0;

 private:
  absl::StatusOr<FwdResult> fwd_impl(
      cuda::Executor& executor,
      absl::Span<const Buffer> inputs) const override {
    BackwardState state;
    ++forward_calls;
    return FwdResult{std::move(losses_), std::move(state)};
  }

  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state) override {
    ++backward_calls;
    return BufferVec{gradient_};
  }

  FakeLoss(Buffer losses, Buffer gradient)
      : losses_(std::move(losses)), gradient_(std::move(gradient)) {}

  Buffer losses_;
  Buffer gradient_;
};

class FakeOptimizer final : public Optimizer {
 public:
  absl::Status ZeroGrad() override {
    ++zero_grad_calls;
    return absl::OkStatus();
  }
  absl::Status Step() override {
    ++steps;
    if (step_action)
      return step_action();
    return absl::OkStatus();
  }
  int step() const override { return steps; }
  size_t parameter_tensor_count() const override { return 0; }

  int zero_grad_calls = 0;
  int steps = 0;
  std::function<absl::Status()> step_action;
};

class FixedActivationDataSetIterator final : public DataSetIterator {
 public:
  FixedActivationDataSetIterator(Buffer data, int32_t batch_size,
                                 int32_t sequence_length = 1)
      : data_(std::move(data)),
        batch_size_(batch_size),
        sequence_length_(sequence_length) {}

  absl::StatusOr<DataBatch> Next() override {
    ++next_calls;
    return DataBatch{.data = data_,
                     .batch_size = batch_size_,
                     .sequence_length = sequence_length_};
  }

  absl::Status Reset() override {
    ++reset_calls;
    return absl::OkStatus();
  }

  int next_calls = 0;
  int reset_calls = 0;

 private:
  Buffer data_;
  int32_t batch_size_;
  int32_t sequence_length_;
};

// These batches already contain their loss values, so this objective isolates
// Evaluate's accumulation and normalization from model/loss kernel behavior.
class LossValuesObjective final : public TrainingObjective {
 public:
  absl::StatusOr<ObjectiveForwardPass> Forward(
      cuda::Executor& executor, const DataBatch& batch) const override {
    auto count = batch.token_count();
    if (!count.ok())
      return count.status();
    return ObjectiveForwardPass{.loss = batch.data,
                                .normalization_count = *count};
  }

  absl::Status Backward(cuda::Executor& executor,
                        ObjectiveForwardPass pass) override {
    return absl::OkStatus();
  }
};

class VaryingBatchDataSetIterator final : public DataSetIterator {
 public:
  explicit VaryingBatchDataSetIterator(std::vector<DataBatch> batches)
      : batches_(std::move(batches)) {}

  absl::StatusOr<DataBatch> Next() override {
    return batches_[next_++ % batches_.size()];
  }

  absl::Status Reset() override {
    next_ = 0;
    return absl::OkStatus();
  }

 private:
  std::vector<DataBatch> batches_;
  size_t next_ = 0;
};

absl::StatusOr<float> ReadEvaluationLoss(cuda::Executor& executor,
                                         const Buffer& loss) {
  if (&loss.executor() != &executor || loss.size_bytes() != sizeof(float)) {
    return absl::InvalidArgumentError(
        "evaluation result must be one FP32 value on the test executor");
  }
  auto host_loss = cuda::PageLockedHostArray<float>::Allocate(executor, 1);
  if (!host_loss.ok())
    return host_loss.status();
  const cudaError_t error =
      cudaMemcpyAsync(host_loss->data(), loss.data(), loss.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream());
  if (error != cudaSuccess)
    return cuda::CudaStatus(error, "cudaMemcpyAsync(test evaluation result)");
  const absl::Status synchronized = executor.Synchronize();
  if (!synchronized.ok())
    return synchronized;
  return (*host_loss)[0];
}

class TrainerTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
    auto corpus = cuda::PageLockedHostArray<int>::Allocate(*executor_, 32);
    ASSERT_TRUE(corpus.ok()) << corpus.status();
    corpus_ = *corpus;
    std::iota(corpus_.begin(), corpus_.end(), 0);
  }

  void TearDown() override {
    if (executor_ == nullptr)
      return;
    corpus_ = {};
    EXPECT_TRUE(executor_->Synchronize().ok());
    executor_.reset();
  }

  absl::StatusOr<std::unique_ptr<InMemoryDataSetIterator>> MakeData() {
    return InMemoryDataSetIterator::Create(
        *executor_, corpus_,
        InMemoryDataSetOptions{
            .batch_size = 1,
            .context_length = 4,
            .order = InMemoryDataSetOrder::kSequential,
        });
  }

  absl::StatusOr<std::unique_ptr<FakeLoss>> MakeLoss() {
    auto losses = cuda::PageLockedHostArray<float>::Allocate(*executor_, 4);
    if (!losses.ok())
      return losses.status();
    std::iota(losses->begin(), losses->end(), 1.0f);
    return FakeLoss::Create(*executor_, losses->span());
  }

  std::unique_ptr<cuda::Executor> executor_;
  cuda::PageLockedHostArray<int> corpus_;
};

TEST_F(TrainerTest, EvaluateReturnsDeviceMeanAndResetsDataset) {
  FakeModel model;
  auto loss = MakeLoss();
  auto data = MakeData();
  ASSERT_TRUE(loss.ok()) << loss.status();
  ASSERT_TRUE(data.ok()) << data.status();
  LanguageModelingObjective objective(model, **loss);

  auto mean =
      Evaluate(*executor_, objective, **data, EvaluationOptions{.batches = 2});
  ASSERT_TRUE(mean.ok()) << mean.status();
  EXPECT_EQ(&mean->executor(), executor_.get());
  EXPECT_EQ(mean->size_bytes(), sizeof(float));
  auto host_mean = ReadEvaluationLoss(*executor_, *mean);
  ASSERT_TRUE(host_mean.ok()) << host_mean.status();
  EXPECT_FLOAT_EQ(*host_mean, 2.5f);
  EXPECT_EQ(model.forward_calls, 2);
  EXPECT_EQ((*loss)->forward_calls, 2);

  auto repeated =
      Evaluate(*executor_, objective, **data, EvaluationOptions{.batches = 2});
  ASSERT_TRUE(repeated.ok()) << repeated.status();
  auto repeated_host_mean = ReadEvaluationLoss(*executor_, *repeated);
  ASSERT_TRUE(repeated_host_mean.ok()) << repeated_host_mean.status();
  EXPECT_FLOAT_EQ(*repeated_host_mean, *host_mean);
}

TEST_F(TrainerTest, LanguageModelingNormalizesByTokensNotSequences) {
  FakeModel model;
  auto loss = MakeLoss();
  auto data = MakeData();
  ASSERT_TRUE(loss.ok()) << loss.status();
  ASSERT_TRUE(data.ok()) << data.status();
  auto batch = (*data)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  EXPECT_EQ(batch->batch_size, 1);
  EXPECT_EQ(batch->sequence_length, 4);
  LanguageModelingObjective objective(model, **loss);
  auto pass = objective.Forward(*executor_, *batch);
  ASSERT_TRUE(pass.ok()) << pass.status();
  EXPECT_EQ(pass->normalization_count, 4);
  EXPECT_EQ(pass->loss.size_bytes(), 4 * sizeof(float));

  // The buffer bytes alone are not sufficient: reject inconsistent or
  // overflowing sample metadata before calling the model.
  batch->sequence_length = 3;
  EXPECT_FALSE(objective.Forward(*executor_, *batch).ok());
  batch->sequence_length = std::numeric_limits<int>::max();
  batch->batch_size = 2;
  EXPECT_FALSE(objective.Forward(*executor_, *batch).ok());
  batch->batch_size = 0;
  EXPECT_FALSE(objective.Forward(*executor_, *batch).ok());
  EXPECT_EQ(model.forward_calls, 1);
}

TEST_F(TrainerTest,
       ChecksSequenceBoundariesBeforeFlatteningLanguageModelBatches) {
  auto embedding =
      EmbeddingLookupLayer::Create(*executor_, 32, 16, DataType::FP16);
  auto positions =
      PositionEmbeddingLayer::Create(*executor_, 4, 16, DataType::FP16);
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  ASSERT_TRUE(positions.ok()) << positions.status();
  ComposedLayerBuilder builder;
  ASSERT_TRUE(builder.add(std::move(*embedding)).ok());
  ASSERT_TRUE(
      builder.add(std::make_unique<ResidualLayer>(std::move(*positions))).ok());
  auto model = builder.create();
  auto data = MakeData();
  auto loss = MakeLoss();
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_TRUE(data.ok()) << data.status();
  ASSERT_TRUE(loss.ok()) << loss.status();
  auto batch = (*data)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  LanguageModelingObjective objective(**model, **loss);

  // Both shapes have four token rows. The model nevertheless must not treat
  // two independent two-token samples as one four-token position sequence.
  batch->batch_size = 2;
  batch->sequence_length = 2;
  auto invalid = objective.Forward(*executor_, *batch);
  EXPECT_EQ(invalid.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ((*loss)->forward_calls, 0);

  batch->batch_size = 1;
  batch->sequence_length = 4;
  auto valid = objective.Forward(*executor_, *batch);
  ASSERT_TRUE(valid.ok()) << valid.status();
  EXPECT_EQ(valid->normalization_count, 4);
  EXPECT_EQ((*loss)->forward_calls, 1);
}

TEST_F(TrainerTest, SequenceValidationChecksNestedAttentionAndPerTokenLayers) {
  FakeModel per_token;
  EXPECT_TRUE(per_token.ValidateSequenceLength(1).ok());
  EXPECT_TRUE(per_token.ValidateSequenceLength(1024).ok());
  EXPECT_FALSE(per_token.ValidateSequenceLength(0).ok());
  EXPECT_FALSE(per_token.ValidateSequenceLength(-1).ok());

  auto attention = AttentionLayer::Create(*executor_, 4, 1, 16, DataType::FP16);
  ASSERT_TRUE(attention.ok()) << attention.status();
  EXPECT_TRUE((*attention)->ValidateSequenceLength(4).ok());
  EXPECT_FALSE((*attention)->ValidateSequenceLength(2).ok());
  EXPECT_FALSE((*attention)->ValidateSequenceLength(8).ok());
  EXPECT_FALSE((*attention)->ValidateSequenceLength(0).ok());

  // Validation must reach a stateful descendant even through both sequential
  // and residual wrappers; checking only a composition's outer type misses it.
  ComposedLayerBuilder builder;
  ASSERT_TRUE(builder.add(std::make_unique<FakeModel>()).ok());
  ASSERT_TRUE(
      builder.add(std::make_unique<ResidualLayer>(std::move(*attention))).ok());
  auto model = builder.create();
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_TRUE((*model)->ValidateSequenceLength(4).ok());
  EXPECT_FALSE((*model)->ValidateSequenceLength(2).ok());
  EXPECT_FALSE((*model)->ValidateSequenceLength(8).ok());
  EXPECT_FALSE((*model)->ValidateSequenceLength(-1).ok());
}

TEST_F(TrainerTest, EvaluateWeightsUnequalBatchesByTheirTokenCounts) {
  auto first_host = cuda::PageLockedHostArray<float>::CopyFrom(
      *executor_, std::vector<float>{2.0f, 4.0f});
  auto second_host = cuda::PageLockedHostArray<float>::CopyFrom(
      *executor_, std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
  ASSERT_TRUE(first_host.ok()) << first_host.status();
  ASSERT_TRUE(second_host.ok()) << second_host.status();
  auto first = Buffer::Allocate(*executor_, first_host->size_bytes());
  auto second = Buffer::Allocate(*executor_, second_host->size_bytes());
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  ASSERT_EQ(cudaMemcpyAsync(first->data(), first_host->data(),
                            first_host->size_bytes(), cudaMemcpyHostToDevice,
                            executor_->stream()),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(second->data(), second_host->data(),
                            second_host->size_bytes(), cudaMemcpyHostToDevice,
                            executor_->stream()),
            cudaSuccess);
  VaryingBatchDataSetIterator data(
      {{.data = *first, .batch_size = 1, .sequence_length = 2},
       {.data = *second, .batch_size = 2, .sequence_length = 3}});
  LossValuesObjective objective;
  auto mean =
      Evaluate(*executor_, objective, data, EvaluationOptions{.batches = 2});
  ASSERT_TRUE(mean.ok()) << mean.status();
  auto host_mean = ReadEvaluationLoss(*executor_, *mean);
  ASSERT_TRUE(host_mean.ok()) << host_mean.status();
  // Sum the eight token losses, not the two batch means or three samples.
  EXPECT_FLOAT_EQ(*host_mean, 1.5f);
}

TEST_F(TrainerTest, EvaluateRejectsADatasetFromAnotherExecutor) {
  FakeModel model;
  auto loss = MakeLoss();
  auto data = MakeData();
  auto other_executor = cuda::Executor::Create();
  ASSERT_TRUE(loss.ok()) << loss.status();
  ASSERT_TRUE(data.ok()) << data.status();
  ASSERT_TRUE(other_executor.ok()) << other_executor.status();
  LanguageModelingObjective objective(model, **loss);

  const auto mean = Evaluate(**other_executor, objective, **data,
                             EvaluationOptions{.batches = 1});
  EXPECT_FALSE(mean.ok());
  EXPECT_EQ(mean.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(TrainerTest, TrainRunsForwardBackwardAndOptimizerSteps) {
  FakeModel model;
  FakeOptimizer optimizer;
  auto loss = MakeLoss();
  auto data = MakeData();
  ASSERT_TRUE(loss.ok()) << loss.status();
  ASSERT_TRUE(data.ok()) << data.status();
  LanguageModelingObjective objective(model, **loss);

  auto result = Train(*executor_, objective, optimizer, **data,
                      TrainingOptions{.max_steps = 3});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->steps_completed, 3);
  EXPECT_FALSE(result->reached_stop_loss);
  EXPECT_FALSE(result->reached_time_limit);
  EXPECT_EQ(model.forward_calls, 3);
  EXPECT_EQ(model.backward_calls, 3);
  EXPECT_EQ((*loss)->forward_calls, 3);
  EXPECT_EQ((*loss)->backward_calls, 3);
  EXPECT_EQ(optimizer.zero_grad_calls, 1);
  EXPECT_EQ(optimizer.steps, 3);
}

TEST_F(TrainerTest, TrainsAndEvaluatesSparseAutoEncoderDataBatches) {
  constexpr int kRows = 16;
  constexpr int kInputDimension = 16;
  constexpr int kFeatureDimension = 32;
  auto host_activations = cuda::PageLockedHostArray<float>::Allocate(
      *executor_, kRows * kInputDimension);
  ASSERT_TRUE(host_activations.ok()) << host_activations.status();
  for (size_t index = 0; index < host_activations->size(); ++index) {
    (*host_activations)[index] =
        static_cast<float>(static_cast<int>(index % 13) - 6) / 8.0f;
  }
  auto activations =
      Buffer::Allocate(*executor_, host_activations->size_bytes());
  ASSERT_TRUE(activations.ok()) << activations.status();
  ASSERT_EQ(cudaMemcpyAsync(activations->data(), host_activations->data(),
                            activations->size_bytes(), cudaMemcpyHostToDevice,
                            executor_->stream()),
            cudaSuccess);

  auto model = SparseAutoEncoderLayer::Create(
      *executor_, kInputDimension, kFeatureDimension, DataType::FP16);
  auto loss = SparseAutoEncoderLossLayer::Create(
      *executor_, kInputDimension, kFeatureDimension, 0.5f, DataType::FP16);
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_TRUE(loss.ok()) << loss.status();
  ASSERT_TRUE((*model)->InitializeNormal(0.05f, 19).ok());
  FixedActivationDataSetIterator data(*activations, 2, kRows / 2);
  SparseAutoEncoderObjective objective(**model, **loss);

  auto initial =
      Evaluate(*executor_, objective, data, EvaluationOptions{.batches = 1});
  ASSERT_TRUE(initial.ok()) << initial.status();
  auto host_initial = ReadEvaluationLoss(*executor_, *initial);
  ASSERT_TRUE(host_initial.ok()) << host_initial.status();
  EXPECT_TRUE(std::isfinite(*host_initial));
  EXPECT_GE(*host_initial, 0.0);

  auto batch = data.Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  auto pass = objective.Forward(*executor_, *batch);
  ASSERT_TRUE(pass.ok()) << pass.status();
  EXPECT_EQ(pass->normalization_count, kRows);
  // Grouping the same activation rows into samples cannot change per-token
  // SAE loss. Both single-token samples and longer sequences are supported.
  FixedActivationDataSetIterator single_token_samples(*activations, kRows);
  auto ungrouped = Evaluate(*executor_, objective, single_token_samples,
                            EvaluationOptions{.batches = 1});
  ASSERT_TRUE(ungrouped.ok()) << ungrouped.status();
  auto host_ungrouped = ReadEvaluationLoss(*executor_, *ungrouped);
  ASSERT_TRUE(host_ungrouped.ok()) << host_ungrouped.status();
  EXPECT_FLOAT_EQ(*host_ungrouped, *host_initial);
  batch->sequence_length = kRows;
  EXPECT_FALSE(objective.Forward(*executor_, *batch).ok());
  batch->sequence_length = -1;
  EXPECT_FALSE(objective.Forward(*executor_, *batch).ok());

  FakeOptimizer optimizer;
  std::vector<int> evaluation_steps;
  auto result = Train(*executor_, objective, optimizer, data,
                      TrainingOptions{
                          .max_steps = 2,
                          .evaluation_interval = 1,
                          .evaluation_batches = 1,
                          .evaluation_callback =
                              [&](int step, double value) {
                                evaluation_steps.push_back(step);
                                EXPECT_TRUE(std::isfinite(value));
                                EXPECT_GE(value, 0.0);
                              },
                      });
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->steps_completed, 2);
  EXPECT_EQ(optimizer.zero_grad_calls, 1);
  EXPECT_EQ(optimizer.steps, 2);
  EXPECT_EQ(evaluation_steps, (std::vector<int>{1, 2}));
}

TEST_F(TrainerTest, StopsBeforeFirstUpdateWhenInitialEvaluationQualifies) {
  FakeModel model;
  FakeOptimizer optimizer;
  std::vector<int> evaluation_steps;
  std::vector<double> evaluation_losses;
  auto loss = MakeLoss();
  auto training_data = MakeData();
  auto evaluation_data = MakeData();
  ASSERT_TRUE(loss.ok()) << loss.status();
  ASSERT_TRUE(training_data.ok()) << training_data.status();
  ASSERT_TRUE(evaluation_data.ok()) << evaluation_data.status();
  LanguageModelingObjective objective(model, **loss);

  auto result =
      Train(*executor_, objective, optimizer, **training_data,
            TrainingOptions{
                .max_steps = 3,
                .initial_step = 570,
                .evaluation_interval = 1,
                .evaluation_batches = 1,
                .stop_loss = 2.5,
                .evaluation_tokens = evaluation_data->get(),
                .evaluation_callback = [&](int steps_completed, double loss) {
                  evaluation_steps.push_back(steps_completed);
                  evaluation_losses.push_back(loss);
                }});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->steps_completed, 570);
  EXPECT_TRUE(result->reached_stop_loss);
  EXPECT_EQ(optimizer.steps, 0);
  EXPECT_EQ(evaluation_steps, (std::vector<int>{570}));
  EXPECT_EQ(evaluation_losses, (std::vector<double>{2.5}));
}

TEST_F(TrainerTest, StopsAfterUpdateWhenPeriodicEvaluationQualifies) {
  FakeModel model;
  FakeOptimizer optimizer;
  std::vector<int> evaluation_steps;
  std::vector<double> evaluation_losses;
  auto loss = MakeLoss();
  auto training_data = MakeData();
  auto evaluation_data = MakeData();
  ASSERT_TRUE(loss.ok()) << loss.status();
  ASSERT_TRUE(training_data.ok()) << training_data.status();
  ASSERT_TRUE(evaluation_data.ok()) << evaluation_data.status();
  LanguageModelingObjective objective(model, **loss);

  auto result =
      Train(*executor_, objective, optimizer, **training_data,
            TrainingOptions{
                .max_steps = 3,
                .evaluation_interval = 1,
                .evaluation_batches = 1,
                .stop_loss = 2.5,
                .evaluation_tokens = evaluation_data->get(),
                .initial_loss = 3.0,
                .evaluation_callback = [&](int steps_completed, double loss) {
                  evaluation_steps.push_back(steps_completed);
                  evaluation_losses.push_back(loss);
                }});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->steps_completed, 1);
  EXPECT_TRUE(result->reached_stop_loss);
  EXPECT_EQ(optimizer.steps, 1);
  EXPECT_EQ(model.backward_calls, 1);
  EXPECT_EQ(evaluation_steps, (std::vector<int>{1}));
  EXPECT_EQ(evaluation_losses, (std::vector<double>{2.5}));
}

TEST_F(TrainerTest, CallbackEnablesPeriodicEvaluationWithoutEarlyStopping) {
  FakeModel model;
  FakeOptimizer optimizer;
  std::vector<int> evaluation_steps;
  std::vector<double> evaluation_losses;
  auto loss = MakeLoss();
  auto training_data = MakeData();
  auto evaluation_data = MakeData();
  ASSERT_TRUE(loss.ok()) << loss.status();
  ASSERT_TRUE(training_data.ok()) << training_data.status();
  ASSERT_TRUE(evaluation_data.ok()) << evaluation_data.status();
  LanguageModelingObjective objective(model, **loss);

  auto result =
      Train(*executor_, objective, optimizer, **training_data,
            TrainingOptions{
                .max_steps = 3,
                .evaluation_interval = 2,
                .evaluation_batches = 1,
                .evaluation_tokens = evaluation_data->get(),
                .evaluation_callback = [&](int steps_completed, double loss) {
                  evaluation_steps.push_back(steps_completed);
                  evaluation_losses.push_back(loss);
                }});

  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->steps_completed, 3);
  EXPECT_FALSE(result->reached_stop_loss);
  EXPECT_EQ(optimizer.steps, 3);
  EXPECT_EQ(evaluation_steps, (std::vector<int>{2, 3}));
  EXPECT_EQ(evaluation_losses, (std::vector<double>{2.5, 2.5}));
}

TEST_F(TrainerTest, ResumedRunUsesAbsoluteStepNumbers) {
  FakeModel model;
  FakeOptimizer optimizer;
  std::vector<int> step_callbacks;
  std::vector<int> evaluation_steps;
  auto loss = MakeLoss();
  auto training_data = MakeData();
  auto evaluation_data = MakeData();
  ASSERT_TRUE(loss.ok()) << loss.status();
  ASSERT_TRUE(training_data.ok()) << training_data.status();
  ASSERT_TRUE(evaluation_data.ok()) << evaluation_data.status();
  LanguageModelingObjective objective(model, **loss);

  auto result =
      Train(*executor_, objective, optimizer, **training_data,
            TrainingOptions{.max_steps = 3,
                            .initial_step = 570,
                            .evaluation_interval = 2,
                            .evaluation_batches = 1,
                            .evaluation_tokens = evaluation_data->get(),
                            .evaluation_callback =
                                [&](int steps_completed, double) {
                                  evaluation_steps.push_back(steps_completed);
                                },
                            .step_callback =
                                [&](int steps_completed) {
                                  step_callbacks.push_back(steps_completed);
                                  return absl::OkStatus();
                                }});

  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->steps_completed, 573);
  EXPECT_EQ(optimizer.steps, 3);
  EXPECT_EQ(step_callbacks, (std::vector<int>{571, 572, 573}));
  EXPECT_EQ(evaluation_steps, (std::vector<int>{572, 573}));
}

TEST_F(TrainerTest, StepCallbackRunsAfterUpdatesAndPropagatesErrors) {
  FakeModel model;
  FakeOptimizer optimizer;
  std::vector<int> callback_steps;
  auto loss = MakeLoss();
  auto data = MakeData();
  ASSERT_TRUE(loss.ok()) << loss.status();
  ASSERT_TRUE(data.ok()) << data.status();
  LanguageModelingObjective objective(model, **loss);

  auto result =
      Train(*executor_, objective, optimizer, **data,
            TrainingOptions{
                .max_steps = 3,
                .initial_step = 570,
                .step_callback = [&](int steps_completed) -> absl::Status {
                  callback_steps.push_back(steps_completed);
                  if (steps_completed == 572)
                    return absl::UnavailableError("checkpoint failed");
                  return absl::OkStatus();
                }});

  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnavailable);
  EXPECT_EQ(result.status().message(), "checkpoint failed");
  EXPECT_EQ(callback_steps, (std::vector<int>{571, 572}));
  EXPECT_EQ(optimizer.steps, 2);
  EXPECT_EQ(model.backward_calls, 2);
}

TEST_F(TrainerTest, UnlimitedTrainingRunsUntilCallbackStopsIt) {
  FakeModel model;
  FakeOptimizer optimizer;
  auto loss = MakeLoss();
  auto data = MakeData();
  ASSERT_TRUE(loss.ok()) << loss.status();
  ASSERT_TRUE(data.ok()) << data.status();
  LanguageModelingObjective objective(model, **loss);

  auto result = Train(
      *executor_, objective, optimizer, **data,
      TrainingOptions{.max_steps = kUnlimitedTrainingSteps,
                      .step_callback = [](int steps_completed) -> absl::Status {
                        if (steps_completed == 3)
                          return absl::CancelledError("test requested stop");
                        return absl::OkStatus();
                      }});

  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kCancelled);
  EXPECT_EQ(optimizer.steps, 3);
  EXPECT_EQ(model.backward_calls, 3);
}

TEST_F(TrainerTest, TimeLimitWaitsForStreamWorkAndEvaluatesTheFinalStep) {
  FakeModel model;
  FakeOptimizer optimizer;
  auto loss = MakeLoss();
  auto training_data = MakeData();
  auto evaluation_data = MakeData();
  ASSERT_TRUE(loss.ok()) << loss.status();
  ASSERT_TRUE(training_data.ok()) << training_data.status();
  ASSERT_TRUE(evaluation_data.ok()) << evaluation_data.status();
  LanguageModelingObjective objective(model, **loss);

  // Delay stream completion without delaying the host's launch. A deadline
  // check that only times queued work would call step_callback too early.
  std::atomic<bool> update_completed = false;
  optimizer.step_action = [&]() {
    return cuda::CudaStatus(
        cudaLaunchHostFunc(
            executor_->stream(),
            [](void* state) {
              std::this_thread::sleep_for(std::chrono::milliseconds(40));
              static_cast<std::atomic<bool>*>(state)->store(true);
            },
            &update_completed),
        "cudaLaunchHostFunc(test optimizer completion)");
  };
  std::vector<int> evaluation_steps;
  auto result =
      Train(*executor_, objective, optimizer, **training_data,
            TrainingOptions{.max_steps = kUnlimitedTrainingSteps,
                            .initial_step = 570,
                            .training_seconds = 0.01,
                            .evaluation_interval = 100,
                            .evaluation_tokens = evaluation_data->get(),
                            .evaluation_callback =
                                [&](int step, double value) {
                                  evaluation_steps.push_back(step);
                                  EXPECT_DOUBLE_EQ(value, 2.5);
                                },
                            .step_callback =
                                [&](int step) {
                                  EXPECT_TRUE(update_completed.load());
                                  EXPECT_EQ(step, 571);
                                  return absl::OkStatus();
                                }});
  // Always finish the callback before its stack-owned state is destroyed,
  // including when the training call unexpectedly returns an error.
  ASSERT_TRUE(executor_->Synchronize().ok());
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->steps_completed, 571);
  EXPECT_TRUE(result->reached_time_limit);
  EXPECT_FALSE(result->reached_stop_loss);
  EXPECT_GE(result->elapsed_training_seconds, 0.04);
  EXPECT_EQ(optimizer.steps, 1);
  EXPECT_EQ(model.backward_calls, 1);
  EXPECT_EQ(evaluation_steps, (std::vector<int>{571}));
}

TEST_F(TrainerTest, TinyBudgetCompletesOneRealOptimizerUpdate) {
  constexpr int kRows = 16;
  constexpr int kInputDimension = 16;
  constexpr int kFeatureDimension = 32;
  auto activations =
      Buffer::Allocate(*executor_, kRows * kInputDimension * sizeof(float));
  ASSERT_TRUE(activations.ok()) << activations.status();
  ASSERT_EQ(cudaMemsetAsync(activations->data(), 0, activations->size_bytes(),
                            executor_->stream()),
            cudaSuccess);
  auto model = SparseAutoEncoderLayer::Create(
      *executor_, kInputDimension, kFeatureDimension, DataType::FP16);
  auto loss = SparseAutoEncoderLossLayer::Create(
      *executor_, kInputDimension, kFeatureDimension, 0.5f, DataType::FP16);
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_TRUE(loss.ok()) << loss.status();
  ASSERT_TRUE((*model)->InitializeNormal(0.05f, 19).ok());
  auto optimizer = Optimizer::Create(*executor_, **model, AdamWConfig{});
  ASSERT_TRUE(optimizer.ok()) << optimizer.status();
  FixedActivationDataSetIterator data(*activations, kRows);
  SparseAutoEncoderObjective objective(**model, **loss);
  std::vector<int> evaluation_steps;

  auto result =
      Train(*executor_, objective, **optimizer, data,
            TrainingOptions{.max_steps = kUnlimitedTrainingSteps,
                            .training_seconds = 1e-9,
                            .evaluation_interval = 100,
                            .evaluation_callback = [&](int step, double value) {
                              evaluation_steps.push_back(step);
                              EXPECT_TRUE(std::isfinite(value));
                            }});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->steps_completed, 1);
  EXPECT_EQ((*optimizer)->step(), 1);
  EXPECT_EQ(cudaStreamQuery(executor_->stream()), cudaSuccess);
  EXPECT_TRUE(result->reached_time_limit);
  EXPECT_FALSE(result->reached_stop_loss);
  EXPECT_GE(result->elapsed_training_seconds, 1e-9);
  EXPECT_EQ(evaluation_steps, (std::vector<int>{1}));
}

TEST_F(TrainerTest, StepCapCanStopBeforeTimeBudget) {
  FakeModel model;
  FakeOptimizer optimizer;
  auto loss = MakeLoss();
  auto data = MakeData();
  ASSERT_TRUE(loss.ok()) << loss.status();
  ASSERT_TRUE(data.ok()) << data.status();
  LanguageModelingObjective objective(model, **loss);
  auto result =
      Train(*executor_, objective, optimizer, **data,
            TrainingOptions{.max_steps = 1, .training_seconds = 3600.0});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->steps_completed, 1);
  EXPECT_FALSE(result->reached_time_limit);
  EXPECT_GT(result->elapsed_training_seconds, 0.0);
}

TEST_F(TrainerTest, RejectsInvalidOptions) {
  FakeModel model;
  FakeOptimizer optimizer;
  auto loss = MakeLoss();
  auto data = MakeData();
  ASSERT_TRUE(loss.ok()) << loss.status();
  ASSERT_TRUE(data.ok()) << data.status();
  LanguageModelingObjective objective(model, **loss);

  for (double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_EQ(
        Train(*executor_, objective, optimizer, **data,
              TrainingOptions{.max_steps = 1, .training_seconds = invalid})
            .status()
            .code(),
        absl::StatusCode::kInvalidArgument);
  }
  EXPECT_FALSE(
      Evaluate(*executor_, objective, **data, EvaluationOptions{.batches = 0})
          .ok());
  EXPECT_FALSE(Train(*executor_, objective, optimizer, **data,
                     TrainingOptions{.max_steps = -2})
                   .ok());
  EXPECT_FALSE(Train(*executor_, objective, optimizer, **data,
                     TrainingOptions{.initial_step = -1})
                   .ok());
  EXPECT_FALSE(
      Train(*executor_, objective, optimizer, **data,
            TrainingOptions{.max_steps = 1,
                            .initial_step = std::numeric_limits<int>::max()})
          .ok());
}

}  // namespace
}  // namespace pluto::llm
