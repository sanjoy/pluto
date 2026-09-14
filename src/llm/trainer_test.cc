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
#include <type_traits>
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
#include "src/llm/adamw_optimizer.h"
#include "src/llm/layer.h"
#include "src/llm/layers/attention.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/embedding.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/sparse_autoencoder.h"
#include "src/llm/optimizer.h"

namespace pluto::llm {
namespace {

static_assert(!std::is_default_constructible_v<EvaluationOptions>);
static_assert(!std::is_default_constructible_v<TrainingOptions>);

const ActivationType kTokenType{DataType::INT32,
                                {ActivationType::kBatchDimension, 4}};
const ActivationType kFloatType{DataType::FP32,
                                {ActivationType::kBatchDimension, 4}};

class FakeModel final : public Layer {
 public:
  explicit FakeModel(DataType dtype = DataType::INT32, int sequence_length = 4,
                     size_t input_count = 1)
      : output_type_(dtype, {ActivationType::kBatchDimension, sequence_length}),
        input_types_(input_count, output_type_) {}

  absl::string_view name() const override { return "FakeModel"; }

  absl::Span<const ActivationType> input_types() const override {
    return input_types_;
  }
  absl::Span<const ActivationType> output_types() const override {
    return {&output_type_, 1};
  }

  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return DataType::FP16; }

  mutable int forward_calls = 0;
  int backward_calls = 0;

 private:
  const ActivationType output_type_;
  const std::vector<ActivationType> input_types_;

  absl::StatusOr<FwdResult> fwd_impl(
      cuda::Executor& executor,
      absl::Span<const Buffer> inputs) const override {
    BackwardState state;
    ++forward_calls;
    return FwdResult{{inputs[0]}, std::move(state)};
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
  absl::string_view name() const override { return "FakeLoss"; }

  static absl::StatusOr<std::unique_ptr<FakeLoss>> Create(
      cuda::Executor& executor, absl::Span<const float> losses,
      ActivationType model_output = {DataType::INT32,
                                     {ActivationType::kBatchDimension, 4}},
      ActivationType target = {DataType::INT32,
                               {ActivationType::kBatchDimension, 4}}) {
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
        new FakeLoss(std::move(*device_losses), std::move(*gradient),
                     std::move(model_output), std::move(target)));
  }

  absl::Span<const ActivationType> input_types() const override {
    return input_types_;
  }
  absl::Span<const ActivationType> output_types() const override {
    return output_types_;
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
    return FwdResult{{losses_}, std::move(state)};
  }

  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> output_gradients,
                                     BackwardState state) override {
    ++backward_calls;
    return BufferVec{gradient_};
  }

  FakeLoss(Buffer losses, Buffer gradient, ActivationType model_output,
           ActivationType target)
      : input_types_{std::move(model_output), target},
        output_types_{
            {DataType::FP32,
             {ActivationType::kBatchDimension, target.dimensions()[1]}}},
        losses_(std::move(losses)),
        gradient_(std::move(gradient)) {}

  const ActivationType input_types_[2];
  const ActivationType output_types_[1];
  Buffer losses_;
  Buffer gradient_;
};

// A deliberately non-SAE multi-output layer: routing must depend only on
// the public vector contract, not on concrete layer type or saved-state layout.
class RoutingLayer final : public Layer {
 public:
  absl::string_view name() const override { return "RoutingLayer"; }

  RoutingLayer(BufferVec outputs, BufferVec gradients,
               std::vector<ActivationType> input_types,
               std::vector<ActivationType> output_types)
      : input_types_(std::move(input_types)),
        output_types_(std::move(output_types)),
        outputs_(std::move(outputs)),
        gradients_(std::move(gradients)) {
    // Signatures are independent of returned buffers/gradients, so malformed
    // output fixtures still exercise runtime validation, not just preflight.
  }

  absl::Span<const ActivationType> input_types() const override {
    return input_types_;
  }
  absl::Span<const ActivationType> output_types() const override {
    return output_types_;
  }
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return DataType::FP16; }

  mutable int forward_calls = 0;
  mutable BufferVec observed_inputs;
  BufferVec observed_gradients;
  int backward_calls = 0;

 private:
  absl::StatusOr<FwdResult> fwd_impl(
      cuda::Executor&, absl::Span<const Buffer> inputs) const override {
    ++forward_calls;
    observed_inputs.assign(inputs.begin(), inputs.end());
    return FwdResult{outputs_, {}};
  }

  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&,
                                     absl::Span<const Buffer> gradients,
                                     BackwardState) override {
    ++backward_calls;
    observed_gradients.assign(gradients.begin(), gradients.end());
    return gradients_;
  }

  std::vector<ActivationType> input_types_;
  std::vector<ActivationType> output_types_;
  BufferVec outputs_;
  BufferVec gradients_;
};

class FakeOptimizer final : public Optimizer {
 public:
  absl::Status ZeroGrad() override {
    ++zero_grad_calls;
    return absl::OkStatus();
  }
  absl::Status ApplyStep() override {
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
    return DataBatch{.inputs = data_,
                     .targets = data_,
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

// Evaluate one explicit batch through the public model/loss interface.
absl::StatusOr<Buffer> EvaluateBatch(cuda::Executor& executor,
                                     const Layer& model, const Layer& loss,
                                     const DataBatch& batch) {
  VaryingBatchDataSetIterator data({batch});
  const EvaluationOptions options{
      .loss_layer = loss, .eval_data = data, .batches = 1};
  return Evaluate(executor, model, options);
}

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

  absl::StatusOr<std::unique_ptr<FakeLoss>> MakeLoss(
      ActivationType model_output = {DataType::INT32,
                                     {ActivationType::kBatchDimension, 4}}) {
    auto losses = cuda::PageLockedHostArray<float>::Allocate(*executor_, 4);
    if (!losses.ok())
      return losses.status();
    std::iota(losses->begin(), losses->end(), 1.0f);
    return FakeLoss::Create(*executor_, losses->span(),
                            std::move(model_output));
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

  auto mean =
      Evaluate(*executor_, model,
               EvaluationOptions{
                   .loss_layer = **loss, .eval_data = **data, .batches = 2});
  ASSERT_TRUE(mean.ok()) << mean.status();
  EXPECT_EQ(&mean->executor(), executor_.get());
  EXPECT_EQ(mean->size_bytes(), sizeof(float));
  auto host_mean = ReadEvaluationLoss(*executor_, *mean);
  ASSERT_TRUE(host_mean.ok()) << host_mean.status();
  EXPECT_FLOAT_EQ(*host_mean, 2.5f);
  EXPECT_EQ(model.forward_calls, 2);
  EXPECT_EQ((*loss)->forward_calls, 2);

  auto repeated =
      Evaluate(*executor_, model,
               EvaluationOptions{
                   .loss_layer = **loss, .eval_data = **data, .batches = 2});
  ASSERT_TRUE(repeated.ok()) << repeated.status();
  auto repeated_host_mean = ReadEvaluationLoss(*executor_, *repeated);
  ASSERT_TRUE(repeated_host_mean.ok()) << repeated_host_mean.status();
  EXPECT_FLOAT_EQ(*repeated_host_mean, *host_mean);
}

TEST_F(TrainerTest,
       EvaluateCopiedConstOptionsPreserveAndUseDependencyReferences) {
  const FakeModel model;
  auto loss = MakeLoss();
  auto data = MakeData();
  ASSERT_TRUE(loss.ok()) << loss.status();
  ASSERT_TRUE(data.ok()) << data.status();
  auto batch = (*data)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  FixedActivationDataSetIterator eval_data(batch->inputs, batch->batch_size,
                                           batch->sequence_length);

  const EvaluationOptions options{
      .loss_layer = **loss, .eval_data = eval_data, .batches = 2};
  const EvaluationOptions copied_options = options;
  EXPECT_EQ(&copied_options.loss_layer, loss->get());
  EXPECT_EQ(&copied_options.eval_data, &eval_data);

  auto mean = Evaluate(*executor_, model, copied_options);
  ASSERT_TRUE(mean.ok()) << mean.status();
  EXPECT_EQ(&mean->executor(), executor_.get());
  EXPECT_EQ(mean->size_bytes(), sizeof(float));
  auto host_mean = ReadEvaluationLoss(*executor_, *mean);
  ASSERT_TRUE(host_mean.ok()) << host_mean.status();
  EXPECT_FLOAT_EQ(*host_mean, 2.5f);
  EXPECT_EQ(model.forward_calls, 2);
  EXPECT_EQ((*loss)->forward_calls, 2);
  EXPECT_EQ(eval_data.reset_calls, 1);
  EXPECT_EQ(eval_data.next_calls, 2);
  EXPECT_EQ(model.backward_calls, 0);
  EXPECT_EQ((*loss)->backward_calls, 0);
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
  auto pass = EvaluateBatch(*executor_, model, **loss, *batch);
  ASSERT_TRUE(pass.ok()) << pass.status();
  EXPECT_EQ(*batch->token_count(), 4);
  EXPECT_EQ(pass->size_bytes(), sizeof(float));

  // Inconsistent or overflowing metadata is rejected before model execution,
  // even when the underlying allocation is large enough for some other shape.
  batch->sequence_length = 3;
  EXPECT_FALSE(EvaluateBatch(*executor_, model, **loss, *batch).ok());
  batch->sequence_length = std::numeric_limits<int>::max();
  batch->batch_size = 2;
  EXPECT_FALSE(EvaluateBatch(*executor_, model, **loss, *batch).ok());
  batch->batch_size = 0;
  EXPECT_FALSE(EvaluateBatch(*executor_, model, **loss, *batch).ok());
  EXPECT_EQ(model.forward_calls, 1);
}

TEST_F(TrainerTest,
       ChecksSequenceBoundariesBeforeFlatteningLanguageModelBatches) {
  auto embedding =
      EmbeddingLookupLayer::Create(*executor_, 32, 16, DataType::FP16, 4);
  auto positions =
      PositionEmbeddingLayer::Create(*executor_, 4, 16, DataType::FP16);
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  ASSERT_TRUE(positions.ok()) << positions.status();
  ComposedLayerBuilder builder;
  ASSERT_TRUE(builder.add(std::move(*embedding)).ok());
  ASSERT_TRUE(builder.add(ResidualLayer::Create(std::move(*positions))).ok());
  auto model = builder.create();
  auto data = MakeData();
  auto loss =
      MakeLoss({DataType::FP32, {ActivationType::kBatchDimension, 4, 16}});
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_TRUE(data.ok()) << data.status();
  ASSERT_TRUE(loss.ok()) << loss.status();
  auto batch = (*data)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();

  // Both shapes have four token rows. The model nevertheless must not treat
  // two independent two-token samples as one four-token position sequence.
  batch->batch_size = 2;
  batch->sequence_length = 2;
  auto invalid = EvaluateBatch(*executor_, **model, **loss, *batch);
  EXPECT_EQ(invalid.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ((*loss)->forward_calls, 0);

  batch->batch_size = 1;
  batch->sequence_length = 4;
  auto valid = EvaluateBatch(*executor_, **model, **loss, *batch);
  ASSERT_TRUE(valid.ok()) << valid.status();
  EXPECT_EQ(*batch->token_count(), 4);
  EXPECT_EQ((*loss)->forward_calls, 1);
}

TEST_F(TrainerTest, SequenceValidationChecksNestedAttentionAndPerTokenLayers) {
  auto attention = AttentionLayer::Create(*executor_, 4, 1, 16, DataType::FP16);
  ASSERT_TRUE(attention.ok()) << attention.status();
  // A composition preserves the exact declared sample shape through both
  // sequential and residual wrappers, without a separate virtual validator.
  // Attention consumes packed QKV, so its projection must live inside the
  // residual branch for both ends of the branch to have model width 16.
  ComposedLayerBuilder branch;
  ASSERT_TRUE(branch
                  .add(FullyConnectedLayer::Create(*executor_, 16, 48,
                                                   DataType::FP16, 4))
                  .ok());
  ASSERT_TRUE(branch.add(std::move(*attention)).ok());
  auto attention_branch = branch.create();
  ASSERT_TRUE(attention_branch.ok()) << attention_branch.status();
  ComposedLayerBuilder builder;
  ASSERT_TRUE(
      builder.add(ResidualLayer::Create(std::move(*attention_branch))).ok());
  auto model = builder.create();
  ASSERT_TRUE(model.ok()) << model.status();
  // Dense projections require a full 16-row tile: four four-token samples.
  auto input = Buffer::Allocate(*executor_, 16 * 16 * sizeof(float));
  auto target = Buffer::Allocate(*executor_, 16 * sizeof(int));
  auto host_losses = cuda::PageLockedHostArray<float>::Allocate(*executor_, 16);
  ASSERT_TRUE(input.ok()) << input.status();
  ASSERT_TRUE(target.ok()) << target.status();
  ASSERT_TRUE(host_losses.ok()) << host_losses.status();
  std::iota(host_losses->begin(), host_losses->end(), 1.0f);
  ASSERT_EQ(cudaMemsetAsync(input->data(), 0, input->size_bytes(),
                            executor_->stream()),
            cudaSuccess);
  auto loss = FakeLoss::Create(
      *executor_, host_losses->span(),
      {DataType::FP32, {ActivationType::kBatchDimension, 4, 16}});
  ASSERT_TRUE(loss.ok()) << loss.status();
  DataBatch batch{.inputs = *input,
                  .targets = *target,
                  .batch_size = 4,
                  .sequence_length = 4};
  ASSERT_TRUE(EvaluateBatch(*executor_, **model, **loss, batch).ok());
  EXPECT_EQ((*loss)->forward_calls, 1);
  for (int sequence_length : {2, 8, 0, -1}) {
    batch.sequence_length = sequence_length;
    batch.batch_size = sequence_length > 0 ? 16 / sequence_length : 4;
    const auto invalid = EvaluateBatch(*executor_, **model, **loss, batch);
    EXPECT_EQ(invalid.status().code(), absl::StatusCode::kInvalidArgument);
  }
  EXPECT_EQ((*loss)->forward_calls, 1);
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
  VaryingBatchDataSetIterator data({{.inputs = *first,
                                     .targets = *first,
                                     .batch_size = 1,
                                     .sequence_length = 2},
                                    {.inputs = *second,
                                     .targets = *second,
                                     .batch_size = 3,
                                     .sequence_length = 2}});
  FakeModel model(DataType::FP32, 2);
  FakeModel loss(DataType::FP32, 2, 2);
  auto mean = Evaluate(
      *executor_, model,
      EvaluationOptions{.loss_layer = loss, .eval_data = data, .batches = 2});
  ASSERT_TRUE(mean.ok()) << mean.status();
  auto host_mean = ReadEvaluationLoss(*executor_, *mean);
  ASSERT_TRUE(host_mean.ok()) << host_mean.status();
  // Sum the eight token losses, not the two batch means or four samples.
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

  const auto mean =
      Evaluate(**other_executor, model,
               EvaluationOptions{
                   .loss_layer = **loss, .eval_data = **data, .batches = 1});
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

  auto result = Train(*executor_, model,
                      TrainingOptions{.loss_layer = **loss,
                                      .optimizer = optimizer,
                                      .training_data = **data,
                                      .max_steps = 3});
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

TEST_F(TrainerTest, CopiedConstOptionsPreserveAndUseDependencyReferences) {
  FakeModel model;
  FakeOptimizer optimizer;
  auto loss = MakeLoss();
  auto data = MakeData();
  ASSERT_TRUE(loss.ok()) << loss.status();
  ASSERT_TRUE(data.ok()) << data.status();
  auto batch = (*data)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  FixedActivationDataSetIterator training_data(batch->inputs, batch->batch_size,
                                               batch->sequence_length);

  const TrainingOptions options{.loss_layer = **loss,
                                .optimizer = optimizer,
                                .training_data = training_data,
                                .max_steps = 2};
  const TrainingOptions copied_options = options;
  EXPECT_EQ(&copied_options.loss_layer, loss->get());
  EXPECT_EQ(&copied_options.optimizer, &optimizer);
  EXPECT_EQ(&copied_options.training_data, &training_data);

  auto result = Train(*executor_, model, copied_options);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->steps_completed, 2);
  EXPECT_EQ((*loss)->forward_calls, 2);
  EXPECT_EQ((*loss)->backward_calls, 2);
  EXPECT_EQ(optimizer.zero_grad_calls, 1);
  EXPECT_EQ(optimizer.steps, 2);
  EXPECT_EQ(training_data.reset_calls, 1);
  EXPECT_EQ(training_data.next_calls, 2);
}

TEST_F(TrainerTest, RoutesExactlyOneGradientPerModelOutput) {
  auto data = MakeData();
  ASSERT_TRUE(data.ok()) << data.status();
  auto batch = (*data)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  BufferVec outputs;
  BufferVec gradients;
  for (int i = 0; i < 3; ++i) {
    auto buffer = Buffer::Allocate(*executor_, 4 * sizeof(float));
    ASSERT_TRUE(buffer.ok()) << buffer.status();
    gradients.push_back(*buffer);
    outputs.push_back(*buffer);
  }
  RoutingLayer model(outputs, {}, {kTokenType},
                     {kFloatType, kFloatType, kFloatType});
  // Targets remain a forward input, but have no corresponding gradient.
  RoutingLayer loss({gradients[0]}, gradients,
                    {kFloatType, kFloatType, kFloatType, kTokenType},
                    {kFloatType});
  FakeOptimizer optimizer;
  auto result = Train(*executor_, model,
                      TrainingOptions{.loss_layer = loss,
                                      .optimizer = optimizer,
                                      .training_data = **data,
                                      .max_steps = 1});
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(model.observed_inputs.size(), 1);
  EXPECT_EQ(model.observed_inputs[0].data(), batch->inputs.data());
  ASSERT_EQ(loss.observed_inputs.size(), 4);
  ASSERT_EQ(model.observed_gradients.size(), 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(loss.observed_inputs[i].data(), outputs[i].data());
    EXPECT_EQ(model.observed_gradients[i].data(), gradients[i].data());
  }
  EXPECT_EQ(loss.observed_inputs[3].data(), batch->targets.data());
  EXPECT_EQ(optimizer.steps, 1);
}

TEST_F(TrainerTest, RejectsLossGradientsThatDoNotMatchModelOutputs) {
  auto data = MakeData();
  auto buffer = Buffer::Allocate(*executor_, 4 * sizeof(float));
  ASSERT_TRUE(data.ok()) << data.status();
  ASSERT_TRUE(buffer.ok()) << buffer.status();
  // In particular, the old extra target gradient is now rejected.
  for (size_t count : {size_t{0}, size_t{1}, size_t{2}, size_t{4}, size_t{5}}) {
    RoutingLayer model({*buffer, *buffer, *buffer}, {}, {kTokenType},
                       {kFloatType, kFloatType, kFloatType});
    RoutingLayer loss({*buffer}, BufferVec(count, *buffer),
                      {kFloatType, kFloatType, kFloatType, kTokenType},
                      {kFloatType});
    FakeOptimizer optimizer;
    auto result = Train(*executor_, model,
                        TrainingOptions{.loss_layer = loss,
                                        .optimizer = optimizer,
                                        .training_data = **data,
                                        .max_steps = 1});
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(loss.backward_calls, 1);
    EXPECT_EQ(model.backward_calls, 0);
    EXPECT_EQ(optimizer.steps, 0);
  }
}

TEST_F(TrainerTest, RejectsMissingModelOutputsAndInvalidLossOutputs) {
  auto data = MakeData();
  auto buffer = Buffer::Allocate(*executor_, 4 * sizeof(float));
  auto short_loss = Buffer::Allocate(*executor_, sizeof(float));
  ASSERT_TRUE(data.ok()) << data.status();
  ASSERT_TRUE(buffer.ok()) << buffer.status();
  ASSERT_TRUE(short_loss.ok()) << short_loss.status();
  RoutingLayer no_outputs({}, {}, {kTokenType}, {kFloatType});
  RoutingLayer loss({*buffer}, {}, {kFloatType, kTokenType}, {kFloatType});
  auto invalid_model =
      Evaluate(*executor_, no_outputs,
               EvaluationOptions{.loss_layer = loss, .eval_data = **data});
  EXPECT_EQ(invalid_model.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(loss.observed_inputs.empty());

  RoutingLayer model({*buffer}, {}, {kTokenType}, {kFloatType});
  for (const BufferVec& outputs :
       {BufferVec{}, BufferVec{*buffer, *buffer}, BufferVec{*short_loss}}) {
    RoutingLayer invalid_loss(outputs, {}, {kFloatType, kTokenType},
                              {kFloatType});
    auto result = Evaluate(
        *executor_, model,
        EvaluationOptions{.loss_layer = invalid_loss, .eval_data = **data});
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  }
}

TEST_F(TrainerTest, RejectsTargetsFromAnotherExecutorBeforeForward) {
  auto data = MakeData();
  auto other_executor = cuda::Executor::Create();
  ASSERT_TRUE(data.ok()) << data.status();
  ASSERT_TRUE(other_executor.ok()) << other_executor.status();
  auto target = Buffer::Allocate(**other_executor, 4 * sizeof(int));
  ASSERT_TRUE(target.ok()) << target.status();
  auto batch = (*data)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  batch->targets = *target;
  FakeModel model;
  auto loss = MakeLoss();
  ASSERT_TRUE(loss.ok()) << loss.status();
  auto result = EvaluateBatch(*executor_, model, **loss, *batch);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(model.forward_calls, 0);
}

TEST_F(TrainerTest, RejectsIncompatibleSignaturesBeforeEitherForward) {
  auto data = MakeData();
  auto buffer = Buffer::Allocate(*executor_, 4 * sizeof(float));
  ASSERT_TRUE(data.ok()) << data.status();
  ASSERT_TRUE(buffer.ok()) << buffer.status();
  auto batch = (*data)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();

  struct Signatures {
    std::vector<ActivationType> model_inputs;
    std::vector<ActivationType> model_outputs;
    std::vector<ActivationType> loss_inputs;
    std::vector<ActivationType> loss_outputs;
  };
  const ActivationType bf16(DataType::BF16,
                            {ActivationType::kBatchDimension, 4});
  const ActivationType two_features(DataType::FP32,
                                    {ActivationType::kBatchDimension, 4, 2});
  const std::vector<Signatures> invalid_signatures = {
      // Models consume one dataset input and must produce at least one output.
      {{}, {kFloatType}, {kFloatType, kTokenType}, {kFloatType}},
      {{kTokenType, kTokenType},
       {kFloatType},
       {kFloatType, kTokenType},
       {kFloatType}},
      {{kTokenType}, {}, {kTokenType}, {kFloatType}},
      // Producer/consumer equality checks dtype and shape, not just byte count.
      {{kTokenType}, {kFloatType}, {bf16, kTokenType}, {kFloatType}},
      {{kTokenType}, {kFloatType}, {two_features, kTokenType}, {kFloatType}},
      // The final loss input is exactly one dataset target.
      {{kTokenType}, {kFloatType}, {kFloatType}, {kFloatType}},
      {{kTokenType},
       {kFloatType},
       {kFloatType, kTokenType, kTokenType},
       {kFloatType}},
      {{kTokenType}, {kFloatType}, {kFloatType, kTokenType}, {}},
      {{kTokenType},
       {kFloatType},
       {kFloatType, kTokenType},
       {kFloatType, kFloatType}},
      {{kTokenType}, {kFloatType}, {kFloatType, kTokenType}, {bf16}},
      {{kTokenType}, {kFloatType}, {kFloatType, kTokenType}, {two_features}},
  };
  for (size_t i = 0; i < invalid_signatures.size(); ++i) {
    SCOPED_TRACE(i);
    const auto& signatures = invalid_signatures[i];
    RoutingLayer model({*buffer}, {}, signatures.model_inputs,
                       signatures.model_outputs);
    RoutingLayer loss({*buffer}, {}, signatures.loss_inputs,
                      signatures.loss_outputs);
    const auto result = EvaluateBatch(*executor_, model, loss, *batch);
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(result.status().message().find("RoutingLayer"),
              absl::string_view::npos);
    EXPECT_EQ(model.forward_calls, 0);
    EXPECT_EQ(loss.forward_calls, 0);
  }
}

TEST_F(TrainerTest, RejectsInputTargetAndBatchSizeMismatchBeforeForward) {
  auto data = MakeData();
  auto short_buffer = Buffer::Allocate(*executor_, 3 * sizeof(int));
  auto wide_buffer = Buffer::Allocate(*executor_, 8 * sizeof(int));
  ASSERT_TRUE(data.ok()) << data.status();
  ASSERT_TRUE(short_buffer.ok()) << short_buffer.status();
  ASSERT_TRUE(wide_buffer.ok()) << wide_buffer.status();
  auto batch = (*data)->Next();
  auto loss = MakeLoss();
  ASSERT_TRUE(batch.ok()) << batch.status();
  ASSERT_TRUE(loss.ok()) << loss.status();
  FakeModel model;

  auto invalid = *batch;
  invalid.inputs = *short_buffer;
  EXPECT_EQ(EvaluateBatch(*executor_, model, **loss, invalid).status().code(),
            absl::StatusCode::kInvalidArgument);
  invalid = *batch;
  invalid.targets = *short_buffer;
  EXPECT_EQ(EvaluateBatch(*executor_, model, **loss, invalid).status().code(),
            absl::StatusCode::kInvalidArgument);
  // Inputs bind -2 to batch_size=1; targets cannot independently choose 2.
  invalid = *batch;
  invalid.targets = *wide_buffer;
  EXPECT_EQ(EvaluateBatch(*executor_, model, **loss, invalid).status().code(),
            absl::StatusCode::kInvalidArgument);
  // The same bytes cannot be reinterpreted as two shorter samples.
  invalid = *batch;
  invalid.batch_size = 2;
  invalid.sequence_length = 2;
  EXPECT_EQ(EvaluateBatch(*executor_, model, **loss, invalid).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(model.forward_calls, 0);
  EXPECT_EQ((*loss)->forward_calls, 0);
}

TEST_F(TrainerTest, RejectsReturnedModelBuffersBeforeLossForward) {
  auto data = MakeData();
  auto short_buffer = Buffer::Allocate(*executor_, 3 * sizeof(float));
  auto wide_buffer = Buffer::Allocate(*executor_, 8 * sizeof(float));
  auto valid_buffer = Buffer::Allocate(*executor_, 4 * sizeof(float));
  auto other_executor = cuda::Executor::Create();
  ASSERT_TRUE(data.ok()) << data.status();
  ASSERT_TRUE(short_buffer.ok()) << short_buffer.status();
  ASSERT_TRUE(wide_buffer.ok()) << wide_buffer.status();
  ASSERT_TRUE(valid_buffer.ok()) << valid_buffer.status();
  ASSERT_TRUE(other_executor.ok()) << other_executor.status();
  auto foreign_buffer = Buffer::Allocate(**other_executor, 4 * sizeof(float));
  ASSERT_TRUE(foreign_buffer.ok()) << foreign_buffer.status();
  auto batch = (*data)->Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  for (const BufferVec& outputs :
       {BufferVec{}, BufferVec{*valid_buffer, *valid_buffer},
        BufferVec{*short_buffer}, BufferVec{*wide_buffer},
        BufferVec{*foreign_buffer}}) {
    RoutingLayer model(outputs, {}, {kTokenType}, {kFloatType});
    RoutingLayer loss({*valid_buffer}, {}, {kFloatType, kTokenType},
                      {kFloatType});
    const auto result = EvaluateBatch(*executor_, model, loss, *batch);
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(model.forward_calls, 1);
    EXPECT_EQ(loss.forward_calls, 0);
  }
}

TEST_F(TrainerTest, GenericTrainingMatchesExplicitSparseAutoEncoderUpdate) {
  constexpr int kRows = 16;
  constexpr int kWidth = 16;
  constexpr int kFeatures = 32;
  auto host =
      cuda::PageLockedHostArray<float>::Allocate(*executor_, kRows * kWidth);
  ASSERT_TRUE(host.ok()) << host.status();
  for (size_t i = 0; i < host->size(); ++i)
    (*host)[i] = (static_cast<int>(i % 13) - 6) / 8.0f;
  auto input = Buffer::Allocate(*executor_, host->size_bytes());
  ASSERT_TRUE(input.ok()) << input.status();
  ASSERT_EQ(cudaMemcpyAsync(input->data(), host->data(), host->size_bytes(),
                            cudaMemcpyHostToDevice, executor_->stream()),
            cudaSuccess);

  auto manual = SparseAutoEncoderLayer::Create(
      *executor_, kWidth, kFeatures, DataType::FP16,
      SparseAutoEncoderLayer::Mode::kDefault, kRows / 2);
  auto trained = SparseAutoEncoderLayer::Create(
      *executor_, kWidth, kFeatures, DataType::FP16,
      SparseAutoEncoderLayer::Mode::kDefault, kRows / 2);
  auto loss = SparseAutoEncoderLossLayer::Create(
      *executor_, kWidth, kFeatures, 0.5f, DataType::FP16, kRows / 2);
  ASSERT_TRUE(manual.ok()) << manual.status();
  ASSERT_TRUE(trained.ok()) << trained.status();
  ASSERT_TRUE(loss.ok()) << loss.status();
  ASSERT_TRUE((*manual)->InitializeNormal(0.05f, 19).ok());
  ASSERT_TRUE((*trained)->InitializeNormal(0.05f, 19).ok());
  auto manual_optimizer =
      AdamWOptimizer::Create(*executor_, **manual, AdamWConfig{});
  auto optimizer = AdamWOptimizer::Create(*executor_, **trained, AdamWConfig{});
  ASSERT_TRUE(manual_optimizer.ok()) << manual_optimizer.status();
  ASSERT_TRUE(optimizer.ok()) << optimizer.status();

  auto forward = (*manual)->fwd(*executor_, {*input});
  ASSERT_TRUE(forward.ok()) << forward.status();
  ASSERT_EQ(forward->outputs.size(), 3);
  auto loss_forward = (*loss)->fwd(
      *executor_,
      {forward->outputs[0], forward->outputs[1], forward->outputs[2], *input});
  ASSERT_TRUE(loss_forward.ok()) << loss_forward.status();
  auto gradients = (*loss)->bwd(*executor_, {}, std::move(loss_forward->state));
  ASSERT_TRUE(gradients.ok()) << gradients.status();
  ASSERT_EQ(gradients->size(), forward->outputs.size());
  // All returned gradients are model derivatives, including the regularizer's
  // direct d_D. No target derivative is computed despite the input alias.
  ASSERT_TRUE(
      (*manual)->bwd(*executor_, *gradients, std::move(forward->state)).ok());
  ASSERT_TRUE((*manual_optimizer)->ApplyStep().ok());

  FixedActivationDataSetIterator data(*input, 2, kRows / 2);
  auto result = Train(*executor_, **trained,
                      TrainingOptions{.loss_layer = **loss,
                                      .optimizer = **optimizer,
                                      .training_data = data,
                                      .max_steps = 1});
  ASSERT_TRUE(result.ok()) << result.status();
  for (size_t i = 0; i < (*manual)->weights().size(); ++i) {
    const Buffer& expected = (*manual)->weights()[i];
    const Buffer& actual = (*trained)->weights()[i];
    auto expected_host = cuda::PageLockedHostArray<float>::Allocate(
        *executor_, expected.size_bytes() / sizeof(float));
    auto actual_host = cuda::PageLockedHostArray<float>::Allocate(
        *executor_, actual.size_bytes() / sizeof(float));
    ASSERT_TRUE(expected_host.ok()) << expected_host.status();
    ASSERT_TRUE(actual_host.ok()) << actual_host.status();
    ASSERT_EQ(cudaMemcpyAsync(expected_host->data(), expected.data(),
                              expected.size_bytes(), cudaMemcpyDeviceToHost,
                              executor_->stream()),
              cudaSuccess);
    ASSERT_EQ(
        cudaMemcpyAsync(actual_host->data(), actual.data(), actual.size_bytes(),
                        cudaMemcpyDeviceToHost, executor_->stream()),
        cudaSuccess);
    ASSERT_TRUE(executor_->Synchronize().ok());
    EXPECT_EQ(std::vector<float>(expected_host->begin(), expected_host->end()),
              std::vector<float>(actual_host->begin(), actual_host->end()))
        << "parameter tensor " << i;
  }
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
      *executor_, kInputDimension, kFeatureDimension, DataType::FP16,
      SparseAutoEncoderLayer::Mode::kDefault, kRows / 2);
  auto loss = SparseAutoEncoderLossLayer::Create(*executor_, kInputDimension,
                                                 kFeatureDimension, 0.5f,
                                                 DataType::FP16, kRows / 2);
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_TRUE(loss.ok()) << loss.status();
  ASSERT_TRUE((*model)->InitializeNormal(0.05f, 19).ok());
  FixedActivationDataSetIterator data(*activations, 2, kRows / 2);

  auto initial = Evaluate(
      *executor_, **model,
      EvaluationOptions{.loss_layer = **loss, .eval_data = data, .batches = 1});
  ASSERT_TRUE(initial.ok()) << initial.status();
  auto host_initial = ReadEvaluationLoss(*executor_, *initial);
  ASSERT_TRUE(host_initial.ok()) << host_initial.status();
  EXPECT_TRUE(std::isfinite(*host_initial));
  EXPECT_GE(*host_initial, 0.0);

  auto batch = data.Next();
  ASSERT_TRUE(batch.ok()) << batch.status();
  auto pass = EvaluateBatch(*executor_, **model, **loss, *batch);
  ASSERT_TRUE(pass.ok()) << pass.status();
  EXPECT_EQ(*batch->token_count(), kRows);
  // Batch is symbolic, but the declared per-sample sequence length is exact.
  // Reinterpreting these rows as one-token samples requires another layer
  // instance with that signature; an equal flattened byte size is not enough.
  FixedActivationDataSetIterator single_token_samples(*activations, kRows);
  auto ungrouped = Evaluate(*executor_, **model,
                            EvaluationOptions{.loss_layer = **loss,
                                              .eval_data = single_token_samples,
                                              .batches = 1});
  EXPECT_EQ(ungrouped.status().code(), absl::StatusCode::kInvalidArgument);

  // A separately declared one-token model can process the very same rows.
  // Its identically initialized weights must give the same per-token mean:
  // changing sample grouping does not change SAE math or loss normalization.
  auto single_token_model = SparseAutoEncoderLayer::Create(
      *executor_, kInputDimension, kFeatureDimension, DataType::FP16,
      SparseAutoEncoderLayer::Mode::kDefault, 1);
  auto single_token_loss = SparseAutoEncoderLossLayer::Create(
      *executor_, kInputDimension, kFeatureDimension, 0.5f, DataType::FP16, 1);
  ASSERT_TRUE(single_token_model.ok()) << single_token_model.status();
  ASSERT_TRUE(single_token_loss.ok()) << single_token_loss.status();
  ASSERT_TRUE((*single_token_model)->InitializeNormal(0.05f, 19).ok());
  auto single_token_mean =
      Evaluate(*executor_, **single_token_model,
               EvaluationOptions{.loss_layer = **single_token_loss,
                                 .eval_data = single_token_samples,
                                 .batches = 1});
  ASSERT_TRUE(single_token_mean.ok()) << single_token_mean.status();
  auto host_single_token_mean =
      ReadEvaluationLoss(*executor_, *single_token_mean);
  ASSERT_TRUE(host_single_token_mean.ok()) << host_single_token_mean.status();
  EXPECT_FLOAT_EQ(*host_single_token_mean, *host_initial);

  batch->sequence_length = kRows;
  EXPECT_FALSE(EvaluateBatch(*executor_, **model, **loss, *batch).ok());
  batch->sequence_length = -1;
  EXPECT_FALSE(EvaluateBatch(*executor_, **model, **loss, *batch).ok());

  FakeOptimizer optimizer;
  std::vector<int> evaluation_steps;
  auto result = Train(*executor_, **model,
                      TrainingOptions{
                          .loss_layer = **loss,
                          .optimizer = optimizer,
                          .training_data = data,
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

  auto result =
      Train(*executor_, model,
            TrainingOptions{
                .loss_layer = **loss,
                .optimizer = optimizer,
                .training_data = **training_data,
                .max_steps = 3,
                .initial_step = 570,
                .evaluation_interval = 1,
                .evaluation_batches = 1,
                .stop_loss = 2.5,
                .evaluation_data = evaluation_data->get(),
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

  auto result =
      Train(*executor_, model,
            TrainingOptions{
                .loss_layer = **loss,
                .optimizer = optimizer,
                .training_data = **training_data,
                .max_steps = 3,
                .evaluation_interval = 1,
                .evaluation_batches = 1,
                .stop_loss = 2.5,
                .evaluation_data = evaluation_data->get(),
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

  auto result =
      Train(*executor_, model,
            TrainingOptions{
                .loss_layer = **loss,
                .optimizer = optimizer,
                .training_data = **training_data,
                .max_steps = 3,
                .evaluation_interval = 2,
                .evaluation_batches = 1,
                .evaluation_data = evaluation_data->get(),
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

  auto result =
      Train(*executor_, model,
            TrainingOptions{.loss_layer = **loss,
                            .optimizer = optimizer,
                            .training_data = **training_data,
                            .max_steps = 3,
                            .initial_step = 570,
                            .evaluation_interval = 2,
                            .evaluation_batches = 1,
                            .evaluation_data = evaluation_data->get(),
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

  auto result =
      Train(*executor_, model,
            TrainingOptions{
                .loss_layer = **loss,
                .optimizer = optimizer,
                .training_data = **data,
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

  auto result = Train(
      *executor_, model,
      TrainingOptions{.loss_layer = **loss,
                      .optimizer = optimizer,
                      .training_data = **data,
                      .max_steps = kUnlimitedTrainingSteps,
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
      Train(*executor_, model,
            TrainingOptions{.loss_layer = **loss,
                            .optimizer = optimizer,
                            .training_data = **training_data,
                            .max_steps = kUnlimitedTrainingSteps,
                            .initial_step = 570,
                            .training_seconds = 0.01,
                            .evaluation_interval = 100,
                            .evaluation_data = evaluation_data->get(),
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
  std::vector<int> evaluation_steps;

  auto result =
      Train(*executor_, **model,
            TrainingOptions{.loss_layer = **loss,
                            .optimizer = **optimizer,
                            .training_data = data,
                            .max_steps = kUnlimitedTrainingSteps,
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
  auto result = Train(*executor_, model,
                      TrainingOptions{.loss_layer = **loss,
                                      .optimizer = optimizer,
                                      .training_data = **data,
                                      .max_steps = 1,
                                      .training_seconds = 3600.0});
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

  for (double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_EQ(Train(*executor_, model,
                    TrainingOptions{.loss_layer = **loss,
                                    .optimizer = optimizer,
                                    .training_data = **data,
                                    .max_steps = 1,
                                    .training_seconds = invalid})
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
  EXPECT_FALSE(
      Evaluate(*executor_, model,
               EvaluationOptions{
                   .loss_layer = **loss, .eval_data = **data, .batches = 0})
          .ok());
  EXPECT_FALSE(Train(*executor_, model,
                     TrainingOptions{.loss_layer = **loss,
                                     .optimizer = optimizer,
                                     .training_data = **data,
                                     .max_steps = -2})
                   .ok());
  EXPECT_FALSE(Train(*executor_, model,
                     TrainingOptions{.loss_layer = **loss,
                                     .optimizer = optimizer,
                                     .training_data = **data,
                                     .initial_step = -1})
                   .ok());
  EXPECT_FALSE(
      Train(*executor_, model,
            TrainingOptions{.loss_layer = **loss,
                            .optimizer = optimizer,
                            .training_data = **data,
                            .max_steps = 1,
                            .initial_step = std::numeric_limits<int>::max()})
          .ok());
}

}  // namespace
}  // namespace pluto::llm
