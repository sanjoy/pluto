#include "src/llm/trainer.h"

#include <cuda_runtime.h>

#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/dataset/dataset.h"
#include "src/llm/layer.h"
#include "src/llm/optimizer.h"

namespace pluto::llm {
namespace {

class FakeModel final : public Layer {
 public:
  absl::StatusOr<Buffer> fwd(cuda::Executor& executor,
                             absl::Span<const Buffer> inputs,
                             Tape* tape) const override {
    ++forward_calls;
    return inputs[0];
  }

  absl::StatusOr<BufferVec> bwd(cuda::Executor& executor,
                                absl::Span<const Buffer> output_gradients,
                                Tape tape) override {
    ++backward_calls;
    return BufferVec{};
  }

  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return DataType::FP16; }

  mutable int forward_calls = 0;
  int backward_calls = 0;
};

class FakeLoss final : public Layer {
 public:
  static absl::StatusOr<std::unique_ptr<FakeLoss>> Create(
      cuda::Executor& executor, absl::Span<const float> losses) {
    auto device_losses =
        Buffer::Allocate(executor, losses.size() * sizeof(float));
    if (!device_losses.ok()) return device_losses.status();
    auto gradient = Buffer::Allocate(executor, losses.size() * sizeof(float));
    if (!gradient.ok()) return gradient.status();
    if (cudaMemcpyAsync(device_losses->data(), losses.data(),
                        device_losses->size_bytes(), cudaMemcpyHostToDevice,
                        executor.stream()) != cudaSuccess) {
      return absl::Status(absl::StatusCode::kInternal,
                          "failed to initialize fake losses");
    }
    return std::unique_ptr<FakeLoss>(
        new FakeLoss(std::move(*device_losses), std::move(*gradient)));
  }

  absl::StatusOr<Buffer> fwd(cuda::Executor& executor,
                             absl::Span<const Buffer> inputs,
                             Tape* tape) const override {
    ++forward_calls;
    return losses_;
  }

  absl::StatusOr<BufferVec> bwd(cuda::Executor& executor,
                                absl::Span<const Buffer> output_gradients,
                                Tape tape) override {
    ++backward_calls;
    return BufferVec{gradient_};
  }

  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return DataType::FP16; }

  mutable int forward_calls = 0;
  int backward_calls = 0;

 private:
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
    return absl::OkStatus();
  }
  int step() const override { return steps; }
  size_t parameter_tensor_count() const override { return 0; }

  int zero_grad_calls = 0;
  int steps = 0;
};

class TrainerTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
    corpus_.resize(32);
    std::iota(corpus_.begin(), corpus_.end(), 0);
  }

  void TearDown() override {
    if (executor_ == nullptr) return;
    EXPECT_TRUE(executor_->Synchronize().ok());
    executor_.reset();
  }

  absl::StatusOr<std::unique_ptr<InMemoryDataSetIterator>> MakeData() {
    return InMemoryDataSetIterator::Create(
        *executor_, corpus_,
        InMemoryDataSetOptions{
            .batch_size = 4,
            .context_length = 4,
            .order = InMemoryDataSetOrder::kSequential,
        });
  }

  absl::StatusOr<std::unique_ptr<FakeLoss>> MakeLoss() {
    const std::vector<float> losses = {1.0f, 2.0f, 3.0f, 4.0f};
    return FakeLoss::Create(*executor_, losses);
  }

  std::unique_ptr<cuda::Executor> executor_;
  std::vector<int> corpus_;
};

TEST_F(TrainerTest, EvaluateAveragesLossesAndResetsDataset) {
  FakeModel model;
  auto loss = MakeLoss();
  auto data = MakeData();
  ASSERT_TRUE(loss.ok()) << loss.status();
  ASSERT_TRUE(data.ok()) << data.status();

  auto mean = Evaluate(*executor_, model, **loss, **data,
                       EvaluationOptions{.batches = 2});
  ASSERT_TRUE(mean.ok()) << mean.status();
  EXPECT_DOUBLE_EQ(*mean, 2.5);
  EXPECT_EQ(model.forward_calls, 2);
  EXPECT_EQ((*loss)->forward_calls, 2);

  auto repeated = Evaluate(*executor_, model, **loss, **data,
                           EvaluationOptions{.batches = 2});
  ASSERT_TRUE(repeated.ok()) << repeated.status();
  EXPECT_DOUBLE_EQ(*repeated, *mean);
}

TEST_F(TrainerTest, EvaluateRejectsADatasetFromAnotherExecutor) {
  FakeModel model;
  auto loss = MakeLoss();
  auto data = MakeData();
  auto other_executor = cuda::Executor::Create();
  ASSERT_TRUE(loss.ok()) << loss.status();
  ASSERT_TRUE(data.ok()) << data.status();
  ASSERT_TRUE(other_executor.ok()) << other_executor.status();

  const auto mean = Evaluate(**other_executor, model, **loss, **data,
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

  auto result = Train(*executor_, model, **loss, optimizer, **data,
                      TrainingOptions{.max_steps = 3});
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->steps_completed, 3);
  EXPECT_FALSE(result->reached_stop_loss);
  EXPECT_EQ(model.forward_calls, 3);
  EXPECT_EQ(model.backward_calls, 3);
  EXPECT_EQ((*loss)->forward_calls, 3);
  EXPECT_EQ((*loss)->backward_calls, 3);
  EXPECT_EQ(optimizer.zero_grad_calls, 1);
  EXPECT_EQ(optimizer.steps, 3);
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
      Train(*executor_, model, **loss, optimizer, **training_data,
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

  auto result =
      Train(*executor_, model, **loss, optimizer, **training_data,
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

  auto result =
      Train(*executor_, model, **loss, optimizer, **training_data,
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

  auto result =
      Train(*executor_, model, **loss, optimizer, **training_data,
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

  auto result =
      Train(*executor_, model, **loss, optimizer, **data,
            TrainingOptions{
                .max_steps = 3,
                .initial_step = 570,
                .step_callback = [&](int steps_completed) -> absl::Status {
                  callback_steps.push_back(steps_completed);
                  if (steps_completed == 572) {
                    return absl::UnavailableError("checkpoint failed");
                  }
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
      *executor_, model, **loss, optimizer, **data,
      TrainingOptions{.max_steps = kUnlimitedTrainingSteps,
                      .step_callback = [](int steps_completed) -> absl::Status {
                        if (steps_completed == 3) {
                          return absl::CancelledError("test requested stop");
                        }
                        return absl::OkStatus();
                      }});

  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kCancelled);
  EXPECT_EQ(optimizer.steps, 3);
  EXPECT_EQ(model.backward_calls, 3);
}

TEST_F(TrainerTest, RejectsInvalidOptions) {
  FakeModel model;
  FakeOptimizer optimizer;
  auto loss = MakeLoss();
  auto data = MakeData();
  ASSERT_TRUE(loss.ok()) << loss.status();
  ASSERT_TRUE(data.ok()) << data.status();

  EXPECT_FALSE(Evaluate(*executor_, model, **loss, **data,
                        EvaluationOptions{.batches = 0})
                   .ok());
  EXPECT_FALSE(Train(*executor_, model, **loss, optimizer, **data,
                     TrainingOptions{.max_steps = -2})
                   .ok());
  EXPECT_FALSE(Train(*executor_, model, **loss, optimizer, **data,
                     TrainingOptions{.initial_step = -1})
                   .ok());
  EXPECT_FALSE(
      Train(*executor_, model, **loss, optimizer, **data,
            TrainingOptions{.max_steps = 1,
                            .initial_step = std::numeric_limits<int>::max()})
          .ok());
}

}  // namespace
}  // namespace pluto::llm
