#include "src/llm/badam_optimizer.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

class BAdamOptimizerTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }
  void TearDown() override {
    if (executor_)
      EXPECT_TRUE(executor_->Synchronize().ok());
  }

  template <class T>
  absl::Status Copy(const std::vector<T>& values, Buffer buffer) {
    ASSIGN_OR_RETURN(auto host,
                     cuda::PageLockedHostArray<T>::CopyFrom(*executor_, values));
    if (buffer.size_bytes() != values.size() * sizeof(T))
      return absl::InvalidArgumentError("incorrect test upload size");
    return cuda::CudaStatus(
        cudaMemcpyAsync(buffer.data(), host.data(), buffer.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "test upload");
  }

  template <class T>
  absl::StatusOr<std::shared_ptr<BlockParameter>> Parameter(
      const std::vector<T>& values, DataType storage = DataType::FP32) {
    ASSIGN_OR_RETURN(auto buffer,
                     Buffer::Allocate(*executor_, values.size() * sizeof(T)));
    RETURN_IF_ERROR(Copy(values, buffer));
    return BlockParameter::Create(*executor_, std::move(buffer), storage);
  }

  template <class T = float>
  std::vector<float> Read(const Buffer& buffer) {
    auto host = cuda::PageLockedHostArray<T>::Allocate(
        *executor_, buffer.size_bytes() / sizeof(T));
    EXPECT_TRUE(host.ok()) << host.status();
    if (!host.ok())
      return {};
    EXPECT_EQ(cudaMemcpyAsync(host->data(), buffer.data(), buffer.size_bytes(),
                              cudaMemcpyDeviceToHost, executor_->stream()),
              cudaSuccess);
    EXPECT_TRUE(executor_->Synchronize().ok());
    std::vector<float> result;
    for (auto value : *host)
      result.push_back(static_cast<float>(value));
    return result;
  }

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(BAdamOptimizerTest,
       UpdatesOnlyActiveBlockAndReleasesItsBuffersOnSwitch) {
  auto first = Parameter<float>({1.f, -2.f});
  auto second = Parameter<float>({3.f});
  auto third = Parameter<float>({4.f, 5.f, 6.f});
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  ASSERT_TRUE(third.ok());
  const void* const stable_first = (*first)->value().data();
  BAdamConfig config;
  config.adam.learning_rate = .1f;
  config.adam.weight_decay = 0;
  config.switch_every = 1;
  config.start_block = 0;
  config.descending = false;
  auto created = BAdamOptimizer::Create(
      *executor_, {{"first", {*first, *second}}, {"next", {*third}}}, config);
  ASSERT_TRUE(created.ok()) << created.status();
  auto optimizer = std::move(*created);
  EXPECT_TRUE((*first)->active());
  EXPECT_TRUE((*second)->active());
  EXPECT_FALSE((*third)->active());
  EXPECT_EQ(optimizer->parameter_tensor_count(), 2u);
  EXPECT_EQ(optimizer->active_state_bytes(), 3u * 16);
  EXPECT_TRUE(absl::IsFailedPrecondition(optimizer->ApplyStep()));
  ASSERT_TRUE(optimizer->ZeroGrad().ok());
  ASSERT_TRUE(Copy<float>({1.f, -1.f}, (*first)->gradient()).ok());
  ASSERT_TRUE(Copy<float>({2.f}, (*second)->gradient()).ok());
  ASSERT_TRUE(optimizer->ApplyStep().ok());
  EXPECT_EQ(optimizer->step(), 1);
  EXPECT_EQ(optimizer->block_step(), 1);
  EXPECT_EQ(optimizer->active_block(), 0);
  const auto first_values = Read((*first)->value());
  EXPECT_NEAR(first_values[0], .9f, 1e-6);
  EXPECT_NEAR(first_values[1], -1.9f, 1e-6);
  EXPECT_NEAR(Read((*second)->value())[0], 2.9f, 1e-6);
  EXPECT_EQ(Read((*third)->value()), (std::vector<float>{4, 5, 6}));
  EXPECT_EQ(Read((*first)->gradient()), (std::vector<float>{0, 0}));
  EXPECT_TRUE(absl::IsFailedPrecondition(optimizer->ApplyStep()));

  // The old block stays active until the next batch is prepared. This avoids
  // changing the optimized block between a recorded forward and its backward.
  ASSERT_TRUE(optimizer->ZeroGrad().ok());
  EXPECT_EQ(optimizer->active_block(), 1);
  EXPECT_EQ(optimizer->block_step(), 0);
  EXPECT_FALSE((*first)->active());
  EXPECT_FALSE((*second)->active());
  EXPECT_TRUE((*third)->active());
  EXPECT_EQ(optimizer->parameter_tensor_count(), 1u);
  EXPECT_EQ(optimizer->active_state_bytes(), 3u * 16);
  ASSERT_TRUE(Copy<float>({1.f, 1.f, 1.f}, (*third)->gradient()).ok());
  // Calling ZeroGrad twice discards the pending gradient, not the block.
  ASSERT_TRUE(optimizer->ZeroGrad().ok());
  EXPECT_EQ(optimizer->active_block(), 1);
  EXPECT_EQ(Read((*third)->gradient()), (std::vector<float>{0, 0, 0}));
  ASSERT_TRUE(Copy<float>({-1.f, -1.f, -1.f}, (*third)->gradient()).ok());
  ASSERT_TRUE(optimizer->ApplyStep().ok());
  EXPECT_EQ(Read((*first)->value()), first_values);
  EXPECT_EQ((*first)->value().data(), stable_first);
  EXPECT_NEAR(Read((*third)->value())[0], 4.1f, 1e-6);
  optimizer.reset();
  EXPECT_FALSE((*first)->active());
  EXPECT_FALSE((*second)->active());
  EXPECT_FALSE((*third)->active());
  EXPECT_EQ(Read((*first)->value()), first_values);
  EXPECT_NEAR(Read((*third)->value())[0], 4.1f, 1e-6);
}

TEST_F(BAdamOptimizerTest, DescendingVisitsResetMomentsAndLocalBiasCorrection) {
  auto first = Parameter<float>({1.f});
  auto second = Parameter<float>({2.f});
  auto third = Parameter<float>({3.f});
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  ASSERT_TRUE(third.ok());
  const std::vector<std::shared_ptr<BlockParameter>> parameters{*first, *second,
                                                                *third};
  BAdamConfig config;
  config.adam = {.learning_rate = .07f,
                 .beta1 = .5f,
                 .beta2 = .75f,
                 .epsilon = .01f,
                 .weight_decay = .02f};
  config.switch_every = 2;
  config.start_block = 1;
  auto optimizer = BAdamOptimizer::Create(
      *executor_,
      {{"first", {*first}}, {"second", {*second}}, {"third", {*third}}},
      config);
  ASSERT_TRUE(optimizer.ok()) << optimizer.status();
  double expected[] = {1, 2, 3};
  const int visits[] = {1, 0, 2, 1};
  for (int visit = 0; visit < 4; ++visit) {
    const int active = visits[visit];
    double moment = 0, variance = 0;
    for (int local = 1; local <= 2; ++local) {
      ASSERT_TRUE((*optimizer)->ZeroGrad().ok());
      EXPECT_EQ((*optimizer)->active_block(), active);
      EXPECT_EQ((*optimizer)->block_step(), local - 1);
      const float gradient = visit == 3 ? -3.f * local : .25f + visit + local;
      ASSERT_TRUE(Copy<float>({gradient}, parameters[active]->gradient()).ok());
      moment = config.adam.beta1 * moment + (1 - config.adam.beta1) * gradient;
      variance = config.adam.beta2 * variance +
                 (1 - config.adam.beta2) * gradient * gradient;
      const double mhat = moment / (1 - std::pow(config.adam.beta1, local));
      const double vhat = variance / (1 - std::pow(config.adam.beta2, local));
      expected[active] -= config.adam.learning_rate *
                          (mhat / (std::sqrt(vhat) + config.adam.epsilon) +
                           config.adam.weight_decay * expected[active]);
      ASSERT_TRUE((*optimizer)->ApplyStep().ok());
      for (int p = 0; p < 3; ++p)
        EXPECT_NEAR(Read(parameters[p]->value())[0], expected[p], 2e-6);
      EXPECT_EQ((*optimizer)->step(), visit * 2 + local);
      EXPECT_EQ((*optimizer)->block_step(), local);
    }
  }
}

TEST_F(BAdamOptimizerTest, Bfloat16KeepsSubUlpUpdatesInMasterDuringBlockVisit) {
  auto parameter =
      Parameter<__nv_bfloat16>({__nv_bfloat16(1.f)}, DataType::BF16);
  ASSERT_TRUE(parameter.ok());
  BAdamConfig config;
  config.adam.learning_rate = 1e-4f;
  config.adam.weight_decay = 0;
  config.switch_every = 100;
  auto optimizer =
      BAdamOptimizer::Create(*executor_, {{"only", {*parameter}}}, config);
  ASSERT_TRUE(optimizer.ok());
  for (int i = 0; i < 32; ++i) {
    ASSERT_TRUE((*optimizer)->ZeroGrad().ok());
    ASSERT_TRUE(Copy<float>({1.f}, (*parameter)->gradient()).ok());
    ASSERT_TRUE((*optimizer)->ApplyStep().ok());
    if (i == 0) {
      EXPECT_EQ(Read<__nv_bfloat16>((*parameter)->value())[0], 1.f);
      EXPECT_LT(Read((*parameter)->master())[0], 1.f);
    }
  }
  const float master = Read((*parameter)->master())[0];
  EXPECT_NEAR(master, 1.f - .0032f, 2e-6);
  EXPECT_EQ(Read<__nv_bfloat16>((*parameter)->value())[0],
            static_cast<float>(__nv_bfloat16(master)));
  EXPECT_LT(Read<__nv_bfloat16>((*parameter)->value())[0], 1.f);
}

TEST_F(BAdamOptimizerTest,
       RejectsInvalidConfigBlocksAliasingAndBudgetBeforeActivation) {
  auto first = Parameter<float>({1.f});
  auto second = Parameter<float>({2.f, 3.f});
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  const std::vector<BAdamBlock> blocks{{"first", {*first}},
                                       {"second", {*second}}};
  BAdamConfig config;
  config.start_block = 0;
  config.max_active_bytes = 16;
  // The larger *inactive* block must be checked before activating the first.
  EXPECT_TRUE(absl::IsResourceExhausted(
      BAdamOptimizer::Create(*executor_, blocks, config).status()));
  EXPECT_FALSE((*first)->active());
  EXPECT_FALSE((*second)->active());
  config.max_active_bytes = 32;
  {
    auto optimizer = BAdamOptimizer::Create(*executor_, blocks, config);
    ASSERT_TRUE(optimizer.ok());
  }
  EXPECT_FALSE((*first)->active());
  EXPECT_FALSE((*second)->active());
  config.max_active_bytes = 0;
  for (int which = 0; which < 11; ++which) {
    BAdamConfig invalid = config;
    switch (which) {
      case 0:
        invalid.switch_every = 0;
        break;
      case 1:
        invalid.start_block = -2;
        break;
      case 2:
        invalid.start_block = 2;
        break;
      case 3:
        invalid.adam.learning_rate = 0;
        break;
      case 4:
        invalid.adam.learning_rate = std::numeric_limits<float>::infinity();
        break;
      case 5:
        invalid.adam.beta1 = 1;
        break;
      case 6:
        invalid.adam.beta2 = -1;
        break;
      case 7:
        invalid.adam.epsilon = 0;
        break;
      case 8:
        invalid.adam.weight_decay = -1;
        break;
      case 9:
        invalid.adam.beta1 = std::numeric_limits<float>::quiet_NaN();
        break;
      case 10:
        invalid.adam.epsilon = std::numeric_limits<float>::quiet_NaN();
        break;
    }
    EXPECT_TRUE(absl::IsInvalidArgument(
        BAdamOptimizer::Create(*executor_, blocks, invalid).status()))
        << which;
    EXPECT_FALSE((*first)->active());
    EXPECT_FALSE((*second)->active());
  }
  EXPECT_FALSE(BAdamOptimizer::Create(*executor_, {}).ok());
  EXPECT_FALSE(BAdamOptimizer::Create(*executor_, {{"empty", {}}}).ok());
  EXPECT_FALSE(BAdamOptimizer::Create(*executor_, {{"null", {nullptr}}}).ok());
  EXPECT_FALSE(
      BAdamOptimizer::Create(*executor_, {{"duplicate", {*first, *first}}})
          .ok());
  EXPECT_FALSE(
      BAdamOptimizer::Create(*executor_, {{"one", {*first}}, {"two", {*first}}})
          .ok());
  auto alias =
      BlockParameter::Create(*executor_, (*first)->value(), DataType::FP32);
  ASSERT_TRUE(alias.ok());
  EXPECT_FALSE(
      BAdamOptimizer::Create(*executor_, {{"aliases", {*first, *alias}}}).ok());
  ASSERT_TRUE((*first)->Activate().ok());
  EXPECT_FALSE(BAdamOptimizer::Create(*executor_, blocks).ok());
  EXPECT_TRUE((*first)->active());
  ASSERT_TRUE((*first)->Deactivate().ok());
  auto other_executor = cuda::Executor::Create();
  ASSERT_TRUE(other_executor.ok());
  EXPECT_FALSE(BAdamOptimizer::Create(**other_executor, blocks).ok());
  EXPECT_FALSE((*first)->active());
  EXPECT_FALSE((*second)->active());
}

TEST_F(BAdamOptimizerTest, DefaultsStartAtLastBlock) {
  auto first = Parameter<float>({1.f});
  auto second = Parameter<float>({2.f});
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  auto optimizer = BAdamOptimizer::Create(
      *executor_, {{"first", {*first}}, {"last", {*second}}});
  ASSERT_TRUE(optimizer.ok());
  EXPECT_EQ((*optimizer)->active_block(), 1);
  EXPECT_FALSE((*first)->active());
  EXPECT_TRUE((*second)->active());
}

}  // namespace
}  // namespace pluto::llm
