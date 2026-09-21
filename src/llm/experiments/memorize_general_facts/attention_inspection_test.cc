#include "src/llm/experiments/memorize_general_facts/attention_inspection.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

static_assert(!std::is_copy_constructible_v<AttentionProbabilityInspector>);
static_assert(!std::is_move_constructible_v<AttentionProbabilityInspector>);

class TestDetokenizer final : public tokenizer::Detokenizer {
 public:
  absl::StatusOr<std::string> Decode(
      absl::Span<const int> tokens) const override {
    const std::string vocabulary[] = {"a", "\n", "\"", "\\"};
    std::string text;
    for (int id : tokens) {
      if (id < 0 || id >= vocab_size())
        return absl::InvalidArgumentError("invalid test token");
      text += vocabulary[id];
    }
    return text;
  }
  int vocab_size() const override { return 4; }
};

absl::StatusOr<cuda::Buffer> Upload(cuda::Executor& executor,
                                    absl::Span<const float> values) {
  ASSIGN_OR_RETURN(auto host,
                   cuda::PageLockedHostArray<float>::CopyFrom(executor, values));
  ASSIGN_OR_RETURN(auto device,
                   cuda::Buffer::Allocate(executor, host.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(device.data(), host.data(), host.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "cudaMemcpyAsync(test attention upload)"));
  return device;
}

class AttentionInspectionTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }
  void TearDown() override {
    if (executor_ != nullptr)
      EXPECT_TRUE(executor_->Synchronize().ok());
  }

  // Two heads, each with a four-position padded context. Only two positions
  // are actual input. Poison every upper-triangle/padding value to prove that
  // neither validation nor printing accidentally uses causally unrelated data.
  std::vector<float> Probabilities() const {
    std::vector<float> values(2 * 4 * 4,
                              std::numeric_limits<float>::quiet_NaN());
    values[0] = values[16] = 1.0f;
    values[4] = 0.25f;
    values[5] = 0.75f;
    values[20] = 0.625f;
    values[21] = 0.375f;
    return values;
  }

  absl::Status Record(AttentionProbabilityInspector& inspector) {
    ASSIGN_OR_RETURN(auto probabilities, Upload(*executor_, Probabilities()));
    return inspector.layer_hooks().attention_probabilities_hook(
        *executor_, "AttentionLayer",
        ActivationType(DataType::FP32, {1, 2, 4, 4}), probabilities);
    // The inspector must retain a Buffer copy: this local handle is destroyed
    // before Print, and all GPU work is still allowed to be asynchronous.
  }

  TestDetokenizer detokenizer_;
  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(AttentionInspectionTest, PrintsEachHeadAndOnlyLeadingCausalTriangle) {
  AttentionProbabilityInspector inspector(*executor_);
  EXPECT_FALSE(inspector.layer_hooks().activation_hook);
  EXPECT_FALSE(inspector.layer_hooks().gradient_hook);
  ASSERT_TRUE(Record(inspector).ok());
  std::ostringstream output;
  const auto status =
      inspector.Print(*executor_, detokenizer_, {0, 1}, 2, output);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(output.str(),
            "Output token 2 (\"\\\"\", id 2):\n"
            "  Key columns: 0 (\"a\"), 1 (\"\\n\")\n"
            "  AttentionLayer[0], head 0:\n"
            "    Query 0 (\"a\"): [1.000000]\n"
            "    Query 1 (\"\\n\"): [0.250000, 0.750000]\n"
            "  AttentionLayer[0], head 1:\n"
            "    Query 0 (\"a\"): [1.000000]\n"
            "    Query 1 (\"\\n\"): [0.625000, 0.375000]\n");
}

TEST_F(AttentionInspectionTest, DistinguishesBlocksAndResetsAfterEveryToken) {
  AttentionProbabilityInspector inspector(*executor_);
  auto& hooks = inspector.layer_hooks();
  std::string previous;
  for (int forward = 0; forward < 2; ++forward) {
    ASSERT_TRUE(hooks.enter_combinator(*executor_, "Model").ok());
    for (int block = 0; block < 2; ++block) {
      ASSERT_TRUE(hooks.enter_combinator(*executor_, "Block").ok());
      ASSERT_TRUE(Record(inspector).ok());
      ASSERT_TRUE(hooks.exit_combinator(*executor_).ok());
    }
    ASSERT_TRUE(hooks.exit_combinator(*executor_).ok());
    std::ostringstream output;
    ASSERT_TRUE(
        inspector.Print(*executor_, detokenizer_, {0, 1}, 3, output).ok());
    EXPECT_NE(output.str().find("Model[0]/Block[0]/AttentionLayer[0], head 0:"),
              std::string::npos);
    EXPECT_NE(output.str().find("Model[0]/Block[1]/AttentionLayer[0], head 1:"),
              std::string::npos);
    if (forward == 1)
      EXPECT_EQ(output.str(), previous);
    previous = output.str();
    std::ostringstream empty_output;
    EXPECT_TRUE(
        inspector.Print(*executor_, detokenizer_, {0, 1}, 3, empty_output)
            .ok());
    EXPECT_NE(empty_output.str().find("No attention layers."),
              std::string::npos);
    EXPECT_EQ(empty_output.str().find("AttentionLayer"), std::string::npos);
  }
}

TEST_F(AttentionInspectionTest, RejectsMalformedOrUnresolvedProbabilityShapes) {
  AttentionProbabilityInspector inspector(*executor_);
  auto probabilities = Upload(*executor_, Probabilities());
  ASSERT_TRUE(probabilities.ok()) << probabilities.status();
  const ActivationType bad_types[] = {
      {DataType::BF16, {1, 2, 4, 4}},
      {DataType::FP32, {2, 1, 4, 4}},
      {DataType::FP32, {ActivationType::kBatchDimension, 2, 4, 4}},
      {DataType::FP32, {1, 2, 4, 3}},
      {DataType::FP32, {1, 2, 4}},
      {DataType::FP32, {1, 2, 4, 4, 1}},
      {DataType::FP32, {1, 0, 4, 4}},
      {DataType::FP32, {1, 2, 2, 2}},  // Byte size differs.
      {DataType::FP32, {1, std::numeric_limits<int64_t>::max(), 4, 4}},
  };
  for (const ActivationType& type : bad_types)
    EXPECT_EQ(inspector.layer_hooks()
                  .attention_probabilities_hook(*executor_, "Attention", type,
                                                *probabilities)
                  .code(),
              absl::StatusCode::kInvalidArgument);
  ASSERT_TRUE(Record(inspector).ok());
  std::ostringstream output;
  EXPECT_TRUE(
      inspector.Print(*executor_, detokenizer_, {0, 1}, 2, output).ok());
  // Invalid records must not consume path indices or pollute later reports.
  EXPECT_NE(output.str().find("AttentionLayer[0]"), std::string::npos);
}

TEST_F(AttentionInspectionTest, ValidatesPrefixTokenIdsAndScopeBalance) {
  AttentionProbabilityInspector inspector(*executor_);
  auto& hooks = inspector.layer_hooks();
  std::ostringstream output;
  EXPECT_EQ(hooks.exit_combinator(*executor_).code(),
            absl::StatusCode::kFailedPrecondition);
  ASSERT_TRUE(hooks.enter_combinator(*executor_, "Model").ok());
  ASSERT_TRUE(Record(inspector).ok());
  EXPECT_EQ(inspector.Print(*executor_, detokenizer_, {0, 1}, 2, output).code(),
            absl::StatusCode::kFailedPrecondition);
  ASSERT_TRUE(hooks.exit_combinator(*executor_).ok());
  for (const auto& prefix :
       std::vector<std::vector<int>>{{}, {0, -1}, {0, 4}, {0, 0, 0, 0, 0}})
    EXPECT_EQ(
        inspector.Print(*executor_, detokenizer_, prefix, 2, output).code(),
        absl::StatusCode::kInvalidArgument);
  for (int token : {-1, 4})
    EXPECT_EQ(
        inspector.Print(*executor_, detokenizer_, {0, 1}, token, output).code(),
        absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(output.str().empty());
  EXPECT_TRUE(
      inspector.Print(*executor_, detokenizer_, {0, 1}, 2, output).ok());
}

TEST_F(AttentionInspectionTest, RejectsForeignExecutorsAndBuffers) {
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  AttentionProbabilityInspector inspector(*executor_);
  auto& hooks = inspector.layer_hooks();
  auto probabilities = Upload(**other, Probabilities());
  ASSERT_TRUE(probabilities.ok()) << probabilities.status();
  const ActivationType type(DataType::FP32, {1, 2, 4, 4});
  EXPECT_EQ(hooks
                .attention_probabilities_hook(*executor_, "Attention", type,
                                              *probabilities)
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(hooks
                .attention_probabilities_hook(**other, "Attention", type,
                                              *probabilities)
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(hooks.enter_combinator(**other, "Model").code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(hooks.exit_combinator(**other).code(),
            absl::StatusCode::kInvalidArgument);
  std::ostringstream output;
  EXPECT_EQ(inspector.Print(**other, detokenizer_, {0, 1}, 2, output).code(),
            absl::StatusCode::kInvalidArgument);
  ASSERT_TRUE(Record(inspector).ok());
  EXPECT_TRUE(
      inspector.Print(*executor_, detokenizer_, {0, 1}, 2, output).ok());
  EXPECT_TRUE((*other)->Synchronize().ok());
}

TEST_F(AttentionInspectionTest, FailedOutputCanBeRetried) {
  AttentionProbabilityInspector inspector(*executor_);
  ASSERT_TRUE(Record(inspector).ok());
  std::ostringstream failed;
  failed.setstate(std::ios::badbit);
  EXPECT_EQ(inspector.Print(*executor_, detokenizer_, {0, 1}, 2, failed).code(),
            absl::StatusCode::kInternal);
  std::ostringstream retry;
  EXPECT_TRUE(inspector.Print(*executor_, detokenizer_, {0, 1}, 2, retry).ok());
  EXPECT_NE(retry.str().find("AttentionLayer[0]"), std::string::npos);
}

TEST_F(AttentionInspectionTest, InvalidCausalProbabilityFailsWithoutOutput) {
  for (float invalid : {-0.1f, 1.1f, std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
    AttentionProbabilityInspector inspector(*executor_);
    auto values = Probabilities();
    values[4] = invalid;
    auto probabilities = Upload(*executor_, values);
    ASSERT_TRUE(probabilities.ok()) << probabilities.status();
    ASSERT_TRUE(inspector.layer_hooks()
                    .attention_probabilities_hook(
                        *executor_, "Attention", {DataType::FP32, {1, 2, 4, 4}},
                        *probabilities)
                    .ok());
    std::ostringstream output;
    EXPECT_EQ(
        inspector.Print(*executor_, detokenizer_, {0, 1}, 2, output).code(),
        absl::StatusCode::kDataLoss);
    EXPECT_TRUE(output.str().empty());
  }
}

TEST_F(AttentionInspectionTest, AcceptsTinyProbabilityReconstructionRoundoff) {
  AttentionProbabilityInspector inspector(*executor_);
  auto values = Probabilities();
  values[0] = 1.0f + 1e-6f;
  auto probabilities = Upload(*executor_, values);
  ASSERT_TRUE(probabilities.ok()) << probabilities.status();
  ASSERT_TRUE(inspector.layer_hooks()
                  .attention_probabilities_hook(*executor_, "Attention",
                                                {DataType::FP32, {1, 2, 4, 4}},
                                                *probabilities)
                  .ok());
  std::ostringstream output;
  EXPECT_TRUE(
      inspector.Print(*executor_, detokenizer_, {0, 1}, 2, output).ok());
}

TEST_F(AttentionInspectionTest,
       ZeroAttentionLayersProduceAnExplicitEmptyReport) {
  AttentionProbabilityInspector inspector(*executor_);
  ASSERT_TRUE(
      inspector.layer_hooks().enter_combinator(*executor_, "Model").ok());
  ASSERT_TRUE(inspector.layer_hooks().exit_combinator(*executor_).ok());
  std::ostringstream output;
  EXPECT_TRUE(
      inspector.Print(*executor_, detokenizer_, {0, 1}, 2, output).ok());
  EXPECT_EQ(output.str(),
            "Output token 2 (\"\\\"\", id 2):\n"
            "  No attention layers.\n");
}

}  // namespace
}  // namespace pluto::llm
