#include "ai-slop/weight_analysis/phrase_probe.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <utility>

#include "gtest/gtest.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/gelu.h"

namespace pluto::weight_analysis {
namespace {

TEST(PhraseProbeParsingTest, ValidZeroBasedNeuronPairsPreserveRequestOrder) {
  auto empty = ParseNeuronInterventions("");
  ASSERT_TRUE(empty.ok());
  EXPECT_TRUE(empty->empty());
  auto pairs = ParseNeuronInterventions("7:2047,0:0,3:17");
  ASSERT_TRUE(pairs.ok()) << pairs.status();
  const std::vector<std::pair<int, int>> expected{{7, 2047}, {0, 0}, {3, 17}};
  EXPECT_EQ(*pairs, expected);
}

TEST(PhraseProbeParsingTest, RejectsMalformedDuplicateAndOutOfRangePairs) {
  for (const auto* text :
       {"0", "1:2:3", ":1", "1:", "0:1,", ",0:1", "0:1,,2:3", "0:1,0:1", "-1:0",
        "8:0", "0:-1", "0:2048", "0:1.0", "x:1", "1:9999999999999"}) {
    EXPECT_EQ(ParseNeuronInterventions(text).status().code(),
              absl::StatusCode::kInvalidArgument)
        << text;
  }
}

TEST(PhraseProbeRoundingTest, Bf16AdditionIncludesRoundToEvenAndSignedZeros) {
  EXPECT_EQ(AddBf16Bits(0x3f80, 0x3f80), 0x4000);  // 1 + 1 = 2.
  EXPECT_EQ(AddBf16Bits(0x3f80, 0xbf80), 0x0000);  // Exact cancellation.
  EXPECT_EQ(AddBf16Bits(0x8000, 0x8000), 0x8000);  // -0 + -0 = -0.
  EXPECT_EQ(AddBf16Bits(0x3f80, 0x3b80), 0x3f80);  // 1 + 1/256: even tie down.
  EXPECT_EQ(AddBf16Bits(0x3f81, 0x3b80), 0x3f82);  // Odd lower endpoint: up.
  EXPECT_EQ(AddBf16Bits(0xbf80, 0xbb80), 0xbf80);  // Negative even tie.
}

TEST(PhraseProbeTopologyTest, RejectsEmptyTreeBeforeReadingAnyBuffers) {
  llm::Tape empty;
  EXPECT_EQ(ValidateGpt2Tape(empty).code(),
            absl::StatusCode::kFailedPrecondition);
  empty.children.resize(12);
  EXPECT_EQ(ValidateGpt2Tape(empty).code(),
            absl::StatusCode::kFailedPrecondition);
}

class PhraseProbeGpuTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }
  void TearDown() override {
    if (executor_) {
      EXPECT_TRUE(executor_->Synchronize().ok());
    }
  }
  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(PhraseProbeGpuTest,
       NativeProjectionReplayIsByteExactAndDoesNotChangeWeights) {
  constexpr int kRows = 16;
  constexpr int kInput = 16;
  constexpr int kOutput = 32;
  auto host = cuda::PageLockedHostArray<uint16_t>::Allocate(kRows * kInput);
  ASSERT_TRUE(host.ok()) << host.status();
  for (size_t i = 0; i < host->size(); ++i) {
    // Exactly represented BF16 values, alternating sign and varying exponent.
    (*host)[i] =
        static_cast<uint16_t>(0x3e80 + (i % 256) + ((i % 3) ? 0 : 0x8000));
  }
  auto input = cuda::Buffer::Allocate(*executor_, host->size_bytes());
  ASSERT_TRUE(input.ok()) << input.status();
  ASSERT_EQ(cudaMemcpyAsync(input->data(), host->data(), host->size_bytes(),
                            cudaMemcpyHostToDevice, executor_->stream()),
            cudaSuccess);
  auto layer = llm::FullyConnectedLayer::Create(*executor_, kInput, kOutput,
                                                llm::DataType::BF16);
  ASSERT_TRUE(layer.ok()) << layer.status();
  ASSERT_TRUE((*layer)->InitializeNormal(.1f, 17).ok());
  const auto weights = (*layer)->weights();
  auto before = ReadPrefix(*executor_, weights[0], kInput, kOutput, 4);
  ASSERT_TRUE(before.ok()) << before.status();
  llm::Tape tape;
  auto expected =
      (*layer)->fwd(*executor_, absl::MakeConstSpan(&*input, 1), &tape);
  ASSERT_TRUE(expected.ok()) << expected.status();
  auto replay = ReplayProjection(*executor_, *input, weights[0], weights[1],
                                 kInput, kOutput);
  ASSERT_TRUE(replay.ok()) << replay.status();
  auto actual_bytes = ReadPrefix(*executor_, *replay, kRows, kOutput, 2);
  auto expected_bytes = ReadPrefix(*executor_, *expected, kRows, kOutput, 2);
  ASSERT_TRUE(actual_bytes.ok()) << actual_bytes.status();
  ASSERT_TRUE(expected_bytes.ok()) << expected_bytes.status();
  EXPECT_EQ(std::memcmp(actual_bytes->data(), expected_bytes->data(),
                        actual_bytes->size_bytes()),
            0);
  auto after = ReadPrefix(*executor_, weights[0], kInput, kOutput, 4);
  ASSERT_TRUE(after.ok()) << after.status();
  EXPECT_EQ(std::memcmp(before->data(), after->data(), before->size_bytes()),
            0);
}

TEST_F(PhraseProbeGpuTest,
       ReplayedBranchMatchesProductionResidualIncludingRounding) {
  constexpr int kRows = 16;
  constexpr int kWidth = 16;
  auto host = cuda::PageLockedHostArray<uint16_t>::Allocate(kRows * kWidth);
  ASSERT_TRUE(host.ok());
  for (size_t i = 0; i < host->size(); ++i) {
    (*host)[i] = static_cast<uint16_t>(0x3e80 + i + ((i % 2) ? 0x8000 : 0));
  }
  auto input = cuda::Buffer::Allocate(*executor_, host->size_bytes());
  ASSERT_TRUE(input.ok());
  ASSERT_EQ(cudaMemcpyAsync(input->data(), host->data(), host->size_bytes(),
                            cudaMemcpyHostToDevice, executor_->stream()),
            cudaSuccess);
  auto gelu = llm::GeluLayer::Create(*executor_, llm::DataType::BF16);
  ASSERT_TRUE(gelu.ok());
  auto* branch = gelu->get();
  llm::ResidualLayer residual(std::move(*gelu));
  llm::Tape tape, branch_tape;
  auto after = residual.fwd(*executor_, absl::MakeConstSpan(&*input, 1), &tape);
  auto contribution =
      branch->fwd(*executor_, absl::MakeConstSpan(&*input, 1), &branch_tape);
  ASSERT_TRUE(after.ok()) << after.status();
  ASSERT_TRUE(contribution.ok()) << contribution.status();
  EXPECT_TRUE(VerifyResidualReplay(*executor_, *input, *contribution, *after,
                                   kRows, kWidth)
                  .ok());
  // Zeroing the entire result must be detected, not hidden by a loose epsilon.
  ASSERT_EQ(cudaMemsetAsync(after->data(), 0, after->size_bytes(),
                            executor_->stream()),
            cudaSuccess);
  EXPECT_EQ(VerifyResidualReplay(*executor_, *input, *contribution, *after,
                                 kRows, kWidth)
                .code(),
            absl::StatusCode::kDataLoss);
}

TEST_F(PhraseProbeGpuTest,
       PrefixChecksShapesExecutorsAndReturnsOnlyRequestedRows) {
  auto input = cuda::Buffer::Allocate(*executor_, 4 * 8 * sizeof(float));
  ASSERT_TRUE(input.ok());
  ASSERT_EQ(cudaMemsetAsync(input->data(), 0, input->size_bytes(),
                            executor_->stream()),
            cudaSuccess);
  auto prefix = ReadPrefix(*executor_, *input, 2, 8, 4);
  ASSERT_TRUE(prefix.ok()) << prefix.status();
  EXPECT_EQ(prefix->size_bytes(), 2 * 8 * sizeof(float));
  EXPECT_TRUE(std::all_of(prefix->begin(), prefix->end(),
                          [](uint8_t x) { return x == 0; }));
  for (const auto& [rows, width, bytes] :
       std::array<std::array<int, 3>, 7>{{{0, 8, 4},
                                          {-1, 8, 4},
                                          {5, 8, 4},
                                          {1, 0, 4},
                                          {1, 7, 4},
                                          {1, 8, 1},
                                          {1, 8, 8}}}) {
    EXPECT_EQ(
        ReadPrefix(*executor_, *input, rows, width, bytes).status().code(),
        absl::StatusCode::kInvalidArgument);
  }
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok());
  EXPECT_EQ(ReadPrefix(**other, *input, 1, 8, 4).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(PhraseProbeGpuTest,
       DeviceWeightCopyRejectsAliasingAndValidatesAllBeforeWriting) {
  auto a = cuda::Buffer::Allocate(*executor_, 16);
  auto b = cuda::Buffer::Allocate(*executor_, 16);
  auto wrong = cuda::Buffer::Allocate(*executor_, 8);
  ASSERT_TRUE(a.ok());
  ASSERT_TRUE(b.ok());
  ASSERT_TRUE(wrong.ok());
  llm::BufferVec source{*a};
  llm::BufferVec destination{*a};
  EXPECT_EQ(
      CopyDeviceWeights(*executor_, source, absl::MakeSpan(destination)).code(),
      absl::StatusCode::kInvalidArgument);
  destination = {*b, *wrong};
  EXPECT_EQ(
      CopyDeviceWeights(*executor_, source, absl::MakeSpan(destination)).code(),
      absl::StatusCode::kInvalidArgument);
  source = {*a, *a};
  ASSERT_EQ(
      cudaMemsetAsync(b->data(), 0x5a, b->size_bytes(), executor_->stream()),
      cudaSuccess);
  EXPECT_EQ(
      CopyDeviceWeights(*executor_, source, absl::MakeSpan(destination)).code(),
      absl::StatusCode::kInvalidArgument);
  auto unchanged = ReadPrefix(*executor_, *b, 1, 4, 4);
  ASSERT_TRUE(unchanged.ok());
  EXPECT_TRUE(std::all_of(unchanged->begin(), unchanged->end(),
                          [](uint8_t x) { return x == 0x5a; }));
}

TEST_F(PhraseProbeGpuTest,
       TapeTopologyChecksAllBranchesAndAttentionSavedOutput) {
  auto buffer = cuda::Buffer::Allocate(*executor_, 16);
  ASSERT_TRUE(buffer.ok());
  llm::Tape tape;
  tape.children.resize(12);
  for (int index : {0, 10, 11})
    tape.children[index].intermediates.push_back(*buffer);
  for (int block = 0; block < 8; ++block) {
    auto& body = tape.children[block + 2];
    body.children.resize(2);
    for (int branch = 0; branch < 2; ++branch) {
      auto& residual = body.children[branch];
      residual.intermediates.push_back(*buffer);
      residual.children.resize(1);
      auto& leaves = residual.children[0].children;
      leaves.resize(4);
      for (auto& leaf : leaves) leaf.intermediates.push_back(*buffer);
      if (branch == 0) leaves[2].intermediates.push_back(*buffer);
    }
  }
  EXPECT_TRUE(ValidateGpt2Tape(tape).ok());
  auto changed = tape;
  changed.children[7]
      .children[0]
      .children[0]
      .children[2]
      .intermediates.pop_back();
  EXPECT_EQ(ValidateGpt2Tape(changed).code(),
            absl::StatusCode::kFailedPrecondition);
  changed = tape;
  changed.children[9].children[1].children[0].children.pop_back();
  EXPECT_EQ(ValidateGpt2Tape(changed).code(),
            absl::StatusCode::kFailedPrecondition);
  auto lens = NativeLogitLens::Create(*executor_, {});
  EXPECT_EQ(lens.status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pluto::weight_analysis
