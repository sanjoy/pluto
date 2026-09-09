#include "scripts/weight_analysis/causal_probe.h"

#include <cuda_runtime_api.h>
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/util/status_macros.h"

namespace pluto::weight_analysis {
namespace {

class ProbeTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
    std::string pattern =
        (std::filesystem::temp_directory_path() / "causal-probe-test.XXXXXX")
            .string();
    char* created = mkdtemp(pattern.data());
    ASSERT_NE(created, nullptr);
    directory_ = created;
  }
  void TearDown() override {
    if (executor_) {
      EXPECT_TRUE(executor_->Synchronize().ok());
    }
    // This exclusively created test-owned directory contains only fixtures.
    if (!directory_.empty()) std::filesystem::remove_all(directory_);
  }
  absl::StatusOr<cuda::Buffer> Upload(absl::Span<const float> values) {
    ASSIGN_OR_RETURN(auto host,
                     cuda::PageLockedHostArray<float>::CopyFrom(values));
    ASSIGN_OR_RETURN(auto device,
                     cuda::Buffer::Allocate(*executor_, host.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(device.data(), host.data(), host.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "test upload"));
    RETURN_IF_ERROR(executor_->Synchronize());
    return device;
  }
  absl::StatusOr<cuda::PageLockedHostArray<float>> Download(
      const cuda::Buffer& buffer) {
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<float>::Allocate(
                                    buffer.size_bytes() / sizeof(float)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), buffer.data(), buffer.size_bytes(),
                        cudaMemcpyDeviceToHost, executor_->stream()),
        "test download"));
    RETURN_IF_ERROR(executor_->Synchronize());
    return host;
  }
  std::unique_ptr<cuda::Executor> executor_;
  std::filesystem::path directory_;
};

TEST_F(ProbeTest, FrozenArmOrderAndPhysicalByteLayout) {
  const auto arms = CausalArms();
  ASSERT_EQ(arms.size(), 35U);
  EXPECT_EQ(arms[0].name, "clean_before");
  EXPECT_EQ(arms[1].name, "clean_repeat");
  EXPECT_EQ(arms.back().name, "clean_after");
  for (int b = 0; b < 8; ++b) {
    const auto& attention = arms[2 + b * 4];
    EXPECT_EQ(attention.weight_indices,
              (std::vector<int>{6 + b * 12, 7 + b * 12}));
    EXPECT_EQ(attention.scale, 0.5f);
    EXPECT_EQ(arms[3 + b * 4].scale, 0);
    EXPECT_EQ(arms[4 + b * 4].weight_indices,
              (std::vector<int>{12 + b * 12, 13 + b * 12}));
    EXPECT_EQ(arms[5 + b * 4].scale, 0);
  }
  const auto sizes = Gpt2WeightByteSizes();
  ASSERT_EQ(sizes.size(), 100U);
  size_t total = 0;
  for (size_t size : sizes) total += size;
  EXPECT_EQ(total, 205934592U);
  EXPECT_EQ(sizes[6], 512U * 512 * 4);
  EXPECT_EQ(sizes[12], 2048U * 512 * 4);
  EXPECT_EQ(sizes[7], 512U * 4);
  EXPECT_EQ(sizes[99], 512U * 4);
}

TEST_F(ProbeTest, DedupKeepsFirstOccurrenceAndActualStorage) {
  auto a = Upload({1, 2});
  auto b = Upload({3});
  ASSERT_TRUE(a.ok());
  ASSERT_TRUE(b.ok());
  std::vector<cuda::Buffer> handles{*a, *b, *a};
  auto unique = UniqueWeights(*executor_, handles);
  ASSERT_TRUE(unique.ok());
  ASSERT_EQ(unique->size(), 2U);
  EXPECT_EQ((*unique)[0].data(), a->data());
  EXPECT_EQ((*unique)[1].data(), b->data());
  EXPECT_FALSE(ValidateGpt2Weights(*unique).ok());
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok());
  EXPECT_FALSE(UniqueWeights(**other, handles).ok());
}

TEST_F(ProbeTest, HalfAndZeroIncludeBiasAndRestoreNegativeZeroBytes) {
  const std::vector<float> matrix{1.25f, -2.5f, -0.0f, 0.0f};
  const std::vector<float> bias{3.0f, -0.0f};
  auto a = Upload(matrix);
  auto b = Upload(bias);
  ASSERT_TRUE(a.ok());
  ASSERT_TRUE(b.ok());
  // These aliases stand in for the separate handles cached in a real layer.
  auto layer_matrix = *a;
  auto layer_bias = *b;
  std::vector<cuda::Buffer> weights{*a, *b};
  auto intervention = WeightIntervention::Capture(*executor_, weights, {0, 1});
  ASSERT_TRUE(intervention.ok()) << intervention.status();
  EXPECT_FALSE((*intervention)->Apply(0.25f).ok());
  ASSERT_TRUE((*intervention)->Apply(0.5f).ok());
  auto half_matrix = Download(layer_matrix);
  auto half_bias = Download(layer_bias);
  ASSERT_TRUE(half_matrix.ok());
  ASSERT_TRUE(half_bias.ok());
  EXPECT_EQ((*half_matrix)[0], 0.625f);
  EXPECT_EQ((*half_matrix)[1], -1.25f);
  EXPECT_EQ((*half_bias)[0], 1.5f);
  EXPECT_FALSE((*intervention)->Apply(0).ok());
  ASSERT_TRUE((*intervention)->RestoreAndVerify().ok());
  ASSERT_TRUE((*intervention)->Apply(0).ok());
  auto zero_matrix = Download(layer_matrix);
  auto zero_bias = Download(layer_bias);
  ASSERT_TRUE(zero_matrix.ok());
  ASSERT_TRUE(zero_bias.ok());
  for (float x : *zero_matrix) EXPECT_EQ(x, 0);
  for (float x : *zero_bias) EXPECT_EQ(x, 0);
  ASSERT_TRUE((*intervention)->RestoreAndVerify().ok());
  auto restored_matrix = Download(layer_matrix);
  auto restored_bias = Download(layer_bias);
  ASSERT_TRUE(restored_matrix.ok());
  ASSERT_TRUE(restored_bias.ok());
  EXPECT_EQ(
      std::memcmp(restored_matrix->data(), matrix.data(), matrix.size() * 4),
      0);
  EXPECT_EQ(std::memcmp(restored_bias->data(), bias.data(), bias.size() * 4),
            0);
}

TEST_F(ProbeTest, DestructorRestoresOnEarlyExit) {
  auto a = Upload({-0.0f, 5.0f});
  ASSERT_TRUE(a.ok());
  {
    auto intervention = WeightIntervention::Capture(*executor_, {*a}, {0});
    ASSERT_TRUE(intervention.ok());
    ASSERT_TRUE((*intervention)->Apply(0).ok());
  }
  auto restored = Download(*a);
  ASSERT_TRUE(restored.ok());
  EXPECT_TRUE(std::signbit((*restored)[0]));
  EXPECT_EQ((*restored)[1], 5);
}

TEST_F(ProbeTest, InterventionRejectsAliasesInvalidIndicesAndNonfinite) {
  auto a = Upload({1});
  auto bad = Upload({std::numeric_limits<float>::quiet_NaN()});
  ASSERT_TRUE(a.ok());
  ASSERT_TRUE(bad.ok());
  EXPECT_FALSE(WeightIntervention::Capture(*executor_, {*a, *a}, {0, 1}).ok());
  EXPECT_FALSE(WeightIntervention::Capture(*executor_, {*a}, {-1}).ok());
  EXPECT_FALSE(WeightIntervention::Capture(*executor_, {*a}, {1}).ok());
  EXPECT_FALSE(WeightIntervention::Capture(*executor_, {*a}, {}).ok());
  EXPECT_FALSE(WeightIntervention::Capture(*executor_, {*bad}, {0}).ok());
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok());
  EXPECT_FALSE(WeightIntervention::Capture(**other, {*a}, {0}).ok());
}

TEST_F(ProbeTest, PackedBatchUsesGlobalTargetHalfAndRejectsBadInputs) {
  const std::vector<int32_t> values{0, 1, 2, 3, 4, 5, 6, 0, 2, 2, 2, 3,
                                    1, 2, 3, 4, 5, 6, 0, 1, 2, 2, 3, 4};
  const auto file = directory_ / "batch.bin";
  ASSERT_TRUE(WriteExclusive(file, values.data(), values.size() * 4).ok());
  auto batch = LoadPackedBatch(file, 4, 7);
  ASSERT_TRUE(batch.ok()) << batch.status();
  EXPECT_EQ(batch->passage_count, 3);
  EXPECT_EQ(batch->inputs(1, 1)[0], 4);
  EXPECT_EQ(batch->targets(1, 1)[0], 5);
  EXPECT_EQ(batch->targets(2, 1)[3], 4);
  EXPECT_FALSE(LoadPackedBatch(file, 5, 7).ok());
  EXPECT_FALSE(LoadPackedBatch(file, 4, 6).ok());
  EXPECT_FALSE(LoadPackedBatch(file, 0, 7).ok());
  auto malformed = values;
  malformed[13] = 6;
  const auto wrong_shift = directory_ / "wrong.bin";
  ASSERT_TRUE(
      WriteExclusive(wrong_shift, malformed.data(), malformed.size() * 4).ok());
  EXPECT_FALSE(LoadPackedBatch(wrong_shift, 4, 7).ok());
  const auto empty = directory_ / "empty.bin";
  ASSERT_TRUE(WriteExclusive(empty, nullptr, 0).ok());
  EXPECT_FALSE(LoadPackedBatch(empty, 4, 7).ok());
}

TEST_F(ProbeTest, ExclusiveArtifactsAndJsonEscapes) {
  const auto new_directory = directory_ / "output";
  EXPECT_TRUE(CreateNewOutputDirectory(new_directory).ok());
  EXPECT_FALSE(CreateNewOutputDirectory(new_directory).ok());
  const auto link = directory_ / "dangling";
  std::filesystem::create_symlink(directory_ / "missing", link);
  EXPECT_FALSE(CreateNewOutputDirectory(link).ok());
  const auto file = directory_ / "artifact";
  EXPECT_TRUE(WriteExclusive(file, "abc", 3).ok());
  EXPECT_FALSE(WriteExclusive(file, "xyz", 3).ok());
  EXPECT_EQ(std::filesystem::file_size(file), 3U);
  EXPECT_EQ(JsonQuote("a\"b\\c\n\t"), "\"a\\\"b\\\\c\\u000a\\u0009\"");
}

// Obvious test-only model: each row favors the input token by exactly two
// logits. It deliberately supports arbitrary row counts to catch the final
// partial microbatch and guarantees that batch grouping cannot change values.
class ToyModel final : public llm::Layer {
 public:
  ToyModel(int vocab, int padded) : vocab_(vocab), padded_(padded) {}
  absl::StatusOr<cuda::Buffer> fwd(cuda::Executor& executor,
                                   absl::Span<const cuda::Buffer> inputs,
                                   llm::Tape* tape) const override {
    if (!tape || inputs.size() != 1)
      return absl::InvalidArgumentError("toy input");
    const size_t rows = inputs[0].size_bytes() / 4;
    ASSIGN_OR_RETURN(auto tokens,
                     cuda::PageLockedHostArray<int32_t>::Allocate(rows));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(tokens.data(), inputs[0].data(), rows * 4,
                        cudaMemcpyDeviceToHost, executor.stream()),
        "toy tokens"));
    RETURN_IF_ERROR(executor.Synchronize());
    ASSIGN_OR_RETURN(auto logits, cuda::PageLockedHostArray<float>::Allocate(
                                      rows * padded_));
    for (size_t row = 0; row < rows; ++row) {
      for (int col = 0; col < padded_; ++col) {
        logits[row * padded_ + col] = col >= vocab_        ? -1e30f
                                      : col == tokens[row] ? 2.0f
                                                           : 0.0f;
      }
    }
    ASSIGN_OR_RETURN(auto result,
                     cuda::Buffer::Allocate(executor, logits.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(result.data(), logits.data(), logits.size_bytes(),
                        cudaMemcpyHostToDevice, executor.stream()),
        "toy logits"));
    RETURN_IF_ERROR(executor.Synchronize());
    return result;
  }
  absl::StatusOr<llm::BufferVec> bwd(cuda::Executor&,
                                     absl::Span<const cuda::Buffer>,
                                     llm::Tape) override {
    ADD_FAILURE() << "validation must never call backward";
    return absl::UnimplementedError("not a training model");
  }
  absl::Span<cuda::Buffer> weights() override { return {}; }
  llm::DataType output_type() const override { return llm::DataType::BF16; }

 private:
  int vocab_;
  int padded_;
};

TEST_F(ProbeTest, PerTokenLossAndArgmaxPreserveMicrobatchAndPassageOrdering) {
  const std::vector<int32_t> values{0, 1, 2, 3, 4, 5, 6, 0, 2, 2, 2, 3,
                                    1, 2, 3, 4, 5, 6, 0, 1, 2, 2, 3, 4};
  auto host = cuda::PageLockedHostArray<int32_t>::CopyFrom(values);
  ASSERT_TRUE(host.ok());
  PackedBatch batch{std::move(*host), 4, 3};
  auto loss =
      llm::CrossEntropyLossLayer::Create(*executor_, 7, llm::DataType::BF16);
  ASSERT_TRUE(loss.ok());
  ToyModel model(7, (*loss)->padded_vocab_size());
  auto measured = EvaluatePassages(*executor_, model, **loss, batch, 2, 7,
                                   (*loss)->padded_vocab_size());
  ASSERT_TRUE(measured.ok()) << measured.status();
  auto singleton = EvaluatePassages(*executor_, model, **loss, batch, 1, 7,
                                    (*loss)->padded_vocab_size());
  ASSERT_TRUE(singleton.ok()) << singleton.status();
  ASSERT_EQ(measured->losses.size(), 12U);
  for (int i = 0; i < 12; ++i) {
    EXPECT_EQ(measured->argmax[i], values[i]);
    const double expected =
        std::log(std::exp(2.0) + 6) - (values[i] == values[i + 12] ? 2 : 0);
    EXPECT_NEAR(measured->losses[i], expected, 1e-6);
    EXPECT_EQ(measured->losses[i], singleton->losses[i]);
    EXPECT_EQ(measured->argmax[i], singleton->argmax[i]);
  }
  EXPECT_FALSE(
      EvaluatePassages(*executor_, model, **loss, batch, 0, 7, 32).ok());
}

}  // namespace
}  // namespace pluto::weight_analysis
