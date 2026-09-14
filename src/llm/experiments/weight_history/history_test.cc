#include "src/llm/experiments/weight_history/history.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::llm::weight_history {
namespace {

class WeightHistoryTest : public testing::Test {
 protected:
  void SetUp() override {
    const std::string pattern =
        (std::filesystem::path(testing::TempDir()) / "weight_history.XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const char* created = mkdtemp(writable.data());
    ASSERT_NE(created, nullptr);
    directory_ = created;
  }

  void TearDown() override {
    // Only remove the unique fixture this test successfully created.
    if (directory_.empty())
      return;
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
    EXPECT_FALSE(error) << error.message();
  }

  void Directory(const std::string& relative) {
    std::error_code error;
    std::filesystem::create_directories(directory_ / relative, error);
    ASSERT_FALSE(error) << error.message();
  }

  void Bytes(const std::string& relative, const std::string& bytes) {
    std::ofstream output(directory_ / relative, std::ios::binary);
    output.write(bytes.data(), bytes.size());
    ASSERT_TRUE(output.good());
  }

  void Weight(const std::string& step, const std::string& weight,
              const std::vector<float>& values) {
    Directory(step);
    std::ofstream output(directory_ / step / weight, std::ios::binary);
    output.write(reinterpret_cast<const char*>(values.data()),
                 values.size() * sizeof(float));
    ASSERT_TRUE(output.good());
  }

  auto Analyze(size_t chunk_elements = 262144) {
    return AnalyzeDirectory(directory_, [](size_t, size_t) {}, chunk_elements);
  }

  std::filesystem::path directory_;
};

TEST_F(WeightHistoryTest, ComputesDeltasAndDriftInNumericCheckpointOrder) {
  // Deliberately create checkpoints and sparse tensor IDs in nonnumeric order.
  // Four elements cross chunk boundaries when the chunk size is three.
  Weight("step_1000", "weight_10.bin", {2});
  Weight("step_1000", "weight_2.bin", {0, 4, 4, 3});
  Weight("step_2", "weight_10.bin", {2});
  Weight("step_2", "weight_2.bin", {3, 4, 0, 0});
  Weight("step_10", "weight_10.bin", {2});
  Weight("step_10", "weight_2.bin", {0, 4, 0, 3});

  std::vector<std::pair<size_t, size_t>> progress;
  auto result = AnalyzeDirectory(
      directory_,
      [&progress](size_t completed, size_t total) {
        progress.emplace_back(completed, total);
      },
      3);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->steps, (std::vector<int64_t>{2, 10, 1000}));
  EXPECT_EQ(progress, (std::vector<std::pair<size_t, size_t>>{{1, 2}, {2, 2}}));
  ASSERT_EQ(result->tensors.size(), 2);
  const auto& tensor = result->tensors[0];
  EXPECT_EQ(tensor.weight_id, 2);
  EXPECT_EQ(tensor.element_count, 4);
  ASSERT_EQ(tensor.samples.size(), 3);

  const auto& first = tensor.samples[0];
  EXPECT_EQ(first.step, 2);
  EXPECT_DOUBLE_EQ(first.l2, 5);
  EXPECT_DOUBLE_EQ(first.rms, 2.5);
  EXPECT_DOUBLE_EQ(first.from_first_rms, 0);
  EXPECT_FALSE(first.delta_rms.has_value());
  EXPECT_FALSE(first.delta_l2.has_value());
  EXPECT_FALSE(first.relative_l2.has_value());
  EXPECT_FALSE(first.max_abs_delta.has_value());
  EXPECT_FALSE(first.changed_fraction.has_value());

  const auto& second = tensor.samples[1];
  EXPECT_EQ(second.step, 10);
  EXPECT_DOUBLE_EQ(second.l2, 5);
  EXPECT_DOUBLE_EQ(second.rms, 2.5);
  ASSERT_TRUE(second.delta_l2.has_value());
  EXPECT_DOUBLE_EQ(*second.delta_l2, std::sqrt(18.0));
  ASSERT_TRUE(second.delta_rms.has_value());
  EXPECT_DOUBLE_EQ(*second.delta_rms, std::sqrt(18.0) / 2);
  ASSERT_TRUE(second.relative_l2.has_value());
  EXPECT_DOUBLE_EQ(*second.relative_l2, std::sqrt(18.0) / 5);
  ASSERT_TRUE(second.max_abs_delta.has_value());
  EXPECT_DOUBLE_EQ(*second.max_abs_delta, 3);
  ASSERT_TRUE(second.changed_fraction.has_value());
  EXPECT_DOUBLE_EQ(*second.changed_fraction, 0.5);
  EXPECT_DOUBLE_EQ(second.from_first_rms, std::sqrt(18.0) / 2);

  const auto& third = tensor.samples[2];
  EXPECT_EQ(third.step, 1000);
  EXPECT_DOUBLE_EQ(third.l2, std::sqrt(41.0));
  EXPECT_DOUBLE_EQ(third.rms, std::sqrt(41.0) / 2);
  ASSERT_TRUE(third.delta_l2.has_value());
  EXPECT_DOUBLE_EQ(*third.delta_l2, 4);
  ASSERT_TRUE(third.delta_rms.has_value());
  EXPECT_DOUBLE_EQ(*third.delta_rms, 2);
  ASSERT_TRUE(third.relative_l2.has_value());
  EXPECT_DOUBLE_EQ(*third.relative_l2, 0.8);
  ASSERT_TRUE(third.max_abs_delta.has_value());
  EXPECT_DOUBLE_EQ(*third.max_abs_delta, 4);
  ASSERT_TRUE(third.changed_fraction.has_value());
  EXPECT_DOUBLE_EQ(*third.changed_fraction, 0.25);
  EXPECT_DOUBLE_EQ(third.from_first_rms, std::sqrt(34.0) / 2);

  const auto& constant = result->tensors[1];
  EXPECT_EQ(constant.weight_id, 10);
  EXPECT_EQ(constant.element_count, 1);
  ASSERT_EQ(constant.samples.size(), 3);
  for (size_t index = 1; index < constant.samples.size(); ++index) {
    const auto& sample = constant.samples[index];
    EXPECT_EQ(sample.delta_l2, 0);
    EXPECT_EQ(sample.delta_rms, 0);
    EXPECT_EQ(sample.relative_l2, 0);
    EXPECT_EQ(sample.changed_fraction, 0);
    EXPECT_EQ(sample.max_abs_delta, 0);
    EXPECT_DOUBLE_EQ(sample.from_first_rms, 0);
  }
}

TEST_F(WeightHistoryTest, ZeroPreviousNormIsUndefinedAndSignedZeroIsUnchanged) {
  Weight("step_0", "weight_0.bin", {-0.0f, 0.0f});
  Weight("step_1", "weight_0.bin", {0.0f, -0.0f});
  Weight("step_2", "weight_0.bin", {3, 4});
  Weight("step_3", "weight_0.bin", {6, 8});
  auto result = Analyze(2);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->tensors.size(), 1);
  const auto& samples = result->tensors[0].samples;
  ASSERT_EQ(samples.size(), 4);
  EXPECT_FALSE(samples[1].relative_l2.has_value());
  EXPECT_EQ(samples[1].changed_fraction, 0);
  EXPECT_EQ(samples[1].delta_l2, 0);
  EXPECT_FALSE(samples[2].relative_l2.has_value());
  EXPECT_EQ(samples[2].changed_fraction, 1);
  EXPECT_EQ(samples[2].delta_l2, 5);
  EXPECT_EQ(samples[3].relative_l2, 1);
  EXPECT_EQ(samples[3].delta_l2, 5);
}

TEST_F(WeightHistoryTest, PromotesValuesBeforeSubtractionAndSquaring) {
  const float largest = std::numeric_limits<float>::max();
  Weight("step_0", "weight_0.bin", {largest, -largest, largest});
  Weight("step_1", "weight_0.bin", {-largest, largest, -largest});
  auto result = Analyze(2);
  ASSERT_TRUE(result.ok()) << result.status();
  const auto& samples = result->tensors[0].samples;
  const double magnitude = static_cast<double>(largest);
  EXPECT_TRUE(std::isfinite(samples[0].l2));
  EXPECT_DOUBLE_EQ(samples[0].rms, magnitude);
  EXPECT_DOUBLE_EQ(samples[0].l2, magnitude * std::sqrt(3.0));
  ASSERT_TRUE(samples[1].delta_l2.has_value());
  EXPECT_TRUE(std::isfinite(*samples[1].delta_l2));
  EXPECT_DOUBLE_EQ(*samples[1].delta_l2, 2 * magnitude * std::sqrt(3.0));
  ASSERT_TRUE(samples[1].delta_rms.has_value());
  EXPECT_DOUBLE_EQ(*samples[1].delta_rms, 2 * magnitude);
  EXPECT_DOUBLE_EQ(samples[1].from_first_rms, 2 * magnitude);
  EXPECT_EQ(samples[1].relative_l2, 2);
  EXPECT_EQ(samples[1].max_abs_delta, 2 * magnitude);
}

TEST_F(WeightHistoryTest, AcceptsOneCheckpointAndIgnoresUnrelatedEntries) {
  Weight("step_0007", "weight_0003.bin", {1, 2, 3, 4, 5});
  Directory("unrelated");
  Directory("step_8.bak");
  Directory("step_-2");
  Directory("step_+2");
  Directory("step_");
  Bytes("step_9.tar.gz", "archive is not an unpacked checkpoint");
  Bytes("step_0007/notes.txt", "ignored");
  Bytes("step_0007/weight_-1.bin", "ignored");
  Bytes("step_0007/weight_4.bin.bak", "ignored");
  auto result = Analyze(2);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->steps, (std::vector<int64_t>{7}));
  ASSERT_EQ(result->tensors.size(), 1);
  EXPECT_EQ(result->tensors[0].weight_id, 3);
  EXPECT_EQ(result->tensors[0].element_count, 5);
  ASSERT_EQ(result->tensors[0].samples.size(), 1);
  const auto& sample = result->tensors[0].samples[0];
  EXPECT_DOUBLE_EQ(sample.l2, std::sqrt(55.0));
  EXPECT_DOUBLE_EQ(sample.rms, std::sqrt(11.0));
  EXPECT_DOUBLE_EQ(sample.from_first_rms, 0);
  EXPECT_FALSE(sample.delta_rms.has_value());
}

TEST_F(WeightHistoryTest, AcceptsLargestSignedStepAndWeightIdentifiers) {
  Weight("step_9223372036854775807", "weight_9223372036854775807.bin", {1});
  auto result = Analyze();
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->steps,
            (std::vector<int64_t>{std::numeric_limits<int64_t>::max()}));
  EXPECT_EQ(result->tensors[0].weight_id, std::numeric_limits<int64_t>::max());
}

TEST_F(WeightHistoryTest, RejectsEmptyMissingAndNondirectoryInputs) {
  EXPECT_FALSE(Analyze().ok());
  EXPECT_FALSE(
      AnalyzeDirectory(directory_ / "missing", [](size_t, size_t) {}).ok());
  Bytes("ordinary_file", "not a directory");
  EXPECT_FALSE(
      AnalyzeDirectory(directory_ / "ordinary_file", [](size_t, size_t) {
      }).ok());
}

TEST_F(WeightHistoryTest, RejectsZeroChunkSize) {
  Weight("step_0", "weight_0.bin", {1});
  EXPECT_FALSE(Analyze(0).ok());
}

TEST_F(WeightHistoryTest, RejectsChunkSizeThatOverflowsStreamReadSize) {
  Weight("step_0", "weight_0.bin", {1});
  EXPECT_FALSE(Analyze(std::numeric_limits<size_t>::max()).ok());
}

TEST_F(WeightHistoryTest, DetectsWeightShrinkingAfterDiscovery) {
  Weight("step_0", "weight_0.bin", {1});
  Weight("step_0", "weight_1.bin", {1, 2, 3});
  auto result = AnalyzeDirectory(
      directory_,
      [this](size_t completed, size_t) {
        // Discovery happens before analysis. The callback gives us a
        // deterministic way to exercise a short read without a racing thread.
        if (completed == 1)
          Weight("step_0", "weight_1.bin", {1});
      },
      2);
  EXPECT_FALSE(result.ok());
}

TEST_F(WeightHistoryTest, DetectsWeightGrowingAfterDiscovery) {
  Weight("step_0", "weight_0.bin", {1});
  Weight("step_0", "weight_1.bin", {1, 2, 3});
  auto result = AnalyzeDirectory(
      directory_,
      [this](size_t completed, size_t) {
        // Reading the original element count must not silently ignore bytes
        // appended after the manifest was validated.
        if (completed == 1)
          Weight("step_0", "weight_1.bin", {1, 2, 3, 4});
      },
      2);
  EXPECT_FALSE(result.ok());
}

TEST_F(WeightHistoryTest, RejectsDuplicateNumericCheckpointAliases) {
  Weight("step_1", "weight_0.bin", {1});
  Weight("step_01", "weight_0.bin", {1});
  EXPECT_FALSE(Analyze().ok());
}

TEST_F(WeightHistoryTest, RejectsOverflowingCheckpointIdentifier) {
  Weight("step_9223372036854775808", "weight_0.bin", {1});
  EXPECT_FALSE(Analyze().ok());
}

TEST_F(WeightHistoryTest, RejectsDuplicateNumericWeightAliases) {
  Weight("step_1", "weight_1.bin", {1});
  Weight("step_1", "weight_01.bin", {1});
  EXPECT_FALSE(Analyze().ok());
}

TEST_F(WeightHistoryTest, RejectsOverflowingWeightIdentifier) {
  Weight("step_0", "weight_9223372036854775808.bin", {1});
  EXPECT_FALSE(Analyze().ok());
}

TEST_F(WeightHistoryTest, RejectsCheckpointWithNoWeightFiles) {
  Directory("step_0");
  Bytes("step_0/README", "not a tensor");
  EXPECT_FALSE(Analyze().ok());
}

TEST_F(WeightHistoryTest, RejectsDirectoryWithWeightFilename) {
  Weight("step_0", "weight_0.bin", {1});
  Directory("step_0/weight_1.bin");
  EXPECT_FALSE(Analyze().ok());
}

TEST_F(WeightHistoryTest, RejectsChangedWeightIndexSetEvenWithSameCount) {
  Weight("step_0", "weight_0.bin", {1});
  Weight("step_0", "weight_2.bin", {1});
  Weight("step_1", "weight_0.bin", {1});
  Weight("step_1", "weight_3.bin", {1});
  EXPECT_FALSE(Analyze().ok());
}

TEST_F(WeightHistoryTest, RejectsMissingWeightInLaterCheckpoint) {
  Weight("step_0", "weight_0.bin", {1});
  Weight("step_0", "weight_1.bin", {1});
  Weight("step_1", "weight_0.bin", {1});
  EXPECT_FALSE(Analyze().ok());
}

TEST_F(WeightHistoryTest, RejectsExtraWeightInLaterCheckpoint) {
  Weight("step_0", "weight_0.bin", {1});
  Weight("step_1", "weight_0.bin", {1});
  Weight("step_1", "weight_1.bin", {1});
  EXPECT_FALSE(Analyze().ok());
}

TEST_F(WeightHistoryTest, RejectsChangedWeightSize) {
  Weight("step_0", "weight_0.bin", {1, 2});
  Weight("step_1", "weight_0.bin", {1, 2, 3});
  EXPECT_FALSE(Analyze().ok());
}

TEST_F(WeightHistoryTest, RejectsEmptyWeight) {
  Directory("step_0");
  Bytes("step_0/weight_0.bin", "");
  EXPECT_FALSE(Analyze().ok());
}

TEST_F(WeightHistoryTest, RejectsPartialFloatWeight) {
  Directory("step_0");
  Bytes("step_0/weight_0.bin", "abcde");
  EXPECT_FALSE(Analyze().ok());
}

TEST_F(WeightHistoryTest, RejectsNonfiniteValuesAcrossCheckpointsAndChunks) {
  // Exercise both the reference checkpoint and later checkpoints, and both
  // sides of a chunk boundary. Nonfinite data must not leak into HTML metrics.
  for (const float invalid : {std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()}) {
    for (int checkpoint = 0; checkpoint < 2; ++checkpoint) {
      for (size_t index : {size_t{0}, size_t{3}}) {
        SCOPED_TRACE(testing::Message()
                     << "invalid=" << invalid << " checkpoint=" << checkpoint
                     << " index=" << index);
        std::vector<float> first = {1, 2, 3, 4};
        std::vector<float> second = {2, 3, 4, 5};
        (checkpoint == 0 ? first : second)[index] = invalid;
        Weight("step_0", "weight_0.bin", first);
        Weight("step_1", "weight_0.bin", second);
        EXPECT_FALSE(Analyze(2).ok());
      }
    }
  }
}

}  // namespace
}  // namespace pluto::llm::weight_history
