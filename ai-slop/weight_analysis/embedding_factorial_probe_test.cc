#include "ai-slop/weight_analysis/embedding_factorial_probe.h"

#include <unistd.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace pluto::weight_analysis {
namespace {

// Deliberately CPU-only: linking native probe helpers must not cause these
// validation tests to initialize CUDA or interfere with a live training run.
TEST(EmbeddingFactorialCpuTest, RowsSupportThreeTokensAndFollowingBoundary) {
  EXPECT_TRUE(ValidateFactorialRows({127, 128, 129, 2, 3, 4}, 2, 3, 1024).ok());
  EXPECT_TRUE(ValidateFactorialRows({127, 128, 129, 130}, 1, 4, 1024).ok());
  EXPECT_TRUE(ValidateFactorialRows({0, 512, 1023}, 1, 3, 1024).ok());
  EXPECT_TRUE(ValidateFactorialRows({0}, 1, 1, 1).ok());
}

TEST(EmbeddingFactorialCpuTest, RejectsAmbiguousOrOutOfBoundsRows) {
  for (const auto& rows : std::vector<std::vector<int32_t>>{
           {}, {-1, 0, 1}, {0, 1, 1024}, {2, 1, 3}, {1, 1, 2}, {0, 1}}) {
    EXPECT_FALSE(ValidateFactorialRows(rows, 1, 3, 1024).ok());
  }
  EXPECT_FALSE(ValidateFactorialRows({0}, 0, 1, 1024).ok());
  EXPECT_FALSE(ValidateFactorialRows({0}, 1, 0, 1024).ok());
  EXPECT_FALSE(ValidateFactorialRows({0}, 1, 1, 0).ok());
  EXPECT_FALSE(ValidateFactorialRows({0, 1}, 1, 2, 1).ok());
  EXPECT_FALSE(ValidateFactorialRows({0}, std::numeric_limits<int>::max(),
                                     std::numeric_limits<int>::max(),
                                     std::numeric_limits<int>::max())
                   .ok());
}

TEST(EmbeddingFactorialCpuTest, ScoresAllVocabularyAndBreaksTiesByTokenId) {
  for (int target = 0; target < 3; ++target) {
    auto score = ScoreFactorialToken({0, 0, 0}, target);
    ASSERT_TRUE(score.ok());
    EXPECT_DOUBLE_EQ(score->nll, std::log(3.0));
    EXPECT_EQ(score->argmax, 0);
    EXPECT_EQ(score->target_rank, target + 1);
  }
  auto score = ScoreFactorialToken({0, 0, 2}, 0);
  ASSERT_TRUE(score.ok());
  EXPECT_NEAR(score->nll, std::log(2 + std::exp(2.0)), 1e-14);
  EXPECT_EQ(score->argmax, 2);
  EXPECT_EQ(score->target_rank, 2);
}

TEST(EmbeddingFactorialCpuTest, StableNllAtExtremeFiniteLogits) {
  const float largest = std::numeric_limits<float>::max();
  auto score = ScoreFactorialToken({largest, -largest}, 1);
  ASSERT_TRUE(score.ok());
  EXPECT_DOUBLE_EQ(score->nll, 2.0 * largest);
  EXPECT_EQ(score->argmax, 0);
  EXPECT_EQ(score->target_rank, 2);
  auto singleton = ScoreFactorialToken({-largest}, 0);
  ASSERT_TRUE(singleton.ok());
  EXPECT_EQ(singleton->nll, 0);
}

TEST(EmbeddingFactorialCpuTest, RejectsNonfiniteLogitsAndInvalidTargets) {
  EXPECT_FALSE(ScoreFactorialToken({}, 0).ok());
  EXPECT_FALSE(ScoreFactorialToken({0}, -1).ok());
  EXPECT_FALSE(ScoreFactorialToken({0}, 1).ok());
  for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity()}) {
    EXPECT_FALSE(ScoreFactorialToken({0, bad}, 0).ok());
    EXPECT_FALSE(ScoreFactorialToken({bad, 0}, 1).ok());
  }
}

TEST(EmbeddingFactorialCpuTest, ChangedRowsAreExactAndIncludeSignedZero) {
  const std::array<float, 8> original{0, 1, 2, 3, 0, 0, 0, 0};
  auto patched = original;
  auto same = ChangedEmbeddingRows(original, patched, 3, 2);
  ASSERT_TRUE(same.ok());
  EXPECT_TRUE(same->empty());
  patched[2] = 7;
  patched[4] = -0.0f;
  auto changed = ChangedEmbeddingRows(original, patched, 3, 2);
  ASSERT_TRUE(changed.ok());
  EXPECT_EQ(*changed, (std::vector<int>{1, 2}));
  patched[6] = -0.0f;
  EXPECT_FALSE(ChangedEmbeddingRows(original, patched, 3, 2).ok());
}

TEST(EmbeddingFactorialCpuTest,
     RejectsInvalidEmbeddingGeometryAndNonfiniteData) {
  EXPECT_FALSE(ChangedEmbeddingRows({0, 0}, {0}, 1, 2).ok());
  EXPECT_FALSE(ChangedEmbeddingRows({0}, {0}, 1, 2).ok());
  EXPECT_FALSE(ChangedEmbeddingRows({0}, {0}, 2, 1).ok());
  EXPECT_FALSE(ChangedEmbeddingRows({}, {}, 1, 1).ok());
  EXPECT_FALSE(ChangedEmbeddingRows({0}, {0}, 0, 1).ok());
  EXPECT_FALSE(ChangedEmbeddingRows({0}, {0}, 1, 0).ok());
  const float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(ChangedEmbeddingRows({nan}, {0}, 1, 1).ok());
  EXPECT_FALSE(ChangedEmbeddingRows({0}, {nan}, 1, 1).ok());
  EXPECT_FALSE(ChangedEmbeddingRows({0, nan}, {0, nan}, 1, 1).ok());
}

class EmbeddingFactorialRowsFileTest : public testing::Test {
 protected:
  void SetUp() override {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "factorial-rows-test.XXXXXX")
            .string();
    const char* root = mkdtemp(pattern.data());
    ASSERT_NE(root, nullptr);
    directory_ = root;
  }
  void TearDown() override {
    // Delete only this test's exclusively created fixtures.
    if (!directory_.empty()) std::filesystem::remove_all(directory_);
  }
  std::filesystem::path directory_;
};

TEST_F(EmbeddingFactorialRowsFileTest, ReadsExactMetadataWithoutCuda) {
  const auto path = directory_ / "rows.i32";
  const std::array<int32_t, 6> rows{127, 128, 129, 4, 5, 6};
  {
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(rows.data()), sizeof(rows));
    ASSERT_TRUE(stream.good());
  }
  auto result = LoadFactorialRows(path, 2, 3, 1024);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(*result, (std::vector<int32_t>(rows.begin(), rows.end())));
  EXPECT_FALSE(LoadFactorialRows(path, 1, 3, 1024).ok());
  EXPECT_FALSE(LoadFactorialRows(path, 2, 3, 128).ok());
  EXPECT_FALSE(LoadFactorialRows(path, 2, 0, 1024).ok());
  EXPECT_FALSE(LoadFactorialRows(path, 2000000, 3, 1024).ok());
  std::filesystem::create_symlink(path, directory_ / "link");
  EXPECT_FALSE(LoadFactorialRows(directory_ / "link", 2, 3, 1024).ok());
  EXPECT_FALSE(LoadFactorialRows(directory_, 2, 3, 1024).ok());
  EXPECT_FALSE(LoadFactorialRows(directory_ / "missing", 2, 3, 1024).ok());
  std::filesystem::resize_file(path, sizeof(rows) - 1);
  EXPECT_FALSE(LoadFactorialRows(path, 2, 3, 1024).ok());
}

}  // namespace
}  // namespace pluto::weight_analysis
