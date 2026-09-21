#include "src/llm/experiments/memorize_general_facts/checkpoint_validation.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

namespace pluto::llm::memorize_general_facts {
namespace {

class CheckpointValidationTest : public testing::Test {
 protected:
  void SetUp() override {
    const std::string pattern =
        (std::filesystem::path(testing::TempDir()) / "exact_checkpoint.XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const char* created = mkdtemp(writable.data());
    ASSERT_NE(created, nullptr);
    directory_ = created;
  }

  void TearDown() override {
    if (directory_.empty()) return;
    // This fixture owns only the unique directory successfully created above.
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
    EXPECT_FALSE(error) << error.message();
  }

  void File(const std::string& name) {
    std::ofstream output(directory_ / name, std::ios::binary);
    output << "weight contents are checked separately by ReadFromDirectory";
    output.close();
    ASSERT_TRUE(output);
  }

  void Weight(size_t index) { File(absl::StrCat("weight_", index, ".bin")); }

  std::filesystem::path directory_;
};

TEST_F(CheckpointValidationTest, AcceptsExactCanonicalFilesAndMetadata) {
  Weight(2);
  Weight(0);
  Weight(1);
  File("config.txt");
  File("weight_notes.txt");
  EXPECT_TRUE(ValidateExactCheckpointFiles(directory_, 3).ok());
}

TEST_F(CheckpointValidationTest, RejectsLargerDepthEvenWhenPrefixShapesMatch) {
  // Eight GPT-2 blocks have 100 unique tensors; seven have 88. The prefix's
  // tensors 86/87 are the eighth block's input norm, which have the same shapes
  // as a seven-block model's final norm. The generic prefix reader accepts
  // that mismatch, but full-model certification must reject the extra files.
  for (size_t index = 0; index < 100; ++index) Weight(index);
  EXPECT_TRUE(ValidateExactCheckpointFiles(directory_, 100).ok());
  EXPECT_EQ(ValidateExactCheckpointFiles(directory_, 88).code(),
            absl::StatusCode::kDataLoss);
}

TEST_F(CheckpointValidationTest, RejectsMissingTrailingWeight) {
  Weight(0);
  Weight(1);
  EXPECT_EQ(ValidateExactCheckpointFiles(directory_, 3).code(),
            absl::StatusCode::kDataLoss);
}

TEST_F(CheckpointValidationTest,
       SixteenBlockCheckpointCannotCertifyEightBlocks) {
  // Sixteen blocks have 196 unique tensors, versus 100 for eight blocks.
  // The generic prefix loader could mistake the ninth block's input norm for
  // the shallower final norm, so certification must enforce the full file set.
  for (size_t index = 0; index < 196; ++index)
    Weight(index);
  EXPECT_TRUE(ValidateExactCheckpointFiles(directory_, 196).ok());
  EXPECT_EQ(ValidateExactCheckpointFiles(directory_, 100).code(),
            absl::StatusCode::kDataLoss);
}

TEST_F(CheckpointValidationTest, SixteenBlockCheckpointRequiresFinalNormBias) {
  for (size_t index = 0; index < 195; ++index)
    Weight(index);
  EXPECT_EQ(ValidateExactCheckpointFiles(directory_, 196).code(),
            absl::StatusCode::kDataLoss);
  Weight(195);
  EXPECT_TRUE(ValidateExactCheckpointFiles(directory_, 196).ok());
}

TEST_F(CheckpointValidationTest, RejectsHoleDespiteMatchingFileCount) {
  Weight(0);
  Weight(2);
  Weight(3);
  EXPECT_EQ(ValidateExactCheckpointFiles(directory_, 3).code(),
            absl::StatusCode::kDataLoss);
}

TEST_F(CheckpointValidationTest, RejectsNoncanonicalNumberedAlias) {
  Weight(0);
  File("weight_00.bin");
  EXPECT_EQ(ValidateExactCheckpointFiles(directory_, 1).code(),
            absl::StatusCode::kDataLoss);
}

TEST_F(CheckpointValidationTest, RejectsWeightDirectory) {
  std::error_code error;
  ASSERT_TRUE(
      std::filesystem::create_directory(directory_ / "weight_0.bin", error));
  ASSERT_FALSE(error);
  EXPECT_EQ(ValidateExactCheckpointFiles(directory_, 1).code(),
            absl::StatusCode::kDataLoss);
}

TEST_F(CheckpointValidationTest, RejectsInvalidCheckpointPaths) {
  EXPECT_EQ(ValidateExactCheckpointFiles({}, 1).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ValidateExactCheckpointFiles(directory_ / "absent", 1).code(),
            absl::StatusCode::kNotFound);
  File("not_a_directory");
  EXPECT_EQ(
      ValidateExactCheckpointFiles(directory_ / "not_a_directory", 1).code(),
      absl::StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace pluto::llm::memorize_general_facts
