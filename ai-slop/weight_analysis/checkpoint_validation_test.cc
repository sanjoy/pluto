#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#include "gtest/gtest.h"
#include "ai-slop/weight_analysis/causal_probe.h"

namespace pluto::weight_analysis {
namespace {
namespace fs = std::filesystem;

// These layout tests neither initialize CUDA nor allocate host/device buffers.
// Sparse files have the real checkpoint sizes without writing a model's worth
// of dummy bytes. The only files removed are in this exclusive test directory.
class CheckpointValidationTest : public testing::Test {
 protected:
  void SetUp() override {
    std::string pattern =
        (fs::temp_directory_path() / "checkpoint-validation-test.XXXXXX")
            .string();
    const char* created = mkdtemp(pattern.data());
    ASSERT_NE(created, nullptr);
    root_ = created;
    checkpoint_ = root_ / "step_0";
    ASSERT_TRUE(fs::create_directory(checkpoint_));
    const auto sizes = Gpt2WeightByteSizes();
    for (size_t i = 0; i < sizes.size(); ++i) {
      const auto path = Weight(i);
      ASSERT_TRUE(std::ofstream(path).good());
      fs::resize_file(path, sizes[i]);
    }
  }

  void TearDown() override {
    if (!root_.empty()) fs::remove_all(root_);
  }

  fs::path Weight(size_t index) const {
    return checkpoint_ / ("weight_" + std::to_string(index) + ".bin");
  }

  void AddMetadata() {
    std::ofstream metadata(checkpoint_ / "patch.json");
    metadata << "{\"test_metadata\":true}\n";
    ASSERT_TRUE(metadata.good());
  }

  fs::path root_;
  fs::path checkpoint_;
};

TEST_F(CheckpointValidationTest, AcceptsOriginalAndPatchedLayouts) {
  auto plain = InspectGpt2CheckpointFiles(checkpoint_);
  ASSERT_TRUE(plain.ok()) << plain.status();
  ASSERT_EQ(plain->size(), 100U);
  for (size_t i = 0; i < plain->size(); ++i) {
    EXPECT_EQ((*plain)[i].path, Weight(i));
  }
  EXPECT_TRUE(VerifyCheckpointFilesUnchanged(*plain).ok());

  AddMetadata();
  auto patched = InspectGpt2CheckpointFiles(checkpoint_);
  ASSERT_TRUE(patched.ok()) << patched.status();
  ASSERT_EQ(patched->size(), 101U);
  EXPECT_EQ(patched->back().path, checkpoint_ / "patch.json");
  EXPECT_TRUE(VerifyCheckpointFilesUnchanged(*patched).ok());
  // Adding metadata after a snapshot is also a directory-layout change.
  EXPECT_FALSE(VerifyCheckpointFilesUnchanged(*plain).ok());
}

TEST_F(CheckpointValidationTest, MetadataDoesNotSubstituteForMissingWeight) {
  AddMetadata();
  ASSERT_TRUE(fs::remove(Weight(99)));
  // There are still 100 files, but they are not the 100 expected weights.
  EXPECT_FALSE(InspectGpt2CheckpointFiles(checkpoint_).ok());
}

TEST_F(CheckpointValidationTest, RejectsExtraWeightsAndUnknownFiles) {
  for (const auto* name : {"weight_100.bin", "weight_00.bin", "weight_bad.bin",
                           "notes.txt", "patch.json.backup"}) {
    SCOPED_TRACE(name);
    const auto extra = checkpoint_ / name;
    ASSERT_TRUE(std::ofstream(extra).good());
    EXPECT_FALSE(InspectGpt2CheckpointFiles(checkpoint_).ok());
    ASSERT_TRUE(fs::remove(extra));
  }
}

TEST_F(CheckpointValidationTest, RejectsWrongSizeEvenWithMetadata) {
  AddMetadata();
  const auto sizes = Gpt2WeightByteSizes();
  for (size_t index : {0, 20, 99}) {
    for (int adjustment : {-1, 1}) {
      SCOPED_TRACE(index);
      SCOPED_TRACE(adjustment);
      fs::resize_file(Weight(index), sizes[index] + adjustment);
      EXPECT_FALSE(InspectGpt2CheckpointFiles(checkpoint_).ok());
      fs::resize_file(Weight(index), sizes[index]);
    }
  }
}

TEST_F(CheckpointValidationTest, RejectsSymlinksAndDirectories) {
  fs::create_directory_symlink(checkpoint_, root_ / "alias");
  EXPECT_FALSE(InspectGpt2CheckpointFiles(root_ / "alias").ok());

  const auto saved = root_ / "saved_weight.bin";
  fs::rename(Weight(99), saved);
  fs::create_symlink(saved, Weight(99));
  EXPECT_FALSE(InspectGpt2CheckpointFiles(checkpoint_).ok());
  ASSERT_TRUE(fs::remove(Weight(99)));
  fs::rename(saved, Weight(99));

  for (const auto& target : {Weight(99), root_ / "missing"}) {
    fs::create_symlink(target, checkpoint_ / "patch.json");
    EXPECT_FALSE(InspectGpt2CheckpointFiles(checkpoint_).ok());
    ASSERT_TRUE(fs::remove(checkpoint_ / "patch.json"));
  }
  ASSERT_TRUE(fs::create_directory(checkpoint_ / "patch.json"));
  EXPECT_FALSE(InspectGpt2CheckpointFiles(checkpoint_).ok());
  ASSERT_TRUE(fs::remove(checkpoint_ / "patch.json"));
  ASSERT_TRUE(fs::create_directory(checkpoint_ / "nested"));
  EXPECT_FALSE(InspectGpt2CheckpointFiles(checkpoint_).ok());
}

TEST_F(CheckpointValidationTest, DetectsWeightAndMetadataStatChanges) {
  AddMetadata();
  auto snapshot = InspectGpt2CheckpointFiles(checkpoint_);
  ASSERT_TRUE(snapshot.ok()) << snapshot.status();
  for (const auto& path : {Weight(0), Weight(99), checkpoint_ / "patch.json"}) {
    const auto modified = fs::last_write_time(path);
    fs::last_write_time(path, modified + std::chrono::seconds(1));
    EXPECT_FALSE(VerifyCheckpointFilesUnchanged(*snapshot).ok());
    fs::last_write_time(path, modified);
    EXPECT_TRUE(VerifyCheckpointFilesUnchanged(*snapshot).ok());
  }
  fs::resize_file(checkpoint_ / "patch.json", 1);
  EXPECT_FALSE(VerifyCheckpointFilesUnchanged(*snapshot).ok());
}

TEST_F(CheckpointValidationTest, DetectsEntriesAddedOrRemovedAfterSnapshot) {
  auto snapshot = InspectGpt2CheckpointFiles(checkpoint_);
  ASSERT_TRUE(snapshot.ok()) << snapshot.status();
  ASSERT_TRUE(std::ofstream(checkpoint_ / "extra.bin").good());
  EXPECT_FALSE(VerifyCheckpointFilesUnchanged(*snapshot).ok());
  ASSERT_TRUE(fs::remove(checkpoint_ / "extra.bin"));
  ASSERT_TRUE(fs::remove(Weight(99)));
  EXPECT_FALSE(VerifyCheckpointFilesUnchanged(*snapshot).ok());
}

TEST_F(CheckpointValidationTest, RejectsMissingDirectoryAndEmptySnapshot) {
  EXPECT_FALSE(InspectGpt2CheckpointFiles(root_ / "absent").ok());
  EXPECT_FALSE(InspectGpt2CheckpointFiles(Weight(0)).ok());
  EXPECT_FALSE(VerifyCheckpointFilesUnchanged({}).ok());
}

}  // namespace
}  // namespace pluto::weight_analysis
