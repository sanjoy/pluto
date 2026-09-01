#include "src/llm/checkpoint.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/llm/layer.h"

namespace pluto::llm {
namespace {

class CheckpointLayer final : public Layer {
 public:
  explicit CheckpointLayer(BufferVec weights) : weights_(std::move(weights)) {}

  absl::StatusOr<Buffer> fwd(cuda::Executor&, absl::Span<const Buffer>,
                             Tape*) const override {
    return absl::UnimplementedError("CheckpointLayer has no data path");
  }

  absl::StatusOr<BufferVec> bwd(cuda::Executor&, absl::Span<const Buffer>,
                                Tape) override {
    return absl::UnimplementedError("CheckpointLayer has no data path");
  }

  absl::Span<Buffer> weights() override { return absl::MakeSpan(weights_); }

  DataType output_type() const override { return DataType::FP16; }

 private:
  BufferVec weights_;
};

class CheckpointTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }

  void TearDown() override {
    if (executor_ == nullptr) return;
    EXPECT_TRUE(executor_->Synchronize().ok());
    executor_.reset();
  }

  std::vector<unsigned char> Pattern(size_t size, unsigned char seed) {
    std::vector<unsigned char> result(size);
    for (size_t index = 0; index < size; ++index) {
      result[index] = static_cast<unsigned char>(seed + index * 17);
    }
    return result;
  }

  void CopyToDevice(Buffer& buffer,
                    const std::vector<unsigned char>& contents) {
    ASSERT_EQ(buffer.size_bytes(), contents.size());
    ASSERT_EQ(cudaMemcpyAsync(buffer.data(), contents.data(), contents.size(),
                              cudaMemcpyHostToDevice, executor_->stream()),
              cudaSuccess);
    ASSERT_TRUE(executor_->Synchronize().ok());
  }

  std::vector<unsigned char> CopyFromDevice(const Buffer& buffer) {
    std::vector<unsigned char> result(buffer.size_bytes());
    EXPECT_EQ(cudaMemcpyAsync(result.data(), buffer.data(), result.size(),
                              cudaMemcpyDeviceToHost, executor_->stream()),
              cudaSuccess);
    EXPECT_TRUE(executor_->Synchronize().ok());
    return result;
  }

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(CheckpointTest, RoundTripsUniqueWeightsAndRemovesStaleWeights) {
  auto first = Buffer::Allocate(*executor_, 37);
  auto second = Buffer::Allocate(*executor_, 65);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  CheckpointLayer layer(BufferVec{*first, *second, *first});
  const std::vector<unsigned char> first_contents = Pattern(37, 3);
  const std::vector<unsigned char> second_contents = Pattern(65, 11);
  CopyToDevice(layer.weights()[0], first_contents);
  CopyToDevice(layer.weights()[1], second_contents);

  const std::filesystem::path directory =
      std::filesystem::path(testing::TempDir()) / "checkpoint-round-trip";
  ASSERT_TRUE(std::filesystem::create_directories(directory));
  std::ofstream(directory / "weight_99.bin", std::ios::binary) << "stale";

  const Layer& const_layer = layer;
  ASSERT_TRUE(WriteToDirectory(*executor_, const_layer, directory).ok());
  EXPECT_EQ(std::filesystem::file_size(directory / "weight_0.bin"), 37u);
  EXPECT_EQ(std::filesystem::file_size(directory / "weight_1.bin"), 65u);
  EXPECT_FALSE(std::filesystem::exists(directory / "weight_2.bin"));
  EXPECT_FALSE(std::filesystem::exists(directory / "weight_99.bin"));
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(directory),
                          std::filesystem::directory_iterator()),
            2);

  ASSERT_EQ(
      cudaMemsetAsync(layer.weights()[0].data(), 0,
                      layer.weights()[0].size_bytes(), executor_->stream()),
      cudaSuccess);
  ASSERT_EQ(
      cudaMemsetAsync(layer.weights()[1].data(), 0,
                      layer.weights()[1].size_bytes(), executor_->stream()),
      cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());

  ASSERT_TRUE(ReadFromDirectory(*executor_, layer, directory).ok());
  EXPECT_EQ(CopyFromDevice(layer.weights()[0]), first_contents);
  EXPECT_EQ(CopyFromDevice(layer.weights()[1]), second_contents);
  EXPECT_EQ(layer.weights()[0].data(), layer.weights()[2].data());
}

TEST_F(CheckpointTest, RejectsWrongFileSizeBeforeModifyingAnyWeight) {
  auto first = Buffer::Allocate(*executor_, 32);
  auto second = Buffer::Allocate(*executor_, 48);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  CheckpointLayer layer(BufferVec{*first, *second});
  CopyToDevice(layer.weights()[0], Pattern(32, 5));
  CopyToDevice(layer.weights()[1], Pattern(48, 7));

  const std::filesystem::path directory =
      std::filesystem::path(testing::TempDir()) / "checkpoint-wrong-size";
  ASSERT_TRUE(WriteToDirectory(*executor_, layer, directory).ok());
  std::filesystem::resize_file(directory / "weight_1.bin", 47);

  ASSERT_EQ(
      cudaMemsetAsync(layer.weights()[0].data(), 0x7b,
                      layer.weights()[0].size_bytes(), executor_->stream()),
      cudaSuccess);
  ASSERT_EQ(
      cudaMemsetAsync(layer.weights()[1].data(), 0x7b,
                      layer.weights()[1].size_bytes(), executor_->stream()),
      cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());

  const absl::Status status = ReadFromDirectory(*executor_, layer, directory);
  EXPECT_EQ(status.code(), absl::StatusCode::kDataLoss);
  EXPECT_EQ(CopyFromDevice(layer.weights()[0]),
            std::vector<unsigned char>(32, 0x7b));
  EXPECT_EQ(CopyFromDevice(layer.weights()[1]),
            std::vector<unsigned char>(48, 0x7b));
}

TEST_F(CheckpointTest, RejectsWrongWeightCountAndExecutor) {
  auto weight = Buffer::Allocate(*executor_, 24);
  ASSERT_TRUE(weight.ok()) << weight.status();
  CheckpointLayer layer(BufferVec{*weight});
  CopyToDevice(layer.weights()[0], Pattern(24, 13));

  const std::filesystem::path directory =
      std::filesystem::path(testing::TempDir()) / "checkpoint-wrong-count";
  ASSERT_TRUE(WriteToDirectory(*executor_, layer, directory).ok());
  std::ofstream(directory / "weight_1.bin", std::ios::binary) << "extra";
  EXPECT_EQ(ReadFromDirectory(*executor_, layer, directory).code(),
            absl::StatusCode::kDataLoss);

  auto other_executor = cuda::Executor::Create();
  ASSERT_TRUE(other_executor.ok()) << other_executor.status();
  EXPECT_EQ(WriteToDirectory(**other_executor, layer, directory).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ReadFromDirectory(**other_executor, layer, directory).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE((*other_executor)->Synchronize().ok());
}

TEST_F(CheckpointTest, ReadLatestFallsBackFromMalformedCheckpoint) {
  auto weight = Buffer::Allocate(*executor_, 24);
  ASSERT_TRUE(weight.ok()) << weight.status();
  CheckpointLayer layer(BufferVec{*weight});
  const std::vector<unsigned char> expected = Pattern(24, 19);
  CopyToDevice(layer.weights()[0], expected);

  const std::filesystem::path parent =
      std::filesystem::path(testing::TempDir()) / "checkpoint-fallback";
  ASSERT_TRUE(WriteToDirectory(*executor_, layer, parent / "step_7").ok());
  ASSERT_TRUE(std::filesystem::create_directories(parent / "step_8"));
  CopyToDevice(layer.weights()[0], std::vector<unsigned char>(24, 0));

  std::vector<int> warned_steps;
  std::vector<absl::StatusCode> warning_codes;
  auto loaded = ReadLatestCheckpoint(
      *executor_, layer, parent,
      [&](const CheckpointInfo& malformed, const absl::Status& status) {
        warned_steps.push_back(malformed.step);
        warning_codes.push_back(status.code());
      });

  ASSERT_TRUE(loaded.ok()) << loaded.status();
  EXPECT_EQ(loaded->step, 7);
  EXPECT_EQ(loaded->directory, parent / "step_7");
  EXPECT_EQ(warned_steps, std::vector<int>{8});
  EXPECT_EQ(warning_codes,
            std::vector<absl::StatusCode>{absl::StatusCode::kDataLoss});
  EXPECT_EQ(CopyFromDevice(layer.weights()[0]), expected);
}

TEST(CheckpointDirectoryTest, FindsNumericallyLatestStepDirectory) {
  const std::filesystem::path parent =
      std::filesystem::path(testing::TempDir()) / "checkpoint-latest";
  ASSERT_TRUE(std::filesystem::create_directories(parent / "step_9"));
  ASSERT_TRUE(std::filesystem::create_directories(parent / "step_570"));
  ASSERT_TRUE(std::filesystem::create_directories(parent / "step_00042"));
  ASSERT_TRUE(std::filesystem::create_directories(parent / "notes"));
  std::ofstream(parent / "step_999") << "not a directory";

  auto latest = FindLatestCheckpoint(parent);
  ASSERT_TRUE(latest.ok()) << latest.status();
  EXPECT_EQ(latest->step, 570);
  EXPECT_EQ(latest->directory, parent / "step_570");

  auto inspected = InspectCheckpointDirectory(parent / "step_00042");
  ASSERT_TRUE(inspected.ok()) << inspected.status();
  EXPECT_EQ(inspected->step, 42);
  EXPECT_EQ(inspected->directory, parent / "step_00042");

  const std::filesystem::path trailing_separator(
      (parent / "step_00042").string() + "/");
  auto trailing = InspectCheckpointDirectory(trailing_separator);
  ASSERT_TRUE(trailing.ok()) << trailing.status();
  EXPECT_EQ(trailing->step, 42);
}

TEST(CheckpointDirectoryTest, RejectsMissingMalformedAndEmptyParents) {
  const std::filesystem::path root =
      std::filesystem::path(testing::TempDir()) / "checkpoint-invalid";
  const std::filesystem::path empty_parent = root / "empty";
  const std::filesystem::path malformed = root / "step_bad";
  const std::filesystem::path overflow = root / "step_2147483648";
  ASSERT_TRUE(std::filesystem::create_directories(empty_parent / "notes"));
  ASSERT_TRUE(std::filesystem::create_directories(malformed));
  ASSERT_TRUE(std::filesystem::create_directories(overflow));
  std::ofstream(empty_parent / "step_700") << "not a directory";

  EXPECT_EQ(FindLatestCheckpoint(root / "missing").status().code(),
            absl::StatusCode::kNotFound);
  EXPECT_EQ(FindLatestCheckpoint(empty_parent).status().code(),
            absl::StatusCode::kNotFound);
  EXPECT_EQ(InspectCheckpointDirectory(malformed).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(InspectCheckpointDirectory(overflow).status().code(),
            absl::StatusCode::kOutOfRange);
}

}  // namespace
}  // namespace pluto::llm
