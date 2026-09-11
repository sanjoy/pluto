#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <cstddef>
#include <memory>

#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"

namespace pluto::cuda {
namespace {

constexpr int kElementCount = 128;
constexpr int kTileSize = 8;

// A deliberately small CUDA Tile C++ kernel. Each logical block loads one
// eight-element tile, adds the two tiles element-wise, and stores the result.
// The compiler, rather than explicit threadIdx arithmetic, maps the tile work
// onto the block's hardware threads.
__tile_global__ void AddTiles(const int* __restrict__ left,
                              const int* __restrict__ right,
                              int* __restrict__ output) {
  namespace ct = ::cuda::tiles;
  using namespace ct::literals;

  left = ct::assume_aligned(left, 16_ic);
  right = ct::assume_aligned(right, 16_ic);
  output = ct::assume_aligned(output, 16_ic);

  auto left_view = ct::partition_view{
      ct::tensor_span{left, ct::extents{128_ic}}, ct::shape{8_ic}};
  auto right_view = ct::partition_view{
      ct::tensor_span{right, ct::extents{128_ic}}, ct::shape{8_ic}};
  auto output_view = ct::partition_view{
      ct::tensor_span{output, ct::extents{128_ic}}, ct::shape{8_ic}};

  const int block = ct::bid().x;
  output_view.store(left_view.load(block) + right_view.load(block), block);
}

class CuTileTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }

  void TearDown() override {
    if (executor_ == nullptr)
      return;
    EXPECT_TRUE(executor_->Synchronize().ok());
    executor_.reset();
  }

  std::unique_ptr<Executor> executor_;
};

TEST_F(CuTileTest, AddsOneTilePerLogicalBlock) {
  auto left_host = PageLockedHostArray<int>::Allocate(kElementCount);
  auto right_host = PageLockedHostArray<int>::Allocate(kElementCount);
  auto output_host = PageLockedHostArray<int>::Allocate(kElementCount);
  ASSERT_TRUE(left_host.ok()) << left_host.status();
  ASSERT_TRUE(right_host.ok()) << right_host.status();
  ASSERT_TRUE(output_host.ok()) << output_host.status();
  for (int index = 0; index < kElementCount; ++index) {
    (*left_host)[index] = index;
    (*right_host)[index] = 3 * index + 7;
  }

  auto left = Buffer::Allocate(*executor_, left_host->size_bytes());
  auto right = Buffer::Allocate(*executor_, right_host->size_bytes());
  auto output = Buffer::Allocate(*executor_, output_host->size_bytes());
  ASSERT_TRUE(left.ok()) << left.status();
  ASSERT_TRUE(right.ok()) << right.status();
  ASSERT_TRUE(output.ok()) << output.status();

  ASSERT_EQ(
      cudaMemcpyAsync(left->data(), left_host->data(), left_host->size_bytes(),
                      cudaMemcpyHostToDevice, executor_->stream()),
      cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(right->data(), right_host->data(),
                            right_host->size_bytes(), cudaMemcpyHostToDevice,
                            executor_->stream()),
            cudaSuccess);

  // Tile kernels use ordinary launch syntax, but their block dimension must be
  // one because the tile compiler chooses the physical thread configuration.
  AddTiles<<<kElementCount / kTileSize, 1, 0, executor_->stream()>>>(
      static_cast<const int*>(left->data()),
      static_cast<const int*>(right->data()),
      static_cast<int*>(output->data()));
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(output_host->data(), output->data(),
                            output_host->size_bytes(), cudaMemcpyDeviceToHost,
                            executor_->stream()),
            cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());

  for (int index = 0; index < kElementCount; ++index) {
    EXPECT_EQ((*output_host)[index], (*left_host)[index] + (*right_host)[index])
        << "index " << index;
  }
}

}  // namespace
}  // namespace pluto::cuda
