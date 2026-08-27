#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <array>
#include <cstddef>

#include "gtest/gtest.h"
#include "src/gpu/buffer.h"

namespace pluto::gpu {
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
  namespace ct = cuda::tiles;
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
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
              cudaSuccess);
  }

  void TearDown() override {
    if (stream_ == nullptr) return;
    EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream_), cudaSuccess);
  }

  cudaStream_t stream_ = nullptr;
};

TEST_F(CuTileTest, AddsOneTilePerLogicalBlock) {
  std::array<int, kElementCount> left_host{};
  std::array<int, kElementCount> right_host{};
  std::array<int, kElementCount> output_host{};
  for (int index = 0; index < kElementCount; ++index) {
    left_host[index] = index;
    right_host[index] = 3 * index + 7;
  }

  auto left = Buffer::Allocate(sizeof(left_host), stream_);
  auto right = Buffer::Allocate(sizeof(right_host), stream_);
  auto output = Buffer::Allocate(sizeof(output_host), stream_);
  ASSERT_TRUE(left.ok()) << left.status();
  ASSERT_TRUE(right.ok()) << right.status();
  ASSERT_TRUE(output.ok()) << output.status();

  ASSERT_EQ(cudaMemcpyAsync(left->data(), left_host.data(), sizeof(left_host),
                            cudaMemcpyHostToDevice, stream_),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(right->data(), right_host.data(),
                            sizeof(right_host), cudaMemcpyHostToDevice, stream_),
            cudaSuccess);

  // Tile kernels use ordinary launch syntax, but their block dimension must be
  // one because the tile compiler chooses the physical thread configuration.
  AddTiles<<<kElementCount / kTileSize, 1, 0, stream_>>>(
      static_cast<const int*>(left->data()),
      static_cast<const int*>(right->data()),
      static_cast<int*>(output->data()));
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(output_host.data(), output->data(),
                            sizeof(output_host), cudaMemcpyDeviceToHost,
                            stream_),
            cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  for (int index = 0; index < kElementCount; ++index) {
    EXPECT_EQ(output_host[index], left_host[index] + right_host[index])
        << "index " << index;
  }
}

}  // namespace
}  // namespace pluto::gpu
