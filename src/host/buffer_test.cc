#include "src/host/buffer.h"

#include <cstddef>
#include <cstring>

#include "gtest/gtest.h"

namespace pluto::host {
namespace {

TEST(BufferTest, CopiesShareStorage) {
  auto buffer = Buffer::Allocate(sizeof(float));
  ASSERT_TRUE(buffer.ok()) << buffer.status();
  const float value = 3.25f;
  std::memcpy(buffer->data(), &value, sizeof(value));

  Buffer copy = *buffer;
  float observed = 0.0f;
  std::memcpy(&observed, copy.data(), sizeof(observed));
  EXPECT_FLOAT_EQ(observed, value);
  EXPECT_EQ(copy.data(), buffer->data());
}

TEST(BufferTest, ZeroByteAllocationHasNoStorage) {
  auto buffer = Buffer::Allocate(0);
  ASSERT_TRUE(buffer.ok()) << buffer.status();
  EXPECT_EQ(buffer->size_bytes(), 0u);
  EXPECT_EQ(buffer->data(), nullptr);
}

}  // namespace
}  // namespace pluto::host
