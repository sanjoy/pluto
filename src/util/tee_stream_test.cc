#include "src/util/tee_stream.h"

#include <ostream>
#include <sstream>

#include "gtest/gtest.h"

namespace pluto::util {
namespace {

class CountingStringBuffer final : public std::stringbuf {
 public:
  int sync() override {
    ++sync_count_;
    return std::stringbuf::sync();
  }

  int sync_count() const { return sync_count_; }

 private:
  int sync_count_ = 0;
};

TEST(TeeStreamTest, CopiesValuesAndManipulatorsToBothStreams) {
  std::ostringstream first;
  std::ostringstream second;
  TeeStream stream(first, second);

  stream << "value=" << std::hex << 255 << std::endl;

  EXPECT_EQ(first.str(), "value=ff\n");
  EXPECT_EQ(second.str(), first.str());
}

TEST(TeeStreamTest, FlushesSecondStreamAfterEveryInsertion) {
  std::ostringstream first;
  CountingStringBuffer second_buffer;
  std::ostream second(&second_buffer);
  TeeStream stream(first, second);

  stream << "step=" << 7;

  EXPECT_EQ(second_buffer.sync_count(), 2);
  EXPECT_EQ(second_buffer.str(), "step=7");
}

}  // namespace
}  // namespace pluto::util
