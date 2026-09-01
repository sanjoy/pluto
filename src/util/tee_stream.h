#pragma once

#include <ios>
#include <ostream>

namespace pluto::util {

// A small ostream-like adapter that copies each insertion to two streams.
// TeeStream does not own either stream, so both must outlive it. The second
// stream is flushed after every insertion, which makes it suitable for a log
// file that should remain readable while a long-running process is active.
class TeeStream final {
 public:
  TeeStream(std::ostream& first, std::ostream& second)
      : first_(first), second_(second) {}

  template <class Value>
  TeeStream& operator<<(const Value& value) {
    first_ << value;
    second_ << value;
    second_.flush();
    return *this;
  }

  TeeStream& operator<<(std::ostream& (*manipulator)(std::ostream&)) {
    manipulator(first_);
    manipulator(second_);
    second_.flush();
    return *this;
  }

  TeeStream& operator<<(std::ios_base& (*manipulator)(std::ios_base&)) {
    manipulator(first_);
    manipulator(second_);
    second_.flush();
    return *this;
  }

 private:
  std::ostream& first_;
  std::ostream& second_;
};

}  // namespace pluto::util
