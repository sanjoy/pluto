#include <cstddef>
#include <limits>
#include <new>

#include "gtest/gtest.h"

// These checks run in both the ordinary C++ and NVCC compilation paths.
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
#error "Pluto must be built with C++ exceptions disabled"
#endif

namespace {
TEST(NoExceptionsDeathTest, AllocationFailureTerminates) {
  // This size cannot represent an actual allocation. Volatile prevents the
  // compiler from eliding the request or diagnosing its size at compile time.
  // operator new lives in the standard library: even if that library was built
  // with exceptions, its bad_alloc must terminate our exception-free program.
  EXPECT_DEATH_IF_SUPPORTED(
      {
        volatile size_t size = std::numeric_limits<size_t>::max();
        void* allocation = ::operator new(size);
        ::operator delete(allocation);
      },
      "");
}
}  // namespace
