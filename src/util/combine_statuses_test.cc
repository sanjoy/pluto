#include "src/util/combine_statuses.h"

#include <array>

#include "absl/status/status.h"
#include "gtest/gtest.h"

namespace pluto::util {
namespace {

TEST(CombineStatusesTest, EmptyInputIsOk) {
  EXPECT_TRUE(CombineStatuses({}).ok());
}

TEST(CombineStatusesTest, AllSuccessfulInputIsOk) {
  const std::array statuses = {absl::OkStatus(), absl::OkStatus(),
                               absl::OkStatus()};
  EXPECT_TRUE(CombineStatuses(statuses).ok());
}

TEST(CombineStatusesTest, SingleFailureIncludesItsIndex) {
  const std::array statuses = {absl::InvalidArgumentError("bad input")};
  const absl::Status result = CombineStatuses(statuses);
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(result.message(), "Failures:\n  [0] INVALID_ARGUMENT: bad input");
}

TEST(CombineStatusesTest, NoncontiguousFailuresKeepOriginalIndicesAndCodes) {
  const std::array statuses = {
      absl::OkStatus(), absl::InvalidArgumentError("first"),
      absl::OkStatus(), absl::NotFoundError("other"),
      absl::OkStatus(), absl::ResourceExhaustedError("last"),
      absl::OkStatus()};
  const absl::Status result = CombineStatuses(statuses);
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(result.message(),
            "Failures:\n  [1] INVALID_ARGUMENT: first\n  [3] NOT_FOUND: other"
            "\n  [5] RESOURCE_EXHAUSTED: last");
}

TEST(CombineStatusesTest, AllFailuresAreIncludedInInputOrder) {
  const std::array statuses = {absl::UnavailableError("device unavailable"),
                               absl::InternalError("second\nwith detail"),
                               absl::CancelledError("third")};
  const absl::Status result = CombineStatuses(statuses);
  EXPECT_EQ(result.code(), absl::StatusCode::kUnavailable);
  EXPECT_EQ(result.message(),
            "Failures:\n  [0] UNAVAILABLE: device unavailable"
            "\n  [1] INTERNAL: second\nwith detail\n  [2] CANCELLED: third");
}

TEST(CombineStatusesTest, DoesNotChangeInputStatuses) {
  const std::array statuses = {absl::OkStatus(),
                               absl::FailedPreconditionError("unchanged"),
                               absl::NotFoundError("also unchanged")};
  const auto original = statuses;
  EXPECT_FALSE(CombineStatuses(statuses).ok());
  EXPECT_EQ(statuses, original);
}

}  // namespace
}  // namespace pluto::util
