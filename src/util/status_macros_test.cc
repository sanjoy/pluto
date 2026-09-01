#include "src/util/status_macros.h"

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"

namespace {

absl::Status ReturnStatus(absl::Status status, int* evaluations) {
  RETURN_IF_ERROR(([&] {
    ++*evaluations;
    return status;
  })());
  return absl::OkStatus();
}

absl::StatusOr<int> AssignDeclared(absl::StatusOr<int> result,
                                   int* evaluations) {
  ASSIGN_OR_RETURN(int value, ([&]() -> absl::StatusOr<int> {
                     ++*evaluations;
                     return result;
                   })());
  return value + 1;
}

absl::StatusOr<int> AssignExisting(absl::StatusOr<int> result) {
  int value = 0;
  ASSIGN_OR_RETURN(value, result);
  return value;
}

absl::StatusOr<std::unique_ptr<int>> AssignMoveOnly() {
  ASSIGN_OR_RETURN(auto value, absl::StatusOr<std::unique_ptr<int>>(
                                   std::make_unique<int>(42)));
  return value;
}

absl::StatusOr<int> AssignTwiceInOneScope() {
  ASSIGN_OR_RETURN(int left, absl::StatusOr<int>(20));
  ASSIGN_OR_RETURN(int right, absl::StatusOr<int>(22));
  return left + right;
}

TEST(ReturnIfErrorTest, ReturnsOkAndEvaluatesExpressionOnce) {
  int evaluations = 0;
  EXPECT_TRUE(ReturnStatus(absl::OkStatus(), &evaluations).ok());
  EXPECT_EQ(evaluations, 1);
}

TEST(ReturnIfErrorTest, PropagatesErrorUnchanged) {
  int evaluations = 0;
  const absl::Status result =
      ReturnStatus(absl::InvalidArgumentError("bad input"), &evaluations);
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(result.message(), "bad input");
  EXPECT_EQ(evaluations, 1);
}

TEST(AssignOrReturnTest, DeclaresValueInSurroundingScope) {
  int evaluations = 0;
  const absl::StatusOr<int> result = AssignDeclared(41, &evaluations);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(*result, 42);
  EXPECT_EQ(evaluations, 1);
}

TEST(AssignOrReturnTest, PropagatesErrorAndEvaluatesExpressionOnce) {
  int evaluations = 0;
  const absl::StatusOr<int> result =
      AssignDeclared(absl::NotFoundError("missing"), &evaluations);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(result.status().message(), "missing");
  EXPECT_EQ(evaluations, 1);
}

TEST(AssignOrReturnTest, AssignsAnExistingVariable) {
  const absl::StatusOr<int> result = AssignExisting(42);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(*result, 42);
}

TEST(AssignOrReturnTest, MovesMoveOnlyValues) {
  absl::StatusOr<std::unique_ptr<int>> result = AssignMoveOnly();
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(**result, 42);
}

TEST(AssignOrReturnTest, UsesUniqueTemporaries) {
  const absl::StatusOr<int> result = AssignTwiceInOneScope();
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(*result, 42);
}

}  // namespace
