#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/llm/layer.h"

namespace pluto::llm {
namespace {

constexpr int64_t kBatch = ActivationType::kBatchDimension;
static_assert(kBatch == -2);

// Both graph implementations must expose immutable signatures through const
// layers. Otherwise callers could silently invalidate an already-checked edge.
using DeviceSignature = absl::Span<const ActivationType> (Layer::*)() const;
using ReferenceSignature =
    absl::Span<const ActivationType> (LayerReference::*)() const;
static_assert(std::is_same_v<decltype(&Layer::input_types), DeviceSignature>);
static_assert(std::is_same_v<decltype(&Layer::output_types), DeviceSignature>);
static_assert(
    std::is_same_v<decltype(&LayerReference::input_types), ReferenceSignature>);
static_assert(std::is_same_v<decltype(&LayerReference::output_types),
                             ReferenceSignature>);
static_assert(
    std::is_same_v<decltype(std::declval<const ActivationType&>().dimensions()),
                   absl::Span<const int64_t>>);
static_assert(
    std::is_same_v<decltype(std::declval<ActivationType&>().dimensions()),
                   absl::Span<const int64_t>>);

TEST(ActivationTypeTest, BatchMatchesOnlyTheSameBatchSymbol) {
  const ActivationType type(DataType::BF16, {kBatch, 1024, 512});
  EXPECT_EQ(type, ActivationType(DataType::BF16, {kBatch, 1024, 512}));
  for (int64_t dimension :
       {int64_t{1}, int64_t{10}, int64_t{1024}, int64_t{-1}, int64_t{-3}}) {
    SCOPED_TRACE(dimension);
    const ActivationType other(DataType::BF16, {dimension, 1024, 512});
    EXPECT_NE(type, other);
    EXPECT_NE(other, type);
  }
}

TEST(ActivationTypeTest, EqualityIncludesDataTypeRankAndEveryExtent) {
  const ActivationType type(DataType::FP32, {kBatch, 16, 32});
  const std::array alternatives = {
      ActivationType(DataType::BF16, {kBatch, 16, 32}),
      ActivationType(DataType::INT32, {kBatch, 16, 32}),
      ActivationType(DataType::FP32, {kBatch, 16}),
      ActivationType(DataType::FP32, {kBatch, 16, 32, 1}),
      ActivationType(DataType::FP32, {kBatch, 8, 32}),
      ActivationType(DataType::FP32, {kBatch, 16, 64}),
      ActivationType(DataType::FP32, {kBatch, 32, 16}),
  };
  for (const auto& other : alternatives) {
    EXPECT_FALSE(type == other);
    EXPECT_TRUE(type != other);
  }
  const auto copy = type;
  EXPECT_TRUE(type == copy);
  EXPECT_FALSE(type != copy);
}

TEST(ActivationTypeTest, EqualityDoesNotFlattenOrBroadcast) {
  // Equal element counts do not make a sequence and a wider single token the
  // same type. In particular, -2 denotes samples, not flattened token rows.
  EXPECT_NE(ActivationType(DataType::FP32, {kBatch, 16, 32}),
            ActivationType(DataType::FP32, {kBatch, 512}));
  EXPECT_NE(ActivationType(DataType::FP32, {kBatch, 16, 32}),
            ActivationType(DataType::FP32, {kBatch, 1, 512}));
  EXPECT_NE(ActivationType(DataType::FP32, {16, 32}),
            ActivationType(DataType::FP32, {512}));
  EXPECT_NE(ActivationType(DataType::FP32, {kBatch, 16, 32}),
            ActivationType(DataType::FP32, {1, 16, 32}));
  EXPECT_NE(ActivationType(DataType::FP32, {kBatch, 16, 32}),
            ActivationType(DataType::FP32, {kBatch, 1, 32}));
  EXPECT_NE(ActivationType(DataType::FP32, {}),
            ActivationType(DataType::FP32, {1}));
}

TEST(ActivationTypeTest, ValidatesScalarsUnbatchedAndSymbolicBatchShapes) {
  const std::array valid = {
      ActivationType(DataType::FP32, {}),
      ActivationType(DataType::FP32, {1}),
      ActivationType(DataType::FP32, {512, 8192}),
      ActivationType(DataType::INT32, {kBatch}),
      ActivationType(DataType::INT32, {kBatch, 1024}),
      ActivationType(DataType::BF16, {kBatch, 1024, 512}),
      ActivationType(DataType::FP32, {std::numeric_limits<int64_t>::max()}),
  };
  for (const auto& type : valid)
    EXPECT_TRUE(type.Validate().ok()) << type.Validate();
}

TEST(ActivationTypeTest, DimensionsCanExceedInlineCapacity) {
  const ActivationType type(DataType::FP32, {kBatch, 2, 3, 4, 5, 6});
  ASSERT_TRUE(type.Validate().ok());
  ASSERT_EQ(type.dimensions().size(), 6);
  const std::array<int64_t, 6> expected{kBatch, 2, 3, 4, 5, 6};
  for (size_t index = 0; index < expected.size(); ++index)
    EXPECT_EQ(type.dimensions()[index], expected[index]);
  const auto copy = type;
  EXPECT_EQ(copy, type);
  EXPECT_NE(type, ActivationType(DataType::FP32, {kBatch, 2, 3, 4, 5, 7}));
  EXPECT_TRUE(
      ActivationType(DataType::FP32, {1, 2, 3, 4, 5, 6}).Validate().ok());
}

TEST(ActivationTypeTest, RejectsNonpositiveDimensionsOtherThanLeadingBatch) {
  const std::array invalid = {
      ActivationType(DataType::FP32, {-1, 32}),
      ActivationType(DataType::FP32, {-3, 32}),
      ActivationType(DataType::FP32, {0, 32}),
      ActivationType(DataType::FP32, {kBatch, -1, 32}),
      ActivationType(DataType::FP32, {kBatch, -3, 32}),
      ActivationType(DataType::FP32, {kBatch, 0, 32}),
      ActivationType(DataType::FP32, {16, kBatch, 32}),
      ActivationType(DataType::FP32, {kBatch, 16, kBatch}),
      ActivationType(DataType::FP32, {kBatch, kBatch}),
  };
  for (const auto& type : invalid)
    EXPECT_EQ(type.Validate().code(), absl::StatusCode::kInvalidArgument);
}

TEST(ActivationTypeTest, ValidatesEveryPhysicalDataTypeAndRejectsUnknownEnums) {
  for (DataType dtype : {DataType::FP16, DataType::BF16, DataType::FP8,
                         DataType::FP32, DataType::INT32}) {
    const ActivationType type(dtype, {kBatch, 16, 32});
    EXPECT_EQ(type.data_type(), dtype);
    EXPECT_TRUE(type.Validate().ok()) << type.Validate();
  }
  for (int invalid : {-1, 5, 999})
    EXPECT_EQ(ActivationType(static_cast<DataType>(invalid), {kBatch, 16})
                  .Validate()
                  .code(),
              absl::StatusCode::kInvalidArgument);
}

TEST(ActivationTypeTest, LegacyComputePolicyDoesNotMislabelPhysicalStorage) {
  EXPECT_EQ(ActivationDataType(DataType::FP16), DataType::FP32);
  EXPECT_EQ(ActivationDataType(DataType::BF16), DataType::BF16);
  EXPECT_EQ(ActivationDataType(DataType::FP8), DataType::FP8);
  EXPECT_EQ(ActivationDataType(DataType::FP32), DataType::FP32);
  EXPECT_EQ(ActivationDataType(DataType::INT32), DataType::INT32);
  // A literal FP16 physical type remains FP16; the mapping above only describes
  // the legacy compute policy, not a rewrite of ActivationType itself.
  EXPECT_EQ(ActivationType(DataType::FP16, {16}).data_type(), DataType::FP16);
}

}  // namespace
}  // namespace pluto::llm
