#include "src/llm/experiments/one_shot_memorizer/quadratic_features.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layer_hooks.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/fully_connected.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

// The oracle uses simple scalar nested loops, independent of the kernel's
// output-column-to-pair index inversion. All tested operands are finite.
uint16_t RoundBf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fff + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}

float DecodeBf16(uint16_t bits) {
  return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
}

std::vector<uint16_t> Oracle(absl::Span<const uint16_t> input) {
  std::vector<uint16_t> result;
  for (size_t offset = 0; offset < input.size(); offset += 16) {
    result.insert(result.end(), input.begin() + offset,
                  input.begin() + offset + 16);
    for (int i = 0; i < 16; ++i)
      for (int j = i; j < 16; ++j) {
        const float product =
            DecodeBf16(input[offset + i]) * DecodeBf16(input[offset + j]);
        result.push_back(RoundBf16(product));
      }
  }
  return result;
}

std::vector<uint16_t> Inputs(size_t rows) {
  // 1.0078125*1.5 and 1.0234375*1.5 exercise opposite ties-to-even cases.
  constexpr std::array<float, 16> values{
      -0.0f,  0.0f,   1, -1, 1.0078125f, 1.0234375f, 1.5f,        -1.5f,
      0.125f, -0.25f, 2, -4, 31.75f,     -63.5f,     0.00390625f, -0.0078125f};
  std::vector<uint16_t> result(rows * 16);
  for (size_t row = 0; row < rows; ++row)
    for (size_t i = 0; i < 16; ++i)
      result[row * 16 + i] = RoundBf16(values[(i + row) % 16]);
  return result;
}

absl::Status Upload(cuda::Executor& executor, absl::Span<const float> values,
                    const Buffer& destination) {
  if (destination.size_bytes() != values.size() * sizeof(float))
    return absl::InvalidArgumentError("bad test upload shape");
  ASSIGN_OR_RETURN(auto host,
                   cuda::PageLockedHostArray<float>::CopyFrom(executor, values));
  return cuda::CudaStatus(
      cudaMemcpyAsync(destination.data(), host.data(), host.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload test projection");
}

absl::StatusOr<Buffer> Upload(cuda::Executor& executor,
                              absl::Span<const uint16_t> values) {
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<uint16_t>::CopyFrom(
                                  executor, values));
  ASSIGN_OR_RETURN(auto device, Buffer::Allocate(executor, host.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(device.data(), host.data(), host.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload quadratic inputs"));
  return device;
}

absl::StatusOr<std::vector<uint16_t>> Download(cuda::Executor& executor,
                                               const Buffer& source) {
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<uint16_t>::Allocate(
                                  executor, source.size_bytes() / 2));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host.data(), source.data(), host.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "download quadratic test result"));
  RETURN_IF_ERROR(executor.Synchronize());
  return std::vector<uint16_t>(host.begin(), host.end());
}

class QuadraticFeaturesTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }
  void TearDown() override {
    if (!executor_)
      return;
    EXPECT_TRUE(executor_->Synchronize().ok());
  }
  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(QuadraticFeaturesTest, MatchesExactBf16OracleForRowsSamplesAndPadding) {
  for (const auto& [batch, context] : {std::pair{1, 1}, std::pair{2, 3},
                                       std::pair{13, 5}, std::pair{1, 1024}}) {
    SCOPED_TRACE(::testing::Message()
                 << "batch=" << batch << " context=" << context);
    auto layer = QuadraticFeaturesLayer::Create(*executor_, context);
    ASSERT_TRUE(layer.ok()) << layer.status();
    const auto input = Inputs(static_cast<size_t>(batch) * context);
    auto device = Upload(*executor_, input);
    ASSERT_TRUE(device.ok()) << device.status();
    auto result = (*layer)->fwd(*executor_, {*device});
    ASSERT_TRUE(result.ok()) << result.status();
    ASSERT_EQ(result->outputs.size(), 1u);
    EXPECT_EQ(result->outputs[0].size_bytes(),
              static_cast<size_t>(batch) * context * 152 * 2);
    EXPECT_NE(result->outputs[0].data(), device->data());
    auto actual = Download(*executor_, result->outputs[0]);
    ASSERT_TRUE(actual.ok()) << actual.status();
    EXPECT_EQ(*actual, Oracle(input));
    auto unchanged = Download(*executor_, *device);
    ASSERT_TRUE(unchanged.ok()) << unchanged.status();
    EXPECT_EQ(*unchanged, input);
    auto repeat = (*layer)->fwd(*executor_, {*device});
    ASSERT_TRUE(repeat.ok()) << repeat.status();
    auto repeated = Download(*executor_, repeat->outputs[0]);
    ASSERT_TRUE(repeated.ok()) << repeated.status();
    EXPECT_EQ(*repeated, *actual);
  }
}

TEST_F(QuadraticFeaturesTest,
       ZeroRowsAreIndependentAndSignedZerosArePreserved) {
  auto layer = QuadraticFeaturesLayer::Create(*executor_, 3);
  ASSERT_TRUE(layer.ok()) << layer.status();
  auto input = Inputs(3);
  std::fill(input.begin() + 16, input.begin() + 32, uint16_t{0});
  auto device = Upload(*executor_, input);
  ASSERT_TRUE(device.ok()) << device.status();
  auto result = (*layer)->fwd(*executor_, {*device});
  ASSERT_TRUE(result.ok()) << result.status();
  auto actual = Download(*executor_, result->outputs[0]);
  ASSERT_TRUE(actual.ok()) << actual.status();
  EXPECT_EQ(*actual, Oracle(input));
  EXPECT_EQ(actual->front(), 0x8000);  // Linear -0 is copied, not added to +0.
  EXPECT_EQ((*actual)[16], 0);         // (-0)*(-0) is +0.
  EXPECT_EQ((*actual)[17], 0x8000);    // (-0)*(+0) is -0.
  for (int feature = 0; feature < 152; ++feature)
    EXPECT_EQ((*actual)[152 + feature], 0);
}

TEST_F(QuadraticFeaturesTest, EveryForwardReadsFreshInputBytes) {
  auto layer = QuadraticFeaturesLayer::Create(*executor_, 2);
  ASSERT_TRUE(layer.ok()) << layer.status();
  const auto original = Inputs(2);
  auto device = Upload(*executor_, original);
  ASSERT_TRUE(device.ok()) << device.status();
  auto first = (*layer)->fwd(*executor_, {*device});
  ASSERT_TRUE(first.ok()) << first.status();
  auto changed = original;
  changed[5] = RoundBf16(-2.75f);
  changed[31] = RoundBf16(3.125f);
  auto pinned =
      cuda::PageLockedHostArray<uint16_t>::CopyFrom(*executor_, changed);
  ASSERT_TRUE(pinned.ok()) << pinned.status();
  ASSERT_EQ(
      cudaMemcpyAsync(device->data(), pinned->data(), pinned->size_bytes(),
                      cudaMemcpyHostToDevice, executor_->stream()),
      cudaSuccess);
  auto second = (*layer)->fwd(*executor_, {*device});
  ASSERT_TRUE(second.ok()) << second.status();
  auto first_bytes = Download(*executor_, first->outputs[0]);
  auto second_bytes = Download(*executor_, second->outputs[0]);
  ASSERT_TRUE(first_bytes.ok()) << first_bytes.status();
  ASSERT_TRUE(second_bytes.ok()) << second_bytes.status();
  EXPECT_EQ(*first_bytes, Oracle(original));
  EXPECT_EQ(*second_bytes, Oracle(changed));
  EXPECT_NE(*first_bytes, *second_bytes);
}

TEST_F(QuadraticFeaturesTest,
       PublishesTypesHooksAndForwardIdentityWithoutWeights) {
  auto layer = QuadraticFeaturesLayer::Create(*executor_, 3);
  ASSERT_TRUE(layer.ok()) << layer.status();
  const ActivationType input(DataType::BF16,
                             {ActivationType::kBatchDimension, 3, 16});
  const ActivationType output(DataType::BF16,
                              {ActivationType::kBatchDimension, 3, 152});
  EXPECT_EQ((*layer)->input_types()[0], input);
  EXPECT_EQ((*layer)->output_types()[0], output);
  EXPECT_EQ((*layer)->output_type(), DataType::BF16);
  EXPECT_TRUE((*layer)->weights().empty());
  EXPECT_TRUE((*layer)->gradients().empty());
  auto device = Upload(*executor_, Inputs(6));
  ASSERT_TRUE(device.ok()) << device.status();
  int calls = 0;
  LayerHooks hooks;
  hooks.activation_hook = [&](cuda::Executor& executor, absl::string_view name,
                              absl::Span<const ActivationType> types,
                              absl::Span<Buffer> values) {
    EXPECT_EQ(&executor, executor_.get());
    EXPECT_EQ(name, "QuadraticFeaturesLayer");
    EXPECT_EQ(types.size(), 1u);
    EXPECT_EQ(types[0], output);
    EXPECT_EQ(values.size(), 1u);
    EXPECT_EQ(values[0].size_bytes(), 6u * 152 * 2);
    ++calls;
    return absl::OkStatus();
  };
  auto result = (*layer)->fwd(*executor_, {*device}, &hooks);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(result->state.layer, layer->get());
  EXPECT_TRUE(result->state.intermediates.empty());
  EXPECT_TRUE(result->state.children.empty());
  auto unsupported = (*layer)->bwd(*executor_, {}, std::move(result->state));
  EXPECT_EQ(unsupported.status().code(), absl::StatusCode::kUnimplemented);
  auto no_forward = (*layer)->bwd(*executor_, {}, {});
  EXPECT_EQ(no_forward.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(QuadraticFeaturesTest, RejectsBadContextShapesArityAndExecutor) {
  EXPECT_FALSE(QuadraticFeaturesLayer::Create(*executor_, 0).ok());
  EXPECT_FALSE(QuadraticFeaturesLayer::Create(*executor_, -1).ok());
  EXPECT_FALSE(QuadraticFeaturesLayer::Create(*executor_,
                                              std::numeric_limits<int>::max())
                   .ok());
  auto layer = QuadraticFeaturesLayer::Create(*executor_, 3);
  ASSERT_TRUE(layer.ok()) << layer.status();
  auto device = Upload(*executor_, Inputs(3));
  ASSERT_TRUE(device.ok()) << device.status();
  EXPECT_FALSE((*layer)->fwd(*executor_, {}).ok());
  EXPECT_FALSE((*layer)->fwd(*executor_, {*device, *device}).ok());
  for (size_t bytes : {0u, 31u, 32u, 97u}) {
    auto malformed = Buffer::Allocate(*executor_, bytes);
    ASSERT_TRUE(malformed.ok()) << malformed.status();
    EXPECT_FALSE((*layer)->fwd(*executor_, {*malformed}).ok());
  }
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  EXPECT_FALSE((*layer)->fwd(**other, {*device}).ok());
  auto foreign_input = Upload(**other, Inputs(3));
  ASSERT_TRUE(foreign_input.ok()) << foreign_input.status();
  EXPECT_FALSE((*layer)->fwd(*executor_, {*foreign_input}).ok());
  auto forward = (*layer)->fwd(*executor_, {*device});
  ASSERT_TRUE(forward.ok()) << forward.status();
  EXPECT_EQ(
      (*layer)->bwd(**other, {}, std::move(forward->state)).status().code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE((*other)->Synchronize().ok());
}

TEST_F(QuadraticFeaturesTest, ComposesWithNonTileAlignedProductionProjection) {
  constexpr int context = 3;
  constexpr int rows =
      48;  // Complete samples and the dense layer's 16-row tile.
  auto quadratic = QuadraticFeaturesLayer::Create(*executor_, context);
  auto projection =
      FullyConnectedLayer::Create(*executor_, 152, 16, DataType::BF16, context);
  ASSERT_TRUE(quadratic.ok()) << quadratic.status();
  ASSERT_TRUE(projection.ok()) << projection.status();
  // Select columns across all three production FC input tiles, including the
  // last physical feature151. The final 64-column MMA tile is only partly full.
  constexpr std::array<int, 16> columns{0,   15,  16,  17, 31,  32,  45, 75,
                                        135, 151, 150, 66, 100, 120, 91, 42};
  std::vector<float> weights(152 * 16, 0), bias(16, 0.125f);
  for (int channel = 0; channel < 16; ++channel)
    weights[columns[channel] * 16 + channel] = 1;
  ASSERT_TRUE(Upload(*executor_, weights, (*projection)->weights()[0]).ok());
  ASSERT_TRUE(Upload(*executor_, bias, (*projection)->weights()[1]).ok());
  ComposedLayerBuilder builder;
  ASSERT_TRUE(builder.add(std::move(*quadratic)).ok());
  ASSERT_TRUE(builder.add(std::move(*projection)).ok());
  auto model = builder.create("quadratic_test_projection");
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ((*model)->weights().size(), 2u);
  const auto input = Inputs(rows);
  const auto features = Oracle(input);
  auto device = Upload(*executor_, input);
  ASSERT_TRUE(device.ok()) << device.status();
  auto result = (*model)->fwd(*executor_, {*device});
  ASSERT_TRUE(result.ok()) << result.status();
  auto actual = Download(*executor_, result->outputs[0]);
  ASSERT_TRUE(actual.ok()) << actual.status();
  ASSERT_EQ(actual->size(), input.size());
  for (int row = 0; row < rows; ++row)
    for (int channel = 0; channel < 16; ++channel)
      EXPECT_EQ((*actual)[row * 16 + channel],
                RoundBf16(DecodeBf16(features[row * 152 + columns[channel]]) +
                          0.125f));
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
