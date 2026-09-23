#include "src/llm/experiments/one_shot_memorizer/token_trace.h"

#include <cuda_runtime_api.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layer_hooks.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

constexpr int kChannels = 6;
constexpr int kVocabulary = 13;
constexpr int kLogitStride = 16;
constexpr int64_t kBatch = ActivationType::kBatchDimension;

template <class T>
absl::StatusOr<Buffer> Upload(cuda::Executor& executor,
                              absl::Span<const T> values) {
  ASSIGN_OR_RETURN(auto host,
                   cuda::PageLockedHostArray<T>::CopyFrom(executor, values));
  ASSIGN_OR_RETURN(auto buffer, Buffer::Allocate(executor, host.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(buffer.data(), host.data(), host.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload token trace test fixture"));
  return buffer;
}

template <class T>
absl::StatusOr<std::vector<T>> CopyD2H(cuda::Executor& executor,
                                       const Buffer& buffer) {
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<T>::Allocate(
                                  executor, buffer.size_bytes() / sizeof(T)));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host.data(), buffer.data(), host.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "read token trace test fixture"));
  RETURN_IF_ERROR(executor.Synchronize());
  return std::vector<T>(host.begin(), host.end());
}

absl::StatusOr<Buffer> UploadFeatures(cuda::Executor& executor,
                                      absl::Span<const float> values,
                                      DataType type) {
  if (type == DataType::FP32)
    return Upload<float>(executor, values);
  std::vector<uint16_t> bits;
  for (float value : values)
    bits.push_back(static_cast<uint16_t>(std::bit_cast<uint32_t>(value) >> 16));
  return Upload<uint16_t>(executor, bits);
}

absl::StatusOr<std::vector<float>> CopyFeatures(cuda::Executor& executor,
                                                const Buffer& buffer,
                                                DataType type) {
  if (type == DataType::FP32)
    return CopyD2H<float>(executor, buffer);
  ASSIGN_OR_RETURN(auto bits, CopyD2H<uint16_t>(executor, buffer));
  std::vector<float> values;
  for (uint16_t value : bits)
    values.push_back(std::bit_cast<float>(static_cast<uint32_t>(value) << 16));
  return values;
}

class Boundary final : public Layer {
 public:
  Boundary(DataType type, int context)
      : signature_(type, {kBatch, context, kChannels}) {}
  absl::string_view name() const override { return "projection"; }
  absl::Span<const ActivationType> input_types() const override {
    return {&signature_, 1};
  }
  absl::Span<const ActivationType> output_types() const override {
    return {&signature_, 1};
  }
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return DataType::FP16; }

 private:
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor&,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override {
    BackwardState state;
    state.intermediates = {inputs[0]};
    // Hook output initially aliases both the producer and saved state.
    return FwdResult{{inputs[0]}, std::move(state)};
  }
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override {
    return absl::InternalError("token tracing must not run backward");
  }
  ActivationType signature_;
};

struct Observation {
  std::vector<float> before;
  std::vector<float> after;
};

// Deliberately simple host arithmetic with real GPU buffers/hooks: each output
// row depends on its own channel 0 and prefix row 0's channel 1. The fixture
// asserts producer storage survives every patch and gives future padding huge
// logit values in invalid vocabulary columns, which must never win top-1.
class ToyModel final : public Layer {
 public:
  explicit ToyModel(DataType type = DataType::FP32, int context = 4)
      : boundary_(type, context),
        type_(type),
        context_(context),
        input_(DataType::INT32, {kBatch, context}),
        output_(DataType::FP32, {kBatch, context, kLogitStride}) {}
  absl::string_view name() const override { return "toy"; }
  absl::Span<const ActivationType> input_types() const override {
    return {&input_, 1};
  }
  absl::Span<const ActivationType> output_types() const override {
    return {&output_, 1};
  }
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return DataType::FP16; }

  mutable std::vector<int> input_seen;
  mutable std::vector<Observation> observations;
  mutable int calls = 0;
  mutable int probability_callbacks = 0;
  bool invalid_probability = false;
  float probability_scale = 1;
  bool nonfinite_logits = false;
  bool skip_outer_scope_exit = false;

 private:
  absl::StatusOr<std::vector<float>> BoundaryCall(
      cuda::Executor& executor, const std::vector<float>& values,
      LayerHooks* hooks) const {
    ASSIGN_OR_RETURN(auto original, UploadFeatures(executor, values, type_));
    ASSIGN_OR_RETURN(auto result,
                     boundary_.fwd(executor, {&original, 1}, hooks));
    ASSIGN_OR_RETURN(auto after,
                     CopyFeatures(executor, result.outputs[0], type_));
    ASSIGN_OR_RETURN(auto intact, CopyFeatures(executor, original, type_));
    if (intact.size() != values.size() ||
        std::memcmp(intact.data(), values.data(),
                    values.size() * sizeof(float)) != 0)
      return absl::DataLossError("patch mutated its producer or saved state");
    observations.push_back({values, after});
    return after;
  }

  absl::Status Probabilities(cuda::Executor& executor,
                             LayerHooks* hooks) const {
    if (hooks == nullptr || !hooks->attention_probabilities_hook)
      return absl::OkStatus();
    std::vector<float> probabilities(2 * context_ * context_, 0);
    for (int head = 0; head < 2; ++head)
      for (int query = 0; query < context_; ++query)
        for (int key = 0; key <= query; ++key)
          probabilities[(head * context_ + query) * context_ + key] =
              head == 0 ? 1.0f / (query + 1) : static_cast<float>(key == query);
    if (invalid_probability)
      probabilities[1] = 0.5f;
    for (float& probability : probabilities)
      probability *= probability_scale;
    ASSIGN_OR_RETURN(auto device, Upload<float>(executor, probabilities));
    ++probability_callbacks;
    RETURN_IF_ERROR(hooks->attention_probabilities_hook(
        executor, "AttentionLayer",
        ActivationType(DataType::FP32, {1, 2, context_, context_}), device));
    ASSIGN_OR_RETURN(auto intact, CopyD2H<float>(executor, device));
    if (intact.size() != probabilities.size() ||
        std::memcmp(intact.data(), probabilities.data(),
                    probabilities.size() * sizeof(float)) != 0)
      return absl::DataLossError(
          "attention observation modified probabilities");
    return absl::OkStatus();
  }

  absl::StatusOr<FwdResult> Body(cuda::Executor& executor,
                                 absl::Span<const Buffer> inputs,
                                 LayerHooks* hooks) const {
    ASSIGN_OR_RETURN(input_seen, CopyD2H<int>(executor, inputs[0]));
    std::vector<float> features(context_ * kChannels);
    for (int row = 0; row < context_; ++row)
      for (int channel = 0; channel < kChannels; ++channel)
        features[row * kChannels + channel] =
            static_cast<float>(input_seen[row] + channel);
    if (hooks != nullptr && hooks->enter_combinator)
      RETURN_IF_ERROR(hooks->enter_combinator(executor, "block"));
    ASSIGN_OR_RETURN(features, BoundaryCall(executor, features, hooks));
    RETURN_IF_ERROR(Probabilities(executor, hooks));
    ASSIGN_OR_RETURN(features, BoundaryCall(executor, features, hooks));
    if (hooks != nullptr && hooks->exit_combinator)
      RETURN_IF_ERROR(hooks->exit_combinator(executor));
    if (hooks != nullptr && hooks->enter_combinator)
      RETURN_IF_ERROR(hooks->enter_combinator(executor, "other_block"));
    ASSIGN_OR_RETURN(features, BoundaryCall(executor, features, hooks));
    if (hooks != nullptr && hooks->exit_combinator)
      RETURN_IF_ERROR(hooks->exit_combinator(executor));
    std::vector<float> logits(context_ * kLogitStride, -4);
    for (int row = 0; row < context_; ++row) {
      const int winner =
          static_cast<int>(features[row * kChannels] + features[1]) %
          kVocabulary;
      logits[row * kLogitStride + winner] = 4;
      for (int column = kVocabulary; column < kLogitStride; ++column)
        logits[row * kLogitStride + column] =
            std::numeric_limits<float>::infinity();
      if (nonfinite_logits)
        logits[row * kLogitStride] = std::numeric_limits<float>::quiet_NaN();
    }
    ASSIGN_OR_RETURN(auto output, Upload<float>(executor, logits));
    return FwdResult{{std::move(output)}, {}};
  }

  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks* hooks) const override {
    ++calls;
    observations.clear();
    if (hooks != nullptr && hooks->enter_combinator)
      RETURN_IF_ERROR(hooks->enter_combinator(executor, name()));
    auto result = Body(executor, inputs, hooks);
    if (!skip_outer_scope_exit && hooks != nullptr && hooks->exit_combinator) {
      const auto status = hooks->exit_combinator(executor);
      if (!status.ok())
        return status;
    }
    return result;
  }
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override {
    return absl::InternalError("token tracing must not run backward");
  }
  Boundary boundary_;
  DataType type_;
  int context_;
  ActivationType input_;
  ActivationType output_;
};

TokenTraceSite FirstSite() {
  return {
      .scope = {"toy", "block"}, .layer_name = "projection", .occurrence = 0};
}

class TokenTraceTest : public testing::Test {
 protected:
  void SetUp() override {
    auto result = cuda::Executor::Create();
    ASSERT_TRUE(result.ok()) << result.status();
    executor_ = std::move(*result);
  }
  void TearDown() override {
    if (!executor_)
      return;
    EXPECT_TRUE(executor_->Synchronize().ok());
  }
  absl::StatusOr<TokenTraceResult> Run(
      const ToyModel& model, absl::Span<const int> prefix,
      absl::Span<const TokenTracePatch> patches = {}, bool capture = true) {
    return TraceNextToken(*executor_, model, prefix,
                          {.vocabulary_size = kVocabulary,
                           .padding_token = 0,
                           .capture_activations = capture,
                           .capture_attention = capture,
                           .patches = patches});
  }
  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(TokenTraceTest,
       ExactlyExplicitFiveTokenPrefixNoSuffixOrPaddedVocabulary) {
  ToyModel model(DataType::FP32, 8);
  const std::vector<int> prefix{1, 2, 3, 4, 5};
  const auto result = Run(model, prefix, {}, false);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(model.calls, 1);
  EXPECT_EQ(model.input_seen, (std::vector<int>{1, 2, 3, 4, 5, 0, 0, 0}));
  EXPECT_EQ(result->query_row, 4u);
  EXPECT_EQ(result->predicted_token, 7);
  EXPECT_EQ(result->logits.size(), static_cast<size_t>(kVocabulary));
  EXPECT_TRUE(result->activations.empty());
  EXPECT_TRUE(result->attention.empty());
  EXPECT_EQ(model.probability_callbacks, 0);
}

TEST_F(TokenTraceTest, CapturesExactDtypesScopesAndCausalPrefixProbabilities) {
  for (DataType type : {DataType::BF16, DataType::FP32}) {
    ToyModel model(type);
    const std::vector<int> prefix{1, 2};
    auto bare = Run(model, prefix, {}, false);
    auto captured = Run(model, prefix);
    ASSERT_TRUE(bare.ok()) << bare.status();
    ASSERT_TRUE(captured.ok()) << captured.status();
    ASSERT_EQ(bare->logits.size(), captured->logits.size());
    EXPECT_EQ(std::memcmp(bare->logits.data(), captured->logits.data(),
                          bare->logits.size() * sizeof(float)),
              0);
    ASSERT_EQ(captured->activations.size(), 4u);
    for (size_t i = 0; i < 3; ++i) {
      const auto& activation = captured->activations[i];
      EXPECT_EQ(activation.data_type, type);
      EXPECT_EQ(activation.first_row, 0u);
      EXPECT_EQ(activation.row_count, 2u);
      EXPECT_EQ(activation.channels, static_cast<size_t>(kChannels));
      EXPECT_EQ(activation.values.size(), 2u * kChannels);
      EXPECT_EQ(activation.bytes.size(),
                2u * kChannels * (type == DataType::BF16 ? 2u : 4u));
      EXPECT_EQ(
          activation.site.scope,
          (std::vector<std::string>{"toy", i == 2 ? "other_block" : "block"}));
      EXPECT_EQ(activation.site.occurrence, i == 1 ? 1 : 0);
      EXPECT_EQ(activation.values[0], 1);
      EXPECT_EQ(activation.values[kChannels], 2);
    }
    const auto& output = captured->activations.back();
    EXPECT_TRUE(output.site.scope.empty());
    EXPECT_EQ(output.site.layer_name, "toy");
    EXPECT_EQ(output.first_row, 1u);
    EXPECT_EQ(output.row_count, 1u);
    ASSERT_EQ(captured->attention.size(), 1u);
    const auto& attention = captured->attention[0];
    EXPECT_EQ(attention.heads, 2u);
    EXPECT_EQ(attention.prefix_length, 2u);
    EXPECT_EQ(attention.probabilities,
              (std::vector<float>{1, 0, .5, .5, 1, 0, 0, 1}));
  }
}

TEST_F(TokenTraceTest, ZeroPatchRespectsExactRowsChannelsAndProducerStorage) {
  for (DataType type : {DataType::BF16, DataType::FP32}) {
    ToyModel model(type);
    const std::vector<int> prefix{1, 2};
    TokenTracePatch patch{.site = FirstSite(),
                          .rows = TokenTraceRows::kOne,
                          .row = 0,
                          .first_channel = 1,
                          .channel_count = 2};
    auto result = Run(model, prefix, {&patch, 1});
    ASSERT_TRUE(result.ok()) << result.status();
    ASSERT_EQ(model.observations.size(), 3u);
    const auto& observation = model.observations[0];
    for (size_t i = 0; i < observation.after.size(); ++i)
      EXPECT_EQ(observation.after[i],
                i == 1 || i == 2 ? 0 : observation.before[i]);
    EXPECT_EQ(result->predicted_token, 2);

    patch.rows = TokenTraceRows::kQuery;
    patch.first_channel = 0;
    patch.channel_count = 1;
    result = Run(model, prefix, {&patch, 1});
    ASSERT_TRUE(result.ok()) << result.status();
    for (size_t i = 0; i < model.observations[0].after.size(); ++i)
      EXPECT_EQ(model.observations[0].after[i],
                i == kChannels ? 0 : model.observations[0].before[i]);

    patch.rows = TokenTraceRows::kAllPrefix;
    patch.first_channel = 3;
    result = Run(model, prefix, {&patch, 1});
    ASSERT_TRUE(result.ok()) << result.status();
    for (size_t i = 0; i < model.observations[0].after.size(); ++i)
      EXPECT_EQ(model.observations[0].after[i],
                i == 3 || i == 9 ? 0 : model.observations[0].before[i]);
  }
}

TEST_F(TokenTraceTest, DonorAndIdentityPatchesPreserveRawBytesAndDonor) {
  for (DataType type : {DataType::BF16, DataType::FP32}) {
    ToyModel model(type);
    const std::vector<int> clean_prefix{1, 2};
    const std::vector<int> donor_prefix{3, 4};
    auto clean = Run(model, clean_prefix);
    auto donor = Run(model, donor_prefix);
    ASSERT_TRUE(clean.ok()) << clean.status();
    ASSERT_TRUE(donor.ok()) << donor.status();
    const auto original_bytes = donor->activations[0].bytes;
    const auto original_values = donor->activations[0].values;
    TokenTracePatch patch{.site = FirstSite(),
                          .replacement = TokenTraceReplacement::kDonor,
                          .donor = &donor->activations[0]};
    auto patched = Run(model, clean_prefix, {&patch, 1});
    ASSERT_TRUE(patched.ok()) << patched.status();
    EXPECT_EQ(patched->predicted_token,
              6);  // Donor query 4 + recipient first-row channel 1 == 2.
    EXPECT_EQ(donor->activations[0].bytes, original_bytes);
    EXPECT_EQ(donor->activations[0].values, original_values);
    EXPECT_EQ(patched->activations[0].values[0], 1);
    EXPECT_EQ(patched->activations[0].values[kChannels], 4);

    patch.donor = &clean->activations[0];
    patch.rows = TokenTraceRows::kAllPrefix;
    auto identity = Run(model, clean_prefix, {&patch, 1});
    ASSERT_TRUE(identity.ok()) << identity.status();
    EXPECT_EQ(identity->activations[0].bytes, clean->activations[0].bytes);
    EXPECT_EQ(std::memcmp(identity->logits.data(), clean->logits.data(),
                          clean->logits.size() * sizeof(float)),
              0);
  }
}

TEST_F(TokenTraceTest, ExactScopeAndOccurrencePreventAmbiguousSites) {
  ToyModel model;
  const std::vector<int> prefix{1, 2};
  TokenTracePatch patch{.site = FirstSite()};
  patch.site.occurrence = -1;
  auto result = Run(model, prefix, {&patch, 1});
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  patch.site.scope = {"toy", "other_block"};
  result = Run(model, prefix, {&patch, 1});
  EXPECT_TRUE(result.ok()) << result.status();
  patch.site.scope = {"block"};  // Not a substring/ancestor-name search.
  result = Run(model, prefix, {&patch, 1});
  EXPECT_EQ(result.status().code(), absl::StatusCode::kNotFound);
  patch.site = FirstSite();
  patch.site.occurrence = 20;
  EXPECT_EQ(Run(model, prefix, {&patch, 1}).status().code(),
            absl::StatusCode::kNotFound);
  patch.site.occurrence = 0;
  patch.site.output_index = 1;
  EXPECT_EQ(Run(model, prefix, {&patch, 1}).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(TokenTraceTest, DonorSliceUsesAbsoluteRowAndAuthoritativeRawBytes) {
  for (DataType type : {DataType::BF16, DataType::FP32}) {
    ToyModel model(type);
    const std::vector<int> prefix{1, 2};
    auto baseline = Run(model, prefix);
    ASSERT_TRUE(baseline.ok()) << baseline.status();
    auto donor = baseline->activations[0];
    const size_t element_bytes = type == DataType::BF16 ? 2 : 4;
    const size_t row_bytes = kChannels * element_bytes;
    donor.bytes.erase(donor.bytes.begin(), donor.bytes.begin() + row_bytes);
    donor.first_row = 1;
    donor.row_count = 1;
    // Deliberately make the presentation stale. A donor patch must transport
    // raw physical bytes, not re-encode the decoded display values.
    donor.values.assign(kChannels, 999.0f);
    if (type == DataType::BF16) {
      const uint16_t negative_zero = 0x8000;
      std::memcpy(donor.bytes.data() + 4 * element_bytes, &negative_zero, 2);
    } else {
      const uint32_t negative_zero = 0x80000000u;
      std::memcpy(donor.bytes.data() + 4 * element_bytes, &negative_zero, 4);
    }
    const auto original_donor_bytes = donor.bytes;
    TokenTracePatch patch{.site = FirstSite(),
                          .first_channel = 4,
                          .channel_count = 1,
                          .replacement = TokenTraceReplacement::kDonor,
                          .donor = &donor};
    auto patched = Run(model, prefix, {&patch, 1});
    ASSERT_TRUE(patched.ok()) << patched.status();
    auto expected = baseline->activations[0].bytes;
    std::memcpy(expected.data() + row_bytes + 4 * element_bytes,
                donor.bytes.data() + 4 * element_bytes, element_bytes);
    EXPECT_EQ(patched->activations[0].bytes, expected);
    EXPECT_TRUE(std::signbit(patched->activations[0].values[kChannels + 4]));
    EXPECT_EQ(donor.bytes, original_donor_bytes);
    EXPECT_EQ(donor.values, std::vector<float>(kChannels, 999.0f));
    EXPECT_EQ(patched->logits, baseline->logits);
    patch.rows = TokenTraceRows::kAllPrefix;
    EXPECT_EQ(Run(model, prefix, {&patch, 1}).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST_F(TokenTraceTest, MultiplePatchesComposeWithoutMutatingTheirProducer) {
  ToyModel model;
  const std::vector<int> prefix{1, 2};
  const std::vector<TokenTracePatch> patches{
      {.site = FirstSite(),
       .rows = TokenTraceRows::kOne,
       .row = 0,
       .first_channel = 1,
       .channel_count = 1},
      {.site = FirstSite(), .first_channel = 0, .channel_count = 1}};
  auto result = Run(model, prefix, patches);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->predicted_token, 0);
  EXPECT_EQ(result->activations[0].values[1], 0);
  EXPECT_EQ(result->activations[0].values[kChannels], 0);
  const auto independent = Run(model, prefix);
  ASSERT_TRUE(independent.ok()) << independent.status();
  EXPECT_EQ(independent->predicted_token, 4);
}

TEST_F(TokenTraceTest, RejectsUnclosedCombinatorScopes) {
  ToyModel model;
  model.skip_outer_scope_exit = true;
  const std::vector<int> prefix{1, 2};
  EXPECT_EQ(Run(model, prefix).status().code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST_F(TokenTraceTest, RejectsInvalidSelectionsAndMalformedDonors) {
  ToyModel model;
  const std::vector<int> prefix{1, 2};
  TokenTracePatch patch{
      .site = FirstSite(), .rows = TokenTraceRows::kOne, .row = 2};
  EXPECT_FALSE(Run(model, prefix, {&patch, 1}).ok());
  patch.rows = TokenTraceRows::kQuery;
  patch.first_channel = kChannels;
  EXPECT_FALSE(Run(model, prefix, {&patch, 1}).ok());
  patch.first_channel = 1;
  patch.channel_count = kChannels;
  EXPECT_FALSE(Run(model, prefix, {&patch, 1}).ok());
  patch.channel_count = 0;
  patch.replacement = TokenTraceReplacement::kDonor;
  EXPECT_FALSE(Run(model, prefix, {&patch, 1}).ok());
  auto baseline = Run(model, prefix);
  ASSERT_TRUE(baseline.ok()) << baseline.status();
  auto donor = baseline->activations[0];
  patch.donor = &donor;
  donor.bytes.pop_back();
  EXPECT_FALSE(Run(model, prefix, {&patch, 1}).ok());
  donor = baseline->activations[0];
  donor.data_type = DataType::BF16;
  EXPECT_FALSE(Run(model, prefix, {&patch, 1}).ok());
  donor = baseline->activations[0];
  donor.first_row = 3;
  EXPECT_FALSE(Run(model, prefix, {&patch, 1}).ok());
}

TEST_F(TokenTraceTest, InvalidPrefixFailsBeforeForwardAndRealNonfiniteFails) {
  ToyModel model;
  EXPECT_FALSE(Run(model, {}).ok());
  EXPECT_FALSE(Run(model, std::vector<int>{kVocabulary}).ok());
  EXPECT_FALSE(Run(model, std::vector<int>{1, 2, 3, 4, 5}).ok());
  EXPECT_EQ(model.calls, 0);
  model.nonfinite_logits = true;
  EXPECT_EQ(Run(model, std::vector<int>{1, 2}, {}, false).status().code(),
            absl::StatusCode::kDataLoss);
}

TEST_F(TokenTraceTest, RejectsNoncausalAttentionCaptureWithoutModifyingIt) {
  ToyModel model;
  model.invalid_probability = true;
  EXPECT_EQ(Run(model, std::vector<int>{1, 2}).status().code(),
            absl::StatusCode::kDataLoss);
}

TEST_F(TokenTraceTest, RequiresNormalizedAttentionRowsWithRoundingTolerance) {
  ToyModel model;
  for (float scale : {0.0f, 0.5f, 1.01f}) {
    model.probability_scale = scale;
    EXPECT_EQ(Run(model, std::vector<int>{1, 2}).status().code(),
              absl::StatusCode::kDataLoss);
  }
  model.probability_scale = 1.00001f;
  const auto rounded = Run(model, std::vector<int>{1, 2});
  ASSERT_TRUE(rounded.ok()) << rounded.status();
  // Validation tolerates small reconstruction error but never renormalizes
  // the evidence or silently clips a value back to one.
  EXPECT_FLOAT_EQ(rounded->attention[0].probabilities[0], 1.00001f);
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
