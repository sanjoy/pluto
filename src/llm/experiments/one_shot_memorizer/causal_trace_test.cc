#include "src/llm/experiments/one_shot_memorizer/causal_trace.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cstdint>
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

constexpr int kContext = 4;
constexpr int kVocabulary = 7;
constexpr int kEos = 0;
constexpr int kWidth = 2;
constexpr int64_t kBatch = ActivationType::kBatchDimension;
using Corpus = std::vector<std::vector<int>>;
const Corpus kCorpus{{1, 3, 4}, {2, 5, 6}};

template <class T>
absl::StatusOr<Buffer> Upload(cuda::Executor& executor,
                              absl::Span<const T> values) {
  ASSIGN_OR_RETURN(auto host,
                   cuda::PageLockedHostArray<T>::CopyFrom(executor, values));
  ASSIGN_OR_RETURN(auto device, Buffer::Allocate(executor, host.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(device.data(), host.data(), host.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload causal trace fixture"));
  return device;
}

template <class T>
absl::StatusOr<std::vector<T>> Download(cuda::Executor& executor,
                                        const Buffer& device) {
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<T>::Allocate(
                                  executor, device.size_bytes() / sizeof(T)));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host.data(), device.data(), host.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "download causal trace fixture"));
  RETURN_IF_ERROR(executor.Synchronize());
  return std::vector<T>(host.begin(), host.end());
}

absl::StatusOr<Buffer> UploadFeatures(cuda::Executor& executor,
                                      const std::vector<float>& values,
                                      DataType type) {
  if (type != DataType::BF16)
    return Upload<float>(executor, values);
  // Every fixture value is a small integer exactly representable in BF16.
  std::vector<uint16_t> compact;
  for (float value : values)
    compact.push_back(
        static_cast<uint16_t>(std::bit_cast<uint32_t>(value) >> 16));
  return Upload<uint16_t>(executor, compact);
}

absl::StatusOr<std::vector<float>> DownloadFeatures(cuda::Executor& executor,
                                                    const Buffer& device,
                                                    DataType type) {
  if (type != DataType::BF16)
    return Download<float>(executor, device);
  ASSIGN_OR_RETURN(const auto compact, Download<uint16_t>(executor, device));
  std::vector<float> values;
  for (uint16_t value : compact)
    values.push_back(std::bit_cast<float>(static_cast<uint32_t>(value) << 16));
  return values;
}

class BoundaryLayer final : public Layer {
 public:
  explicit BoundaryLayer(DataType type, std::string name = "ResidualLayer")
      : signature_(type, {kBatch, kContext, kWidth}), name_(std::move(name)) {}
  absl::string_view name() const override { return name_; }
  absl::Span<const ActivationType> input_types() const override {
    return {&signature_, 1};
  }
  absl::Span<const ActivationType> output_types() const override {
    return {&signature_, 1};
  }
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return DataType::FP16; }
  ActivationType signature_;

 private:
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor&,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override {
    // The original aliases saved state. A correct intervention must replace
    // the output handle rather than writing into this original allocation.
    BackwardState state;
    state.intermediates = {inputs[0]};
    return FwdResult{{inputs[0]}, std::move(state)};
  }
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override {
    return absl::InternalError("causal trace must never run backward");
  }
  std::string name_;
};

struct RowObservation {
  std::vector<int> tokens;
  std::vector<float> before;
  std::vector<float> after;
};

// Boundary 0 exposes token-local features. The next operation reads the first
// token's feature into every causal query row. Boundary 1 exposes that result.
// Therefore restoring only boundary 0's final row cannot restore earlier
// corrupted source rows, whereas restoring boundary 1's final row can rescue.
class ToyCausalModel final : public Layer {
 public:
  explicit ToyCausalModel(DataType storage = DataType::FP32)
      : early(storage),
        late(storage),
        unique(storage, "unique_boundary"),
        storage_(storage) {}
  absl::string_view name() const override { return "ToyCausalModel"; }
  absl::Span<const ActivationType> input_types() const override {
    return {&input_type_, 1};
  }
  absl::Span<const ActivationType> output_types() const override {
    return {&output_type_, 1};
  }
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return DataType::FP16; }

  BoundaryLayer early;
  BoundaryLayer late;
  BoundaryLayer unique;
  bool reverse_position_one = false;
  bool nonfinite_for_first_two = false;
  bool skip_scope_exit = false;
  mutable std::vector<RowObservation> observations;

 private:
  absl::StatusOr<std::vector<float>> ApplyBoundary(
      cuda::Executor& executor, const BoundaryLayer& boundary,
      const std::vector<int>& tokens, const std::vector<float>& before,
      LayerHooks* hooks) const {
    ASSIGN_OR_RETURN(auto original, UploadFeatures(executor, before, storage_));
    ASSIGN_OR_RETURN(auto forward,
                     boundary.fwd(executor, {&original, 1}, hooks));
    ASSIGN_OR_RETURN(auto after,
                     DownloadFeatures(executor, forward.outputs[0], storage_));
    ASSIGN_OR_RETURN(auto original_after,
                     DownloadFeatures(executor, original, storage_));
    if (original_after != before)
      return absl::DataLossError(
          "causal patch mutated its original activation");
    observations.push_back({tokens, before, after});
    return after;
  }

  absl::StatusOr<FwdResult> Body(cuda::Executor& executor,
                                 absl::Span<const Buffer> inputs,
                                 LayerHooks* hooks) const {
    ASSIGN_OR_RETURN(const auto tokens, Download<int>(executor, inputs[0]));
    std::vector<float> features(tokens.size() * kWidth, 0);
    for (size_t row = 0; row < tokens.size(); ++row)
      features[row * kWidth] = static_cast<float>(tokens[row]);
    ASSIGN_OR_RETURN(features,
                     ApplyBoundary(executor, early, tokens, features, hooks));
    for (size_t row = 0; row < tokens.size(); ++row)
      features[row * kWidth + 1] =
          features[(row / kContext * kContext) * kWidth];
    ASSIGN_OR_RETURN(features,
                     ApplyBoundary(executor, late, tokens, features, hooks));
    ASSIGN_OR_RETURN(features,
                     ApplyBoundary(executor, unique, tokens, features, hooks));
    std::vector<float> logits(tokens.size() * kVocabulary, -4);
    for (size_t row = 0; row < tokens.size(); ++row) {
      const int position = row % kContext;
      const int first_token = static_cast<int>(features[row * kWidth + 1]);
      int winner = kEos;
      if (position == 0)
        winner = first_token == 1 ? 3 : 5;
      if (position == 1) {
        winner = first_token == 1 ? 4 : 6;
        if (reverse_position_one)
          winner = first_token == 1 ? 6 : 4;
      }
      logits[row * kVocabulary + winner] = 4;
      if (nonfinite_for_first_two && position == 1 && first_token == 2)
        logits[row * kVocabulary] = std::numeric_limits<float>::quiet_NaN();
    }
    ASSIGN_OR_RETURN(auto output, Upload<float>(executor, logits));
    return FwdResult{{std::move(output)}, {}};
  }

  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor& executor,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks* hooks) const override {
    if (hooks != nullptr && hooks->enter_combinator)
      RETURN_IF_ERROR(hooks->enter_combinator(executor, "transformer_block_0"));
    auto result = Body(executor, inputs, hooks);
    if (!skip_scope_exit && hooks != nullptr && hooks->exit_combinator) {
      const auto status = hooks->exit_combinator(executor);
      if (!status.ok())
        return status;
    }
    return result;
  }
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override {
    return absl::InternalError("causal trace must never run backward");
  }

  DataType storage_;
  const ActivationType input_type_{DataType::INT32, {kBatch, kContext}};
  const ActivationType output_type_{DataType::FP32,
                                    {kBatch, kContext, kVocabulary}};
};

const std::vector<CausalTraceSite> kSites{
    {.layer_name = "ResidualLayer",
     .enclosing_scope = "transformer_block_0",
     .occurrence = 0},
    {.layer_name = "ResidualLayer",
     .enclosing_scope = "transformer_block_0",
     .occurrence = 1}};

class CausalTraceTest : public testing::Test {
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

  absl::StatusOr<std::vector<CausalTraceResult>> Trace(
      const ToyCausalModel& model, int batch = 4,
      absl::Span<const CausalTraceSite> sites = kSites, int retained = 1) {
    return TracePrefixMediation(*executor_, model, kCorpus, kVocabulary, kEos,
                                1, batch, sites, retained);
  }
  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(CausalTraceTest,
       LateQueryCarriesEarlierInformationButEarlyQueryDoesNot) {
  for (int batch : {1, 4}) {
    ToyCausalModel model;
    const auto results = Trace(model, batch);
    ASSERT_TRUE(results.ok()) << results.status();
    ASSERT_EQ(results->size(), 2u);
    for (size_t site = 0; site < results->size(); ++site) {
      const auto& result = (*results)[site];
      EXPECT_EQ(result.site.occurrence, static_cast<int>(site));
      EXPECT_EQ(result.targets, 6);
      EXPECT_EQ(result.changed_prefixes, 4);
      EXPECT_EQ(result.clean_correct, 6);
      EXPECT_EQ(result.corrupt_correct, 4);
      EXPECT_EQ(result.baseline_wrong, 2);
      EXPECT_EQ(result.rescued, site == 0 ? 0 : 2);
      EXPECT_EQ(result.newly_broken_by_rescue, 0);
      EXPECT_EQ(result.damaged, site == 0 ? 0 : 2);
      EXPECT_EQ(result.rescue_correct, site == 0 ? 4 : 6);
      EXPECT_EQ(result.damage_correct, site == 0 ? 6 : 4);
      EXPECT_EQ(result.clean_nonfinite, 0);
      EXPECT_EQ(result.corrupt_nonfinite, 0);
      EXPECT_EQ(result.rescue_nonfinite, 0);
      EXPECT_EQ(result.damage_nonfinite, 0);
    }
    bool saw_change = false;
    for (const auto& observation : model.observations) {
      ASSERT_EQ(observation.before.size(), observation.tokens.size() * kWidth);
      ASSERT_EQ(observation.after.size(), observation.before.size());
      for (size_t sample = 0; sample < observation.tokens.size() / kContext;
           ++sample) {
        const size_t base = sample * kContext;
        size_t prefix_length = 0;
        while (prefix_length < kContext &&
               observation.tokens[base + prefix_length] != kEos)
          ++prefix_length;
        for (size_t position = 0; position < kContext; ++position)
          for (int channel = 0; channel < kWidth; ++channel) {
            const size_t offset = (base + position) * kWidth + channel;
            if (observation.before[offset] == observation.after[offset])
              continue;
            saw_change = true;
            ASSERT_GT(prefix_length, 0u);
            EXPECT_EQ(position, prefix_length - 1)
                << "patched an earlier/future row or an unused batch slot";
          }
      }
    }
    EXPECT_TRUE(saw_change);
  }
}

TEST_F(CausalTraceTest, SupportsBf16PhysicalRowsAndUniqueLayerNames) {
  ToyCausalModel model(DataType::BF16);
  const std::vector<CausalTraceSite> sites{{.layer_name = "unique_boundary"}};
  const auto result = Trace(model, 4, sites);
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->size(), 1u);
  EXPECT_EQ((*result)[0].clean_correct, 6);
  EXPECT_EQ((*result)[0].corrupt_correct, 4);
  EXPECT_EQ((*result)[0].rescued, 2);
  EXPECT_EQ((*result)[0].damaged, 2);
}

TEST_F(CausalTraceTest, UnchangedPrefixesAreAnIdentityControl) {
  ToyCausalModel model;
  const auto results = Trace(model, 4, kSites, 9);
  ASSERT_TRUE(results.ok()) << results.status();
  for (const auto& result : *results) {
    EXPECT_EQ(result.targets, 6);
    EXPECT_EQ(result.changed_prefixes, 0);
    EXPECT_EQ(result.clean_correct, 6);
    EXPECT_EQ(result.corrupt_correct, 6);
    EXPECT_EQ(result.rescue_correct, 6);
    EXPECT_EQ(result.damage_correct, 6);
    EXPECT_EQ(result.baseline_wrong, 0);
    EXPECT_EQ(result.rescued, 0);
    EXPECT_EQ(result.newly_broken_by_rescue, 0);
    EXPECT_EQ(result.damaged, 0);
  }
}

TEST_F(CausalTraceTest,
       CountsNewlyBrokenCasesAgainstCorruptCorrectDenominator) {
  ToyCausalModel model;
  model.reverse_position_one = true;
  const auto results = Trace(model);
  ASSERT_TRUE(results.ok()) << results.status();
  const auto& late = (*results)[1];
  EXPECT_EQ(late.targets, 6);
  EXPECT_EQ(late.clean_correct, 4);
  EXPECT_EQ(late.corrupt_correct, 6);
  EXPECT_EQ(late.baseline_wrong, 0);
  EXPECT_EQ(late.rescued, 0);
  EXPECT_EQ(late.newly_broken_by_rescue, 2);
  EXPECT_EQ(late.rescue_correct, 4);
  EXPECT_EQ(late.damage_correct, 6);
  EXPECT_EQ(late.damaged, 0);
}

TEST_F(CausalTraceTest, NonfiniteRowsAreWrongAndReportedInEachCondition) {
  ToyCausalModel model;
  model.nonfinite_for_first_two = true;
  const auto results = Trace(model);
  ASSERT_TRUE(results.ok()) << results.status();
  const auto& late = (*results)[1];
  EXPECT_EQ(late.targets, 6);
  EXPECT_EQ(late.clean_correct, 5);
  EXPECT_EQ(late.corrupt_correct, 4);
  EXPECT_EQ(late.baseline_wrong, 2);
  EXPECT_EQ(late.rescued, 1);
  EXPECT_EQ(late.damaged, 1);
  EXPECT_EQ(late.rescue_correct, 5);
  EXPECT_EQ(late.damage_correct, 4);
  EXPECT_EQ(late.clean_nonfinite, 1);
  EXPECT_EQ(late.corrupt_nonfinite, 1);
  EXPECT_EQ(late.rescue_nonfinite, 1);
  EXPECT_EQ(late.damage_nonfinite, 1);
}

TEST_F(CausalTraceTest, MissingAmbiguousAndDuplicateSitesFailExplicitly) {
  ToyCausalModel model;
  for (const auto& site :
       std::vector<CausalTraceSite>{{.layer_name = "missing"},
                                    {.layer_name = "ResidualLayer",
                                     .enclosing_scope = "wrong_scope",
                                     .occurrence = 0},
                                    {.layer_name = "ResidualLayer",
                                     .enclosing_scope = "transformer_block_0",
                                     .occurrence = 2}}) {
    EXPECT_EQ(Trace(model, 4, {&site, 1}).status().code(),
              absl::StatusCode::kNotFound);
  }
  const CausalTraceSite ambiguous{.layer_name = "ResidualLayer"};
  EXPECT_EQ(Trace(model, 4, {&ambiguous, 1}).status().code(),
            absl::StatusCode::kInvalidArgument);
  const std::vector<CausalTraceSite> duplicate{kSites[0], kSites[0]};
  EXPECT_EQ(Trace(model, 4, duplicate).status().code(),
            absl::StatusCode::kInvalidArgument);
  for (const auto& site :
       std::vector<CausalTraceSite>{{}, {.layer_name = "x", .occurrence = -2}})
    EXPECT_EQ(Trace(model, 4, {&site, 1}).status().code(),
              absl::StatusCode::kInvalidArgument);
}

TEST_F(CausalTraceTest, RejectsMalformedActivationShapesAndUnbalancedScopes) {
  for (const auto& signature : std::vector<ActivationType>{
           {DataType::INT32, {kBatch, kContext, kWidth}},
           {DataType::FP32, {kBatch, kContext, kWidth + 1}},
           {DataType::FP32, {kBatch, kContext + 1, kWidth}},
           {DataType::FP32, {kBatch, kWidth}}}) {
    ToyCausalModel model;
    model.early.signature_ = signature;
    EXPECT_EQ(Trace(model).status().code(), absl::StatusCode::kInvalidArgument);
  }
  ToyCausalModel model;
  model.skip_scope_exit = true;
  EXPECT_EQ(Trace(model).status().code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST_F(CausalTraceTest, RejectsInvalidInputsBeforeForward) {
  ToyCausalModel model;
  EXPECT_EQ(Trace(model, 0).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(Trace(model, 4, {}).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(Trace(model, 4, kSites, -1).status().code(),
            absl::StatusCode::kInvalidArgument);
  for (const auto& corpus : std::vector<Corpus>{{},
                                                {{1, 3}},
                                                {{}, {1}},
                                                {{1, 0}, {2}},
                                                {{1, -1}, {2}},
                                                {{1, 7}, {2}},
                                                {{1, 2, 3, 4, 5}, {2}}}) {
    EXPECT_EQ(TracePrefixMediation(*executor_, model, corpus, kVocabulary, kEos,
                                   1, 4, kSites, 1)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
  EXPECT_EQ(TracePrefixMediation(*executor_, model, kCorpus, kVocabulary, kEos,
                                 0, 4, kSites, 1)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(TracePrefixMediation(*executor_, model, kCorpus, kVocabulary,
                                 kVocabulary, 1, 4, kSites, 1)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(model.observations.empty());
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
