#include "src/llm/experiments/one_shot_memorizer/mlp_subset_tail.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/llm/experiments/one_shot_memorizer/token_trace.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

constexpr int kVocabulary = 31;  // Exercise both physical table/logit padding.
const Gpt2Config kConfig{.transformer_block_count = 1,
                         .model_width = 16,
                         .attention_heads = 1,
                         .feed_forward_width = 64,
                         .vocabulary_size = kVocabulary,
                         .pad_vocabulary = false};
const TokenTraceSite kGelu{
    .scope = {"gpt2", "transformer_block_0", "ResidualLayer", "mlp"},
    .layer_name = "GeluLayer",
    .occurrence = 0};
const TokenTraceSite kResidual{.scope = {"gpt2", "transformer_block_0"},
                               .layer_name = "ResidualLayer",
                               .occurrence = 0};
const std::array<int, 5> kPrefix{1, 5, 7, 11, 13};

absl::StatusOr<TokenTraceActivation> Find(const TokenTraceResult& trace,
                                          const TokenTraceSite& wanted) {
  for (const auto& activation : trace.activations)
    if (activation.site.scope == wanted.scope &&
        activation.site.layer_name == wanted.layer_name &&
        activation.site.occurrence == wanted.occurrence &&
        activation.site.output_index == wanted.output_index)
      return activation;
  return absl::NotFoundError("missing cached-tail test activation");
}

template <size_t Width>
absl::StatusOr<std::array<uint16_t, Width>> Query(
    const TokenTraceActivation& activation) {
  if (activation.data_type != DataType::BF16 || activation.channels != Width ||
      activation.first_row > 4 ||
      activation.first_row + activation.row_count <= 4 ||
      activation.bytes.size() !=
          activation.row_count * Width * sizeof(uint16_t))
    return absl::InvalidArgumentError("invalid cached-tail test activation");
  std::array<uint16_t, Width> values;
  std::memcpy(values.data(),
              activation.bytes.data() +
                  (4 - activation.first_row) * Width * sizeof(uint16_t),
              sizeof(values));
  return values;
}

absl::StatusOr<std::vector<uint8_t>> Snapshot(cuda::Executor& executor,
                                              const Layer& model) {
  std::vector<uint8_t> result;
  for (const auto& weight : model.weights()) {
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<uint8_t>::Allocate(
                                    executor, weight.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), weight.data(), host.size_bytes(),
                        cudaMemcpyDeviceToHost, executor.stream()),
        "snapshot cached-tail test weight"));
    RETURN_IF_ERROR(executor.Synchronize());
    result.insert(result.end(), host.begin(), host.end());
  }
  return result;
}

class MlpSubsetTailTest : public testing::Test {
 protected:
  void SetUp() override {
    auto created = cuda::Executor::Create();
    ASSERT_TRUE(created.ok()) << created.status();
    executor_ = std::move(*created);
    auto model = CreateGpt2(*executor_, DataType::BF16, 42, kConfig);
    ASSERT_TRUE(model.ok()) << model.status();
    model_ = std::move(*model);
  }
  std::unique_ptr<cuda::Executor> executor_;
  std::unique_ptr<ComposedLayer> model_;
};

TEST_F(MlpSubsetTailTest,
       ProductionMasksMatchFullForwardAcrossBatchGeometries) {
  // Initialization uses zero biases. Give the contraction and norm nonzero
  // affine parameters so equivalence also detects accidentally dropping them,
  // especially when the retained GELU subset is empty.
  const auto weights = model_->weights();
  for (int parameter = 0; parameter < 3; ++parameter) {
    auto values = cuda::PageLockedHostArray<float>::Allocate(*executor_, 16);
    ASSERT_TRUE(values.ok()) << values.status();
    for (int dim = 0; dim < 16; ++dim)
      (*values)[dim] = parameter == 0   ? 0.002f * (dim - 5)
                       : parameter == 1 ? 1.0f + 0.03f * dim
                                        : 0.015f * (dim - 7);
    auto status = cuda::CudaStatus(
        cudaMemcpyAsync(weights[weights.size() - 4 + parameter].data(),
                        values->data(), values->size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "initialize nonzero cached-tail test affine parameters");
    ASSERT_TRUE(status.ok()) << status;
  }
  auto before = Snapshot(*executor_, *model_);
  ASSERT_TRUE(before.ok()) << before.status();
  const TokenTraceOptions capture{.vocabulary_size = kVocabulary,
                                  .padding_token = 0,
                                  .capture_activations = true};
  auto baseline = TraceNextToken(*executor_, *model_, kPrefix, capture);
  ASSERT_TRUE(baseline.ok()) << baseline.status();
  auto gelu_activation = Find(*baseline, kGelu);
  ASSERT_TRUE(gelu_activation.ok()) << gelu_activation.status();
  auto residual_activation = Find(*baseline, kResidual);
  ASSERT_TRUE(residual_activation.ok()) << residual_activation.status();
  auto gelu = Query<64>(*gelu_activation);
  auto residual = Query<16>(*residual_activation);
  ASSERT_TRUE(gelu.ok()) << gelu.status();
  ASSERT_TRUE(residual.ok()) << residual.status();
  const auto original_gelu = *gelu;
  const auto original_residual = *residual;
  const std::array<FeatureSubset, 4> masks{
      0, ~FeatureSubset{0}, FeatureSubset{1} << 63,
      (FeatureSubset{1} << 12) | (FeatureSubset{1} << 34)};
  std::array<std::vector<float>, 4> expected;
  for (size_t index = 0; index < masks.size(); ++index) {
    auto donor = *gelu_activation;
    auto masked = ApplyBf16FeatureSubset(*gelu, masks[index]);
    ASSERT_TRUE(masked.ok()) << masked.status();
    std::memcpy(
        donor.bytes.data() + (4 - donor.first_row) * 64 * sizeof(uint16_t),
        masked->data(), 64 * sizeof(uint16_t));
    const TokenTracePatch patch{.site = kGelu,
                                .replacement = TokenTraceReplacement::kDonor,
                                .donor = &donor};
    const TokenTraceOptions options{.vocabulary_size = kVocabulary,
                                    .padding_token = 0,
                                    .patches = {&patch, 1}};
    auto result = TraceNextToken(*executor_, *model_, kPrefix, options);
    ASSERT_TRUE(result.ok()) << result.status();
    expected[index] = std::move(result->logits);
  }
  EXPECT_EQ(std::memcmp(expected[1].data(), baseline->logits.data(),
                        kVocabulary * sizeof(float)),
            0);
  auto tail = FinalMlpSubsetTail::Create(*executor_, *model_, kConfig);
  ASSERT_TRUE(tail.ok()) << tail.status();
  EXPECT_EQ((*tail)->vocab_size(), kVocabulary);
  EXPECT_EQ((*tail)->logit_stride(), 32);
  // Cross the dense MMA's 64-row tile and norm's16-row tile; the final case
  // matches a full2016-pair enumeration plus its same-geometry identity row.
  for (size_t count :
       {size_t{1}, size_t{6}, size_t{17}, size_t{64}, size_t{2017}}) {
    std::vector<FinalMlpSubsetInput> inputs;
    for (size_t row = 0; row < count; ++row)
      inputs.push_back({*gelu, *residual, masks[row % masks.size()]});
    auto logits = (*tail)->Evaluate(*executor_, inputs);
    ASSERT_TRUE(logits.ok()) << logits.status();
    EXPECT_EQ(logits->size(), ((count + 15) / 16 * 16) * 32);
    for (size_t row = 0; row < count; ++row) {
      SCOPED_TRACE(testing::Message() << "batch=" << count << " row=" << row);
      EXPECT_EQ(std::memcmp(logits->data() + row * 32,
                            expected[row % masks.size()].data(),
                            kVocabulary * sizeof(float)),
                0);
      EXPECT_EQ((*logits)[row * 32 + kVocabulary],
                std::numeric_limits<float>::lowest());
    }
  }
  EXPECT_EQ(*gelu, original_gelu);
  EXPECT_EQ(*residual, original_residual);
  auto after = Snapshot(*executor_, *model_);
  ASSERT_TRUE(after.ok()) << after.status();
  EXPECT_EQ(*before, *after);
  auto unpatched =
      TraceNextToken(*executor_, *model_, kPrefix,
                     {.vocabulary_size = kVocabulary, .padding_token = 0});
  ASSERT_TRUE(unpatched.ok()) << unpatched.status();
  EXPECT_EQ(std::memcmp(unpatched->logits.data(), baseline->logits.data(),
                        kVocabulary * sizeof(float)),
            0);
}

TEST_F(MlpSubsetTailTest, RejectsMalformedInputsAndDifferentExecutor) {
  auto tail = FinalMlpSubsetTail::Create(*executor_, *model_, kConfig);
  ASSERT_TRUE(tail.ok()) << tail.status();
  EXPECT_EQ((*tail)->Evaluate(*executor_, {}).status().code(),
            absl::StatusCode::kInvalidArgument);
  std::array<uint16_t, 64> gelu{};
  std::array<uint16_t, 16> residual{};
  std::array<FinalMlpSubsetInput, 1> inputs{{{gelu, residual, 0}}};
  inputs[0].gelu = absl::MakeConstSpan(gelu).first(63);
  EXPECT_EQ((*tail)->Evaluate(*executor_, inputs).status().code(),
            absl::StatusCode::kInvalidArgument);
  inputs[0].gelu = gelu;
  inputs[0].residual = absl::MakeConstSpan(residual).first(15);
  EXPECT_EQ((*tail)->Evaluate(*executor_, inputs).status().code(),
            absl::StatusCode::kInvalidArgument);
  inputs[0].residual = residual;
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  EXPECT_EQ((*tail)->Evaluate(**other, inputs).status().code(),
            absl::StatusCode::kInvalidArgument);
  // Validation failures cannot leave the scoped cached-GELU pointer poisoned.
  EXPECT_TRUE((*tail)->Evaluate(*executor_, inputs).ok());
}

TEST_F(MlpSubsetTailTest, RejectsUnsupportedConfiguration) {
  for (int variant = 0; variant < 5; ++variant) {
    auto config = kConfig;
    if (variant == 0)
      config.model_width = 32;
    if (variant == 1)
      config.feed_forward_width = 32;
    if (variant == 2)
      config.transformer_block_count = 0;
    if (variant == 3)
      config.pad_vocabulary = true;
    if (variant == 4)
      config.vocabulary_size = 32;
    EXPECT_FALSE(FinalMlpSubsetTail::Create(*executor_, *model_, config).ok());
  }
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  EXPECT_FALSE(FinalMlpSubsetTail::Create(**other, *model_, kConfig).ok());
}

// A read-only model facade lets validation tests corrupt the inventory rather
// than touching device values or depending on a malformed executable model.
class InventoryView final : public Layer {
 public:
  explicit InventoryView(const Layer& source)
      : source_(source),
        inventory(source.weights().begin(), source.weights().end()) {}
  absl::string_view name() const override { return "InventoryView"; }
  absl::Span<const ActivationType> input_types() const override {
    return source_.input_types();
  }
  absl::Span<const ActivationType> output_types() const override {
    return source_.output_types();
  }
  absl::Span<Buffer> weights() override { return absl::MakeSpan(inventory); }
  DataType output_type() const override { return type; }
  const Layer& source_;
  std::vector<Buffer> inventory;
  DataType type = DataType::BF16;

 private:
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     LayerHooks*) const override {
    return absl::UnimplementedError("inventory only");
  }
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&, absl::Span<const Buffer>,
                                     BackwardState, LayerHooks*) override {
    return absl::UnimplementedError("inventory only");
  }
};

TEST_F(MlpSubsetTailTest, RejectsMalformedMasterInventoryAndComputeType) {
  for (int variant = 0; variant < 5; ++variant) {
    InventoryView view(*model_);
    if (variant == 0)
      view.inventory.pop_back();
    if (variant == 1)
      view.inventory.back() = view.inventory[1];
    if (variant == 2)
      view.inventory[0] = view.inventory[2];
    if (variant == 3)
      view.inventory[2] = view.inventory[3];
    if (variant == 4)
      view.type = DataType::FP16;
    EXPECT_FALSE(FinalMlpSubsetTail::Create(*executor_, view, kConfig).ok());
  }
}

TEST_F(MlpSubsetTailTest, TailDoesNotBorrowOriginalModelWeights) {
  auto tail = FinalMlpSubsetTail::Create(*executor_, *model_, kConfig);
  ASSERT_TRUE(tail.ok()) << tail.status();
  std::array<uint16_t, 64> gelu{};
  std::array<uint16_t, 16> residual{};
  const std::array<FinalMlpSubsetInput, 1> input{{{gelu, residual, 0}}};
  auto before = (*tail)->Evaluate(*executor_, input);
  ASSERT_TRUE(before.ok()) << before.status();
  // Destroying the source must not invalidate the tail's copied masters.
  model_.reset();
  auto after = (*tail)->Evaluate(*executor_, input);
  ASSERT_TRUE(after.ok()) << after.status();
  ASSERT_EQ(before->size(), after->size());
  EXPECT_EQ(std::memcmp(before->data(), after->data(), before->size_bytes()),
            0);
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
