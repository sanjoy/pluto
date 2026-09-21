#include "src/llm/experiments/shakespeare/activation_inspection.h"

#include <cuda_runtime_api.h>

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layers/combinators.h"
#include "src/util/status_macros.h"

namespace pluto::llm {
namespace {

constexpr int kWidth = 32;
constexpr int kVocabulary = 4;
constexpr int kPaddedVocabulary = 16;
constexpr int kSequence = 4;

static_assert(!std::is_base_of_v<LayerHooks, NeighboringVocabInspector>);
static_assert(!std::is_copy_constructible_v<NeighboringVocabInspector>);
static_assert(!std::is_move_constructible_v<NeighboringVocabInspector>);

template <class T>
absl::StatusOr<Buffer> Upload(cuda::Executor& executor,
                              const std::vector<T>& values) {
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<T>::CopyFrom(
                                  executor, absl::MakeConstSpan(values)));
  ASSIGN_OR_RETURN(auto device, Buffer::Allocate(executor, host.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(device.data(), host.data(), host.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "cudaMemcpyAsync(test upload)"));
  return device;
}

template <class T>
absl::StatusOr<std::vector<T>> Download(cuda::Executor& executor,
                                        const Buffer& device) {
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<T>::Allocate(
                                  executor, device.size_bytes() / sizeof(T)));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host.data(), device.data(), device.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "cudaMemcpyAsync(test download)"));
  RETURN_IF_ERROR(executor.Synchronize());
  return std::vector<T>(host.begin(), host.end());
}

// Four byte-level tokens exercise whitespace, quoting, backslashes, and NUL.
class TestDetokenizer final : public tokenizer::Detokenizer {
 public:
  absl::StatusOr<std::string> Decode(
      absl::Span<const int> token_ids) const override {
    const std::string vocabulary[] = {"\n", "\"", "\\", std::string("x\0y", 3)};
    std::string result;
    for (int id : token_ids) {
      if (id < 0 || id >= kVocabulary)
        return absl::InvalidArgumentError("test token is outside vocabulary");
      result += vocabulary[id];
    }
    return result;
  }
  int vocab_size() const override { return kVocabulary; }
};

class IdentityLayer final : public Layer {
 public:
  explicit IdentityLayer(DataType type = DataType::FP32)
      : type_(type, {ActivationType::kBatchDimension, kSequence, kWidth}) {}
  absl::string_view name() const override { return "IdentityLayer"; }
  absl::Span<const ActivationType> input_types() const override {
    return {&type_, 1};
  }
  absl::Span<const ActivationType> output_types() const override {
    return input_types();
  }
  absl::Span<Buffer> weights() override { return {}; }
  DataType output_type() const override { return type_.data_type(); }

 private:
  absl::StatusOr<FwdResult> fwd_impl(cuda::Executor&,
                                     absl::Span<const Buffer> inputs,
                                     LayerHooks*) const override {
    return FwdResult{.outputs = {inputs[0]},
                     .state = {.intermediates = {inputs[0]}}};
  }
  absl::StatusOr<BufferVec> bwd_impl(cuda::Executor&,
                                     absl::Span<const Buffer> gradients,
                                     BackwardState, LayerHooks*) override {
    return BufferVec(gradients.begin(), gradients.end());
  }
  ActivationType type_;
};

class ActivationInspectionTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }
  void TearDown() override {
    if (executor_ != nullptr)
      EXPECT_TRUE(executor_->Synchronize().ok());
  }
  std::vector<float> EmbeddingValues() const {
    std::vector<float> values(kPaddedVocabulary * kWidth, 1000.0f);
    // The logical vocabulary is four basis vectors; padding must be ignored
    // even though its large logits would otherwise dominate the softmax.
    for (int row = 0; row < kVocabulary; ++row)
      for (int col = 0; col < kWidth; ++col)
        values[row * kWidth + col] = row == col ? 1.0f : 0.0f;
    return values;
  }
  std::vector<float> ActivationValues() const {
    std::vector<float> values(kSequence * kWidth, 0.0f);
    values[kWidth + 3] = 2.0f;
    // The final two rows represent padding, not real prompt tokens.
    values[2 * kWidth + 3] = 1000.0f;
    values[3 * kWidth + 3] = 1000.0f;
    return values;
  }
  TestDetokenizer detokenizer_;
  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(ActivationInspectionTest, ProjectsFp32AndBf16WithoutMutatingStorage) {
  const auto embedding_values = EmbeddingValues();
  auto embedding = Upload(*executor_, embedding_values);
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  const auto activation_values = ActivationValues();
  std::string fp32_output;
  for (DataType type : {DataType::FP32, DataType::BF16}) {
    auto input = Upload(*executor_, activation_values);
    ASSERT_TRUE(input.ok()) << input.status();
    if (type == DataType::BF16) {
      std::vector<uint16_t> bf16;
      for (float value : activation_values)
        bf16.push_back(
            static_cast<uint16_t>(std::bit_cast<uint32_t>(value) >> 16));
      input = Upload(*executor_, bf16);
      ASSERT_TRUE(input.ok()) << input.status();
    }
    auto inspector = NeighboringVocabInspector::Create(
        *executor_, *embedding, kVocabulary, kWidth, 2, 7);
    ASSERT_TRUE(inspector.ok()) << inspector.status();
    IdentityLayer layer(type);
    auto forward =
        layer.fwd(*executor_, {*input}, &(*inspector)->layer_hooks());
    ASSERT_TRUE(forward.ok()) << forward.status();
    EXPECT_EQ(forward->outputs[0].data(), input->data());
    EXPECT_EQ(forward->state.intermediates[0].data(), input->data());
    std::ostringstream output;
    ASSERT_TRUE(
        (*inspector)->Print(*executor_, detokenizer_, {0, 1}, output).ok());
    const std::string text = output.str();
    EXPECT_NE(text.find("Position 7 (\"\\n\"):\n"), std::string::npos);
    EXPECT_NE(text.find("Position 8 (\"\\\"\"):\n"), std::string::npos);
    EXPECT_EQ(text.find("Position 9 ("), std::string::npos);
    // Uniform logits must report 25%, not 33% after top-three renormalization.
    EXPECT_NE(text.find(R"("\n" (25.00%), "\"" (25.00%), "\\" (25.00%))"),
              std::string::npos);
    EXPECT_NE(text.find(R"("x\000y" (71.12%), "\n" (9.63%), "\"" (9.63%))"),
              std::string::npos);
    if (type == DataType::FP32) {
      fp32_output = text;
      auto original = Download<float>(*executor_, *input);
      ASSERT_TRUE(original.ok()) << original.status();
      EXPECT_EQ(*original, activation_values);
    } else {
      EXPECT_EQ(text, fp32_output);
    }
  }
  auto unchanged_embedding = Download<float>(*executor_, *embedding);
  ASSERT_TRUE(unchanged_embedding.ok()) << unchanged_embedding.status();
  EXPECT_EQ(*unchanged_embedding, embedding_values);
}

TEST_F(ActivationInspectionTest,
       ReadsLogicalOrPaddedLogitsWithoutReprojection) {
  auto embedding = Upload(*executor_, EmbeddingValues());
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  for (const int stride : {kVocabulary, kPaddedVocabulary}) {
    auto inspector = NeighboringVocabInspector::Create(*executor_, *embedding,
                                                       kVocabulary, kWidth, 1);
    ASSERT_TRUE(inspector.ok()) << inspector.status();
    std::vector<float> logits(kSequence * stride, 1000.0f);
    for (int token = 0; token < kVocabulary; ++token)
      logits[token] = 0.0f;
    auto buffer = Upload(*executor_, logits);
    ASSERT_TRUE(buffer.ok()) << buffer.status();
    BufferVec outputs{*buffer};
    const ActivationType type(DataType::FP32, {-2, kSequence, stride});
    ASSERT_TRUE((*inspector)
                    ->layer_hooks()
                    .ActivationHook(*executor_, "LanguageModelingHeadLayer",
                                    {&type, 1}, absl::MakeSpan(outputs))
                    .ok());
    std::ostringstream output;
    ASSERT_TRUE(
        (*inspector)->Print(*executor_, detokenizer_, {0}, output).ok());
    EXPECT_NE(
        output.str().find(R"("\n" (25.00%), "\"" (25.00%), "\\" (25.00%))"),
        std::string::npos);
    EXPECT_EQ(output.str().find("Position 1 ("), std::string::npos);
  }
}

TEST_F(ActivationInspectionTest,
       PreservesSmallNonzeroProbabilitiesAndExactZero) {
  auto embedding = Upload(*executor_, EmbeddingValues());
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  auto inspector = NeighboringVocabInspector::Create(
      *executor_, *embedding, kVocabulary, kWidth, 2, 0, 0.0);
  ASSERT_TRUE(inspector.ok()) << inspector.status();
  std::vector<float> logits(kSequence * kPaddedVocabulary, 0.0f);
  logits[0] = 10.0f;
  // The second row distinguishes genuinely zero, underflowed probabilities
  // from the first row's small but representable probabilities.
  for (int token = 1; token < kVocabulary; ++token)
    logits[kPaddedVocabulary + token] = -1000.0f;
  auto buffer = Upload(*executor_, logits);
  ASSERT_TRUE(buffer.ok()) << buffer.status();
  BufferVec outputs{*buffer};
  const ActivationType type(DataType::FP32, {-2, kSequence, kPaddedVocabulary});
  ASSERT_TRUE((*inspector)
                  ->layer_hooks()
                  .ActivationHook(*executor_, "Logits", {&type, 1},
                                  absl::MakeSpan(outputs))
                  .ok());
  std::ostringstream output;
  ASSERT_TRUE(
      (*inspector)->Print(*executor_, detokenizer_, {0, 0}, output).ok());
  EXPECT_NE(
      output.str().find(R"("\n" (99.99%), "\"" (0.004539%), "\\" (0.004539%))"),
      std::string::npos);
  EXPECT_NE(output.str().find(R"("\n" (100.00%), "\"" (0.00%), "\\" (0.00%))"),
            std::string::npos);
}

TEST_F(ActivationInspectionTest, NonFiniteRowDoesNotDiscardOtherPositions) {
  auto embedding = Upload(*executor_, EmbeddingValues());
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  auto inspector = NeighboringVocabInspector::Create(*executor_, *embedding,
                                                     kVocabulary, kWidth, 2);
  ASSERT_TRUE(inspector.ok()) << inspector.status();
  std::vector<float> logits(kSequence * kPaddedVocabulary, 0.0f);
  logits[0] = std::numeric_limits<float>::infinity();
  auto buffer = Upload(*executor_, logits);
  ASSERT_TRUE(buffer.ok()) << buffer.status();
  BufferVec outputs{*buffer};
  const ActivationType type(DataType::FP32, {-2, kSequence, kPaddedVocabulary});
  ASSERT_TRUE((*inspector)
                  ->layer_hooks()
                  .ActivationHook(*executor_, "Logits", {&type, 1},
                                  absl::MakeSpan(outputs))
                  .ok());
  std::ostringstream output;
  ASSERT_TRUE(
      (*inspector)->Print(*executor_, detokenizer_, {0, 0}, output).ok());
  EXPECT_EQ(output.str().find("Position 0 ("), std::string::npos);
  EXPECT_EQ(output.str().find("not applicable"), std::string::npos);
  EXPECT_NE(
      output.str().find("Position 1 (\"\\n\"):\n  Logits[0]: \"\\n\" (25.00%)"),
      std::string::npos);
}

TEST_F(ActivationInspectionTest,
       SuppressesUnsupportedOutputsAndEmptyPositions) {
  auto embedding = Upload(*executor_, EmbeddingValues());
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  auto inspector = NeighboringVocabInspector::Create(*executor_, *embedding,
                                                     kVocabulary, kWidth, 2);
  ASSERT_TRUE(inspector.ok()) << inspector.status();
  auto wide = Upload(*executor_, std::vector<float>(kSequence * 2 * kWidth));
  auto ids = Upload(*executor_, std::vector<int32_t>(kSequence * kWidth));
  auto scalar = Upload(*executor_, std::vector<float>{0.0f});
  auto scalar_rows = Upload(*executor_, std::vector<float>(kWidth));
  ASSERT_TRUE(wide.ok()) << wide.status();
  ASSERT_TRUE(ids.ok()) << ids.status();
  ASSERT_TRUE(scalar.ok()) << scalar.status();
  ASSERT_TRUE(scalar_rows.ok()) << scalar_rows.status();
  BufferVec outputs{*wide, *ids, *scalar, *scalar_rows};
  const std::vector<ActivationType> types = {
      {DataType::FP32, {-2, kSequence, 2 * kWidth}},
      {DataType::INT32, {-2, kSequence, kWidth}},
      {DataType::FP32, {}},
      {DataType::FP32, {-2, kWidth}}};
  ASSERT_TRUE((*inspector)
                  ->layer_hooks()
                  .ActivationHook(*executor_, "MultiOutput", types,
                                  absl::MakeSpan(outputs))
                  .ok());
  std::ostringstream output;
  ASSERT_TRUE(
      (*inspector)->Print(*executor_, detokenizer_, {0, 0}, output).ok());
  EXPECT_TRUE(output.str().empty());
}

TEST_F(ActivationInspectionTest, DoesNotGuessWhenHiddenAndLogitWidthsCoincide) {
  auto embedding = Upload(*executor_, std::vector<float>(16 * 16));
  auto input = Upload(*executor_, std::vector<float>(kSequence * 16));
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  ASSERT_TRUE(input.ok()) << input.status();
  auto inspector = NeighboringVocabInspector::Create(*executor_, *embedding,
                                                     kVocabulary, 16, 1);
  ASSERT_TRUE(inspector.ok()) << inspector.status();
  const ActivationType type(DataType::FP32, {-2, kSequence, 16});
  BufferVec outputs{*input};
  ASSERT_TRUE((*inspector)
                  ->layer_hooks()
                  .ActivationHook(*executor_, "Ambiguous", {&type, 1},
                                  absl::MakeSpan(outputs))
                  .ok());
  std::ostringstream output;
  ASSERT_TRUE((*inspector)->Print(*executor_, detokenizer_, {0}, output).ok());
  EXPECT_TRUE(output.str().empty());
}

TEST_F(ActivationInspectionTest, HeadingsUseEscapedInputTokensAtWindowOffset) {
  auto embedding = Upload(*executor_, EmbeddingValues());
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  auto inspector = NeighboringVocabInspector::Create(
      *executor_, *embedding, kVocabulary, kWidth, kSequence, 101);
  ASSERT_TRUE(inspector.ok()) << inspector.status();
  std::vector<float> logits(kSequence * kPaddedVocabulary, -1000.0f);
  for (int row = 0; row < kSequence; ++row)
    logits[row * kPaddedVocabulary] = 0.0f;
  auto buffer = Upload(*executor_, logits);
  ASSERT_TRUE(buffer.ok()) << buffer.status();
  BufferVec outputs{*buffer};
  const ActivationType type(DataType::FP32, {-2, kSequence, kPaddedVocabulary});
  ASSERT_TRUE((*inspector)
                  ->layer_hooks()
                  .ActivationHook(*executor_, "Logits", {&type, 1},
                                  absl::MakeSpan(outputs))
                  .ok());

  // These are exactly the four IDs in the cropped window, not the preceding
  // 101 tokens. Each heading names the input, even when its top readout is a
  // different token. All four troublesome byte spellings must stay on one line.
  const std::array<int, kSequence> window{3, 1, 0, 2};
  std::ostringstream output;
  ASSERT_TRUE(
      (*inspector)->Print(*executor_, detokenizer_, window, output).ok());
  EXPECT_EQ(output.str(), R"(Position 101 ("x\000y"):
  Logits[0]: "\n" (100.00%)
Position 102 ("\""):
  Logits[0]: "\n" (100.00%)
Position 103 ("\n"):
  Logits[0]: "\n" (100.00%)
Position 104 ("\\"):
  Logits[0]: "\n" (100.00%)
)");
}

TEST_F(ActivationInspectionTest,
       ValidatesInputTokensEvenWhenAllRowsAreSuppressed) {
  auto embedding = Upload(*executor_, EmbeddingValues());
  auto buffer =
      Upload(*executor_, std::vector<float>(kSequence * kPaddedVocabulary));
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  ASSERT_TRUE(buffer.ok()) << buffer.status();
  auto inspector = NeighboringVocabInspector::Create(
      *executor_, *embedding, kVocabulary, kWidth, 2, 17, 1.0);
  ASSERT_TRUE(inspector.ok()) << inspector.status();
  BufferVec outputs{*buffer};
  const ActivationType type(DataType::FP32, {-2, kSequence, kPaddedVocabulary});
  ASSERT_TRUE((*inspector)
                  ->layer_hooks()
                  .ActivationHook(*executor_, "Logits", {&type, 1},
                                  absl::MakeSpan(outputs))
                  .ok());

  const std::vector<std::vector<int>> invalid_windows = {
      {}, {0}, {0, 1, 2}, {-1, 0}, {0, -1}, {kVocabulary, 0}, {0, kVocabulary}};
  for (const auto& tokens : invalid_windows) {
    std::ostringstream output;
    EXPECT_EQ(
        (*inspector)->Print(*executor_, detokenizer_, tokens, output).code(),
        absl::StatusCode::kInvalidArgument);
    EXPECT_TRUE(output.str().empty());
  }
  std::ostringstream output;
  ASSERT_TRUE(
      (*inspector)->Print(*executor_, detokenizer_, {3, 1}, output).ok());
  // Uniform probabilities are all below one, so valid input headings must
  // still disappear along with their filtered layers.
  EXPECT_TRUE(output.str().empty());
}

TEST_F(ActivationInspectionTest, InclusiveThresholdDoesNotRenormalizeTopThree) {
  auto embedding = Upload(*executor_, EmbeddingValues());
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  std::vector<float> logits(kSequence * kPaddedVocabulary, 0.0f);
  for (int token = 1; token < kVocabulary; ++token)
    logits[kPaddedVocabulary + token] = -1000.0f;
  logits[2 * kPaddedVocabulary] = 10.0f;
  auto buffer = Upload(*executor_, logits);
  ASSERT_TRUE(buffer.ok()) << buffer.status();
  BufferVec outputs{*buffer};
  const ActivationType type(DataType::FP32, {-2, kSequence, kPaddedVocabulary});
  std::string default_output;
  // The negative value selects the default argument; it is never passed in.
  for (double threshold : {-1.0, 0.01, 0.25, 0.250001, 1.0, 0.0}) {
    auto inspector =
        threshold < 0.0
            ? NeighboringVocabInspector::Create(*executor_, *embedding,
                                                kVocabulary, kWidth, 3)
            : NeighboringVocabInspector::Create(
                  *executor_, *embedding, kVocabulary, kWidth, 3, 0, threshold);
    ASSERT_TRUE(inspector.ok()) << inspector.status();
    ASSERT_TRUE((*inspector)
                    ->layer_hooks()
                    .ActivationHook(*executor_, "Logits", {&type, 1},
                                    absl::MakeSpan(outputs))
                    .ok());
    std::ostringstream output;
    ASSERT_TRUE(
        (*inspector)->Print(*executor_, detokenizer_, {0, 0, 0}, output).ok());
    const std::string text = output.str();
    if (threshold < 0.0) {
      default_output = text;
      EXPECT_EQ(text.find("0.004539%"), std::string::npos);
      EXPECT_NE(text.find("99.99%"), std::string::npos);
    } else if (threshold == 0.01) {
      EXPECT_EQ(text, default_output);
    }
    if (threshold <= 0.25) {
      EXPECT_NE(text.find("Position 0 (\"\\n\"):"), std::string::npos);
      EXPECT_NE(text.find(R"("\n" (25.00%), "\"" (25.00%), "\\" (25.00%))"),
                std::string::npos);
    } else {
      EXPECT_EQ(text.find("Position 0 ("), std::string::npos);
    }
    if (threshold == 1.0)
      EXPECT_EQ(text,
                "Position 1 (\"\\n\"):\n  Logits[0]: \"\\n\" (100.00%)\n");
    if (threshold == 0.0) {
      EXPECT_NE(text.find("0.004539%"), std::string::npos);
      EXPECT_NE(text.find(R"("\n" (100.00%), "\"" (0.00%), "\\" (0.00%))"),
                std::string::npos);
    }
  }
}

TEST_F(ActivationInspectionTest, SuppressedLayersStillConsumeSiblingIndices) {
  auto embedding = Upload(*executor_, EmbeddingValues());
  auto uniform = Upload(*executor_, std::vector<float>(kSequence * kWidth));
  auto wide = Upload(*executor_, std::vector<float>(kSequence * 2 * kWidth));
  auto values = ActivationValues();
  values[3] = 2.0f;
  auto strong = Upload(*executor_, values);
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  ASSERT_TRUE(uniform.ok()) << uniform.status();
  ASSERT_TRUE(wide.ok()) << wide.status();
  ASSERT_TRUE(strong.ok()) << strong.status();
  auto inspector = NeighboringVocabInspector::Create(
      *executor_, *embedding, kVocabulary, kWidth, 1, 0, 0.5);
  ASSERT_TRUE(inspector.ok()) << inspector.status();
  LayerHooks& hooks = (*inspector)->layer_hooks();
  const ActivationType type(DataType::FP32, {-2, kSequence, kWidth});
  const ActivationType wide_type(DataType::FP32, {-2, kSequence, 2 * kWidth});
  BufferVec wide_output{*wide};
  BufferVec uniform_output{*uniform};
  BufferVec strong_output{*strong};
  ASSERT_TRUE(hooks.EnterCombinator(*executor_, "Root").ok());
  ASSERT_TRUE(hooks
                  .ActivationHook(*executor_, "Unsupported", {&wide_type, 1},
                                  absl::MakeSpan(wide_output))
                  .ok());
  ASSERT_TRUE(hooks
                  .ActivationHook(*executor_, "BelowThreshold", {&type, 1},
                                  absl::MakeSpan(uniform_output))
                  .ok());
  ASSERT_TRUE(hooks
                  .ActivationHook(*executor_, "Kept", {&type, 1},
                                  absl::MakeSpan(strong_output))
                  .ok());
  ASSERT_TRUE(hooks.ExitCombinator(*executor_).ok());
  ASSERT_TRUE(hooks
                  .ActivationHook(*executor_, "Root", {&type, 1},
                                  absl::MakeSpan(strong_output))
                  .ok());
  // Transferring the owning pointer cannot invalidate the adapter's owner.
  auto moved_owner = std::move(*inspector);
  EXPECT_EQ(&moved_owner->layer_hooks(), &hooks);
  std::ostringstream output;
  ASSERT_TRUE(moved_owner->Print(*executor_, detokenizer_, {0}, output).ok());
  EXPECT_NE(output.str().find("Root[0]/Kept[2]:"), std::string::npos);
  EXPECT_NE(output.str().find("  Root[0]:"), std::string::npos);
  EXPECT_EQ(output.str().find("Unsupported"), std::string::npos);
  EXPECT_EQ(output.str().find("BelowThreshold"), std::string::npos);
  EXPECT_NE(output.str().find("71.12%"), std::string::npos);
  EXPECT_EQ(output.str().find("100.00%"), std::string::npos);
}

TEST_F(ActivationInspectionTest, NestedCombinatorsRetainTheirAssignedPaths) {
  auto embedding = Upload(*executor_, EmbeddingValues());
  auto input = Upload(*executor_, ActivationValues());
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  ASSERT_TRUE(input.ok()) << input.status();
  auto inspector = NeighboringVocabInspector::Create(*executor_, *embedding,
                                                     kVocabulary, kWidth, 1);
  ASSERT_TRUE(inspector.ok()) << inspector.status();
  ComposedLayerBuilder inner;
  ASSERT_TRUE(inner.add(absl::make_unique<IdentityLayer>()).ok());
  ASSERT_TRUE(inner.add(absl::make_unique<IdentityLayer>()).ok());
  ComposedLayerBuilder outer;
  ASSERT_TRUE(outer.add(absl::make_unique<IdentityLayer>()).ok());
  ASSERT_TRUE(outer.add(inner.create("inner_pipeline")).ok());
  ASSERT_TRUE(outer.add(absl::make_unique<IdentityLayer>()).ok());
  auto graph = outer.create("outer_pipeline");
  ASSERT_TRUE(graph.ok()) << graph.status();
  auto forward =
      (*graph)->fwd(*executor_, {*input}, &(*inspector)->layer_hooks());
  ASSERT_TRUE(forward.ok()) << forward.status();
  std::ostringstream output;
  ASSERT_TRUE((*inspector)->Print(*executor_, detokenizer_, {0}, output).ok());
  const std::string text = output.str();
  const std::vector<std::string> labels = {
      "  outer_pipeline[0]/IdentityLayer[0]:",
      "  outer_pipeline[0]/inner_pipeline[1]/IdentityLayer[0]:",
      "  outer_pipeline[0]/inner_pipeline[1]/IdentityLayer[1]:",
      "  outer_pipeline[0]/inner_pipeline[1]:",
      "  outer_pipeline[0]/IdentityLayer[2]:",
      "  outer_pipeline[0]:"};
  size_t previous = 0;
  for (const auto& label : labels) {
    const size_t found = text.find(label, previous);
    ASSERT_NE(found, std::string::npos) << label << '\n' << text;
    previous = found + label.size();
  }
  EXPECT_EQ(text.find("IdentityLayer[3]"), std::string::npos);
}

TEST_F(ActivationInspectionTest, InheritedGradientHookIsANoOp) {
  auto embedding = Upload(*executor_, EmbeddingValues());
  auto input = Upload(*executor_, ActivationValues());
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  ASSERT_TRUE(input.ok()) << input.status();
  auto inspector = NeighboringVocabInspector::Create(*executor_, *embedding,
                                                     kVocabulary, kWidth, 1);
  ASSERT_TRUE(inspector.ok()) << inspector.status();
  IdentityLayer layer;
  auto forward = layer.fwd(*executor_, {*input}, &(*inspector)->layer_hooks());
  ASSERT_TRUE(forward.ok()) << forward.status();
  auto backward =
      (*inspector)->layer_hooks().GradientHook(*executor_, "unused", {}, {});
  ASSERT_TRUE(backward.ok()) << backward;
  auto gradients = layer.bwd(*executor_, {*input}, std::move(forward->state),
                             &(*inspector)->layer_hooks());
  ASSERT_TRUE(gradients.ok()) << gradients.status();
  EXPECT_EQ((*gradients)[0].data(), input->data());
  std::ostringstream output;
  ASSERT_TRUE((*inspector)->Print(*executor_, detokenizer_, {0}, output).ok());
  EXPECT_EQ(output.str().find("unused"), std::string::npos);
}

TEST_F(ActivationInspectionTest, RejectsInvalidCreationArguments) {
  auto embedding = Upload(*executor_, EmbeddingValues());
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  for (double threshold : {-0.01, 1.01, std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
    auto invalid = NeighboringVocabInspector::Create(
        *executor_, *embedding, kVocabulary, kWidth, 1, 0, threshold);
    EXPECT_EQ(invalid.status().code(), absl::StatusCode::kInvalidArgument);
  }
  for (const auto [vocab, width, positions, offset] :
       {std::array<int, 4>{2, kWidth, 1, 0},
        std::array<int, 4>{kVocabulary, 0, 1, 0},
        std::array<int, 4>{kVocabulary, kWidth, 0, 0},
        std::array<int, 4>{kVocabulary, kWidth, 1, -1},
        std::array<int, 4>{kVocabulary, kWidth, 2,
                           std::numeric_limits<int>::max()},
        std::array<int, 4>{kPaddedVocabulary + 1, kWidth, 1, 0}}) {
    EXPECT_EQ(NeighboringVocabInspector::Create(*executor_, *embedding, vocab,
                                                width, positions, offset)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
  auto partial = Buffer::Allocate(*executor_, 3);
  ASSERT_TRUE(partial.ok()) << partial.status();
  EXPECT_EQ(NeighboringVocabInspector::Create(*executor_, *partial, kVocabulary,
                                              kWidth, 1)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(ActivationInspectionTest, RejectsMalformedShapesAndMissingPrefixRows) {
  auto embedding = Upload(*executor_, EmbeddingValues());
  auto input = Upload(*executor_, ActivationValues());
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  ASSERT_TRUE(input.ok()) << input.status();
  auto inspector = NeighboringVocabInspector::Create(
      *executor_, *embedding, kVocabulary, kWidth, kSequence + 1);
  ASSERT_TRUE(inspector.ok()) << inspector.status();
  BufferVec outputs{*input};
  for (const ActivationType& type :
       {ActivationType(DataType::FP32, {-2, kSequence, kWidth}),
        ActivationType(DataType::FP32, {-2, -1, kWidth}),
        ActivationType(DataType::FP32, {-2, kSequence + 1, kWidth}),
        ActivationType(DataType::FP32,
                       {-2, std::numeric_limits<int64_t>::max(), 2})}) {
    EXPECT_EQ((*inspector)
                  ->layer_hooks()
                  .ActivationHook(*executor_, "invalid", {&type, 1},
                                  absl::MakeSpan(outputs))
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
  EXPECT_EQ(
      (*inspector)
          ->layer_hooks()
          .ActivationHook(*executor_, "invalid", {}, absl::MakeSpan(outputs))
          .code(),
      absl::StatusCode::kInvalidArgument);
}

TEST_F(ActivationInspectionTest, RejectsForeignExecutorAndUnbalancedScopes) {
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  auto embedding = Upload(*executor_, EmbeddingValues());
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  auto inspector = NeighboringVocabInspector::Create(*executor_, *embedding,
                                                     kVocabulary, kWidth, 1);
  ASSERT_TRUE(inspector.ok()) << inspector.status();
  EXPECT_EQ(NeighboringVocabInspector::Create(**other, *embedding, kVocabulary,
                                              kWidth, 1)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      (*inspector)->layer_hooks().EnterCombinator(**other, "wrong").code(),
      absl::StatusCode::kInvalidArgument);
  auto foreign = Upload(**other, ActivationValues());
  ASSERT_TRUE(foreign.ok()) << foreign.status();
  BufferVec outputs{*foreign};
  const ActivationType type(DataType::FP32, {-2, kSequence, kWidth});
  EXPECT_EQ((*inspector)
                ->layer_hooks()
                .ActivationHook(*executor_, "foreign", {&type, 1},
                                absl::MakeSpan(outputs))
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ((*inspector)->layer_hooks().ExitCombinator(*executor_).code(),
            absl::StatusCode::kFailedPrecondition);
  ASSERT_TRUE(
      (*inspector)->layer_hooks().EnterCombinator(*executor_, "open").ok());
  std::ostringstream output;
  EXPECT_EQ((*inspector)->Print(*executor_, detokenizer_, {0}, output).code(),
            absl::StatusCode::kFailedPrecondition);
  ASSERT_TRUE((*inspector)->layer_hooks().ExitCombinator(*executor_).ok());
  EXPECT_EQ((*inspector)->Print(**other, detokenizer_, {0}, output).code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pluto::llm
