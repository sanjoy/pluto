#include "src/llm/recipes/mlp_automaton/model.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/llm/layer.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/embedding.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/gelu.h"
#include "src/llm/layers/norm.h"
#include "src/llm/layers/reference_test_util.h"
#include "src/util/status_macros.h"

namespace pluto::llm::mlp_automaton {
namespace {

constexpr Dimensions kDimensions{17, 16, 32};
constexpr int kPaddedVocabulary = 32;
constexpr std::array<int, 9> kCheckpointIndices{0,  8,  9,  10, 11,
                                                12, 13, 98, 99};
constexpr std::array<size_t, 9> kWeightElements{32 * 16, 16, 16, 16 * 32, 32,
                                                32 * 16, 16, 16, 16};

template <class BufferType>
std::vector<BufferType> DistinctWeights(absl::Span<BufferType> weights) {
  std::unordered_set<const void*> seen;
  std::vector<BufferType> result;
  for (const auto& weight : weights) {
    if (seen.insert(weight.data()).second) result.push_back(weight);
  }
  return result;
}

// This independently assembles the intended scalar CPU formula, including the
// input residual and both learned normalizers. It does not invoke
// CreateReadout.
absl::StatusOr<std::unique_ptr<LayerReference>> CreateReference() {
  ComposedLayerReferenceBuilder mlp;
  RETURN_IF_ERROR(mlp.add(LayerNormLayerReference::Create(
      kDimensions.model_width, 1e-5f, DataType::BF16)));
  RETURN_IF_ERROR(mlp.add(FullyConnectedLayerReference::Create(
      kDimensions.model_width, kDimensions.feed_forward_width,
      DataType::BF16)));
  RETURN_IF_ERROR(mlp.add(GeluLayerReference::Create(DataType::BF16)));
  RETURN_IF_ERROR(mlp.add(FullyConnectedLayerReference::Create(
      kDimensions.feed_forward_width, kDimensions.model_width,
      DataType::BF16)));
  ASSIGN_OR_RETURN(auto branch, mlp.create());

  ComposedLayerReferenceBuilder readout;
  RETURN_IF_ERROR(readout.add(EmbeddingLookupLayerReference::Create(
      kDimensions.vocab_size, kDimensions.model_width, DataType::BF16)));
  auto* embedding = static_cast<EmbeddingLookupLayerReference*>(readout.back());
  RETURN_IF_ERROR(
      readout.add(std::make_unique<ResidualLayerReference>(std::move(branch))));
  RETURN_IF_ERROR(readout.add(LayerNormLayerReference::Create(
      kDimensions.model_width, 1e-5f, DataType::BF16)));
  RETURN_IF_ERROR(
      readout.add(LanguageModelingHeadLayerReference::Create(embedding)));
  ASSIGN_OR_RETURN(auto model, readout.create());
  return std::unique_ptr<LayerReference>(std::move(model));
}

class MlpAutomatonModelTest : public LayerReferenceTest {
 protected:
  void SetUp() override {
    LayerReferenceTest::SetUp();
    if (HasFatalFailure()) return;
    std::string pattern =
        (std::filesystem::path(testing::TempDir()) / "mlp-automaton-XXXXXX")
            .string();
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    const char* created = mkdtemp(name.data());
    ASSERT_NE(created, nullptr);
    directory_ = created;

    auto model = CreateReadout(*executor_, kDimensions);
    ASSERT_TRUE(model.ok()) << model.status();
    model_ = std::move(*model);
    auto reference = CreateReference();
    ASSERT_TRUE(reference.ok()) << reference.status();
    reference_ = std::move(*reference);
    auto weights = DistinctWeights(reference_->weights());
    ASSERT_EQ(weights.size(), kCheckpointIndices.size());
    for (size_t index = 0; index < weights.size(); ++index) {
      ASSERT_EQ(weights[index].size_bytes(), kWeightElements[index] * 4);
      auto* values = static_cast<float*>(weights[index].data());
      for (size_t element = 0; element < kWeightElements[index]; ++element) {
        const float phase = static_cast<float>(element) * 0.271f +
                            static_cast<float>(index) * 0.719f;
        if (index == 1 || index == 7) {
          values[element] =
              0.65f + 0.025f * (element % 7) + (index == 7 ? 0.1f : 0.0f);
        } else {
          values[element] =
              (index == 0 || index == 3 || index == 5 ? 0.14f : 0.055f) *
                  std::sin(phase) +
              0.023f * std::cos(phase * 0.37f);
        }
      }
    }
  }

  void TearDown() override {
    reference_.reset();
    model_.reset();
    LayerReferenceTest::TearDown();
    if (!directory_.empty()) {
      std::error_code error;
      std::filesystem::remove_all(directory_, error);
      EXPECT_FALSE(error) << error.message();
    }
  }

  void WriteCheckpoint(const std::filesystem::path& directory,
                       bool omit_last = false) {
    ASSERT_TRUE(std::filesystem::create_directory(directory));
    auto weights = DistinctWeights(reference_->weights());
    for (size_t index = 0; index < weights.size(); ++index) {
      if (omit_last && index + 1 == weights.size()) continue;
      std::ofstream output(
          directory /
              ("weight_" + std::to_string(kCheckpointIndices[index]) + ".bin"),
          std::ios::binary);
      ASSERT_TRUE(output.good());
      output.write(static_cast<const char*>(weights[index].data()),
                   static_cast<std::streamsize>(weights[index].size_bytes()));
      output.close();
      ASSERT_TRUE(output.good());
    }
  }

  std::vector<std::vector<float>> Snapshot() {
    std::vector<std::vector<float>> result;
    for (const auto& weight : DistinctWeights(model_->weights())) {
      auto values = ReadDeviceFloats(*executor_, weight);
      EXPECT_TRUE(values.ok()) << values.status();
      if (!values.ok()) return {};
      result.emplace_back(values->begin(), values->end());
    }
    return result;
  }

  void ExpectUnchanged(const std::vector<std::vector<float>>& before) {
    const auto after = Snapshot();
    ASSERT_EQ(after.size(), before.size());
    for (size_t index = 0; index < before.size(); ++index) {
      ASSERT_EQ(after[index].size(), before[index].size());
      EXPECT_EQ(std::memcmp(after[index].data(), before[index].data(),
                            before[index].size() * sizeof(float)),
                0)
          << "parameter " << index << " changed after rejected load";
    }
  }

  std::filesystem::path directory_;
  std::unique_ptr<Layer> model_;
  std::unique_ptr<LayerReference> reference_;
};

TEST_F(MlpAutomatonModelTest, TiedWeightLayoutAndSparseFileIndices) {
  const auto raw = model_->weights();
  ASSERT_EQ(raw.size(), 10u);
  EXPECT_EQ(raw.front().data(), raw.back().data());
  EXPECT_EQ(model_->output_type(), DataType::BF16);
  const auto unique = DistinctWeights(raw);
  ASSERT_EQ(unique.size(), kWeightElements.size());
  for (size_t index = 0; index < unique.size(); ++index) {
    EXPECT_EQ(unique[index].size_bytes(),
              kWeightElements[index] * sizeof(float));
    EXPECT_EQ(&unique[index].executor(), executor_.get());
  }

  const auto checkpoint = directory_ / "checkpoint";
  WriteCheckpoint(checkpoint);
  ASSERT_FALSE(HasFatalFailure());
  // Unneeded full-model weights must not be loaded into the nine selected
  // buffers, and unrelated directory entries do not invalidate the subset.
  std::ofstream unused(checkpoint / "weight_1.bin", std::ios::binary);
  unused << "deliberately not a position-embedding tensor";
  unused.close();
  std::ofstream note(checkpoint / "README.txt");
  note << "ignored";
  note.close();
  const auto loaded = LoadB0Weights(*executor_, *model_, checkpoint);
  ASSERT_TRUE(loaded.ok()) << loaded;
  const auto actual = Snapshot();
  const auto expected = DistinctWeights(reference_->weights());
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t index = 0; index < expected.size(); ++index) {
    EXPECT_EQ(std::memcmp(actual[index].data(), expected[index].data(),
                          expected[index].size_bytes()),
              0)
        << "wrong full-checkpoint index for parameter " << index;
  }
}

TEST_F(MlpAutomatonModelTest, NativeReadoutMatchesIndependentCpuFormula) {
  const auto checkpoint = directory_ / "checkpoint";
  WriteCheckpoint(checkpoint);
  ASSERT_FALSE(HasFatalFailure());
  const auto loaded = LoadB0Weights(*executor_, *model_, checkpoint);
  ASSERT_TRUE(loaded.ok()) << loaded;
  std::vector<int> tokens(32);
  for (size_t row = 0; row < tokens.size(); ++row) {
    tokens[row] =
        static_cast<int>((row * 7 + row / 4) % kDimensions.vocab_size);
  }
  tokens[0] = 16;  // Include the last logical token in a padded vocabulary.
  tokens[19] = 16;
  auto input = MakeRawBufferPair<int>(*executor_, tokens);
  ASSERT_TRUE(input.ok()) << input.status();
  const BufferVec device_inputs{input->device};
  const HostBufferVec host_inputs{input->host};
  Tape device_tape;
  ReferenceTape host_tape;
  auto actual = model_->fwd(*executor_, device_inputs, &device_tape);
  auto expected = reference_->fwd(host_inputs, &host_tape);
  ASSERT_TRUE(actual.ok()) << actual.status();
  ASSERT_TRUE(expected.ok()) << expected.status();
  ASSERT_EQ(actual->size_bytes(), tokens.size() * kPaddedVocabulary * 4);
  // Scalar reference reductions and native MMA can cross different BF16
  // rounding boundaries. Compare the complete logits, not merely top-1 IDs.
  EXPECT_TRUE(FloatBuffersNear(*actual, *expected, 0.02f, 0.008f));
  auto logits = ReadDeviceFloats(*executor_, *actual);
  ASSERT_TRUE(logits.ok()) << logits.status();
  for (size_t row = 0; row < tokens.size(); ++row) {
    for (int token = 0; token < kPaddedVocabulary; ++token) {
      const float value = (*logits)[row * kPaddedVocabulary + token];
      if (token < kDimensions.vocab_size) {
        EXPECT_TRUE(std::isfinite(value));
      } else {
        EXPECT_EQ(value, -std::numeric_limits<float>::max());
      }
    }
  }
  // The isolated network has no positions or attention: equal input IDs at
  // different rows must have exactly equal outputs, including padded lanes.
  EXPECT_EQ(std::memcmp(logits->data(), logits->data() + 19 * kPaddedVocabulary,
                        kPaddedVocabulary * sizeof(float)),
            0);
}

TEST_F(MlpAutomatonModelTest, InvalidLateFileDoesNotModifyAnyDeviceWeight) {
  const auto before = Snapshot();
  ASSERT_EQ(before.size(), kCheckpointIndices.size());
  for (const std::string mode : {"missing", "short", "long", "nan", "inf"}) {
    SCOPED_TRACE(mode);
    const auto checkpoint = directory_ / mode;
    WriteCheckpoint(checkpoint, mode == "missing");
    ASSERT_FALSE(HasFatalFailure());
    if (mode != "missing") {
      std::vector<float> invalid(16, 0.25f);
      if (mode == "short") invalid.pop_back();
      if (mode == "long") invalid.push_back(0.5f);
      if (mode == "nan")
        invalid.back() = std::numeric_limits<float>::quiet_NaN();
      if (mode == "inf")
        invalid.back() = std::numeric_limits<float>::infinity();
      std::ofstream last(checkpoint / "weight_99.bin",
                         std::ios::binary | std::ios::trunc);
      ASSERT_TRUE(last.good());
      last.write(reinterpret_cast<const char*>(invalid.data()),
                 static_cast<std::streamsize>(invalid.size() * sizeof(float)));
      last.close();
      ASSERT_TRUE(last.good());
    }
    const auto status = LoadB0Weights(*executor_, *model_, checkpoint);
    EXPECT_FALSE(status.ok());
    ExpectUnchanged(before);
  }
}

TEST_F(MlpAutomatonModelTest, RejectsDifferentExecutorBeforeLoading) {
  const auto checkpoint = directory_ / "checkpoint";
  WriteCheckpoint(checkpoint);
  ASSERT_FALSE(HasFatalFailure());
  const auto before = Snapshot();
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  const auto status = LoadB0Weights(**other, *model_, checkpoint);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  ExpectUnchanged(before);
}

TEST_F(MlpAutomatonModelTest, RejectsIncompatibleWeightLayout) {
  const auto checkpoint = directory_ / "checkpoint";
  WriteCheckpoint(checkpoint);
  ASSERT_FALSE(HasFatalFailure());
  auto unrelated =
      FullyConnectedLayer::Create(*executor_, 16, 32, DataType::BF16);
  ASSERT_TRUE(unrelated.ok()) << unrelated.status();
  EXPECT_FALSE(LoadB0Weights(*executor_, **unrelated, checkpoint).ok());
}

TEST_F(MlpAutomatonModelTest, RejectsInvalidDimensions) {
  for (const Dimensions dimensions :
       {Dimensions{0, 16, 32}, Dimensions{17, 15, 32}, Dimensions{17, 16, 31},
        Dimensions{17, 0, 32}, Dimensions{17, 16, 0}}) {
    EXPECT_FALSE(CreateReadout(*executor_, dimensions).ok());
  }
}

TEST_F(MlpAutomatonModelTest, ScanCoversEveryTokenAndFinalPartialBatch) {
  const auto checkpoint = directory_ / "checkpoint";
  WriteCheckpoint(checkpoint);
  ASSERT_FALSE(HasFatalFailure());
  const auto loaded = LoadB0Weights(*executor_, *model_, checkpoint);
  ASSERT_TRUE(loaded.ok()) << loaded;

  std::vector<int> progress;
  auto scan =
      ScanVocabulary(*executor_, *model_, kDimensions.vocab_size, 16,
                     [&](int completed) { progress.push_back(completed); });
  ASSERT_TRUE(scan.ok()) << scan.status();
  ASSERT_EQ(scan->size(), 17u);
  EXPECT_EQ(progress, (std::vector<int>{16, 17}));

  // Evaluate all logical input IDs in one native call. The seventeenth row
  // must survive the scanner's partial tail batch, whereas padding must not.
  std::vector<int> ids(32, 0);
  for (int token = 0; token < kDimensions.vocab_size; ++token)
    ids[token] = token;
  auto input = MakeRawBufferPair<int>(*executor_, ids);
  ASSERT_TRUE(input.ok()) << input.status();
  Tape tape;
  auto output = model_->fwd(*executor_, {input->device}, &tape);
  ASSERT_TRUE(output.ok()) << output.status();
  auto logits = ReadDeviceFloats(*executor_, *output);
  ASSERT_TRUE(logits.ok()) << logits.status();
  for (int source = 0; source < kDimensions.vocab_size; ++source) {
    const float* row = logits->data() + source * kPaddedVocabulary;
    int winner = 0;
    for (int target = 1; target < kDimensions.vocab_size; ++target) {
      if (row[target] > row[winner]) winner = target;
    }
    double denominator = 0;
    for (int target = 0; target < kDimensions.vocab_size; ++target) {
      denominator += std::exp(static_cast<double>(row[target]) - row[winner]);
    }
    EXPECT_EQ((*scan)[source].token, winner) << "source " << source;
    EXPECT_NEAR((*scan)[source].probability, 1.0 / denominator, 2e-6)
        << "source " << source;
  }

  // The same token-only readout must not depend on how tokens are batched.
  // An oversized request also exercises the tiny-vocabulary allocation cap.
  for (int batch_size : {32, 1008}) {
    SCOPED_TRACE(batch_size);
    progress.clear();
    auto differently_batched =
        ScanVocabulary(*executor_, *model_, kDimensions.vocab_size, batch_size,
                       [&](int completed) { progress.push_back(completed); });
    ASSERT_TRUE(differently_batched.ok()) << differently_batched.status();
    ASSERT_EQ(differently_batched->size(), scan->size());
    EXPECT_EQ(progress, (std::vector<int>{17}));
    EXPECT_EQ(std::memcmp(scan->data(), differently_batched->data(),
                          scan->size_bytes()),
              0);
  }
}

TEST_F(MlpAutomatonModelTest, ScanRejectsInvalidArgumentsWithoutProgress) {
  struct Arguments {
    int vocabulary;
    int batch_size;
  };
  for (const auto arguments :
       {Arguments{0, 16}, Arguments{-1, 16},
        Arguments{std::numeric_limits<int>::max(), 16}, Arguments{17, 0},
        Arguments{17, -16}, Arguments{17, 1}, Arguments{17, 15},
        Arguments{17, 17}}) {
    SCOPED_TRACE(testing::Message() << "vocabulary=" << arguments.vocabulary
                                    << " batch=" << arguments.batch_size);
    bool progressed = false;
    auto scan =
        ScanVocabulary(*executor_, *model_, arguments.vocabulary,
                       arguments.batch_size, [&](int) { progressed = true; });
    ASSERT_FALSE(scan.ok());
    EXPECT_EQ(scan.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_FALSE(progressed);
  }
}

TEST_F(MlpAutomatonModelTest, ScanRejectsNonfiniteModelOutputs) {
  const auto checkpoint = directory_ / "checkpoint";
  WriteCheckpoint(checkpoint);
  ASSERT_FALSE(HasFatalFailure());
  const auto loaded = LoadB0Weights(*executor_, *model_, checkpoint);
  ASSERT_TRUE(loaded.ok()) << loaded;
  auto device_weights = DistinctWeights(model_->weights());
  auto host_weights = DistinctWeights(reference_->weights());

  // Loading rejects these values, so deliberately inject them after a valid
  // load to exercise scan-time invalid-logit handling independently.
  for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity()}) {
    std::vector<float> final_bias(kDimensions.model_width, 0.01f);
    final_bias[0] = invalid;
    ASSERT_TRUE(SetFloatBufferPair(*executor_, device_weights[8],
                                   &host_weights[8], final_bias)
                    .ok());
    bool progressed = false;
    auto scan = ScanVocabulary(*executor_, *model_, kDimensions.vocab_size, 16,
                               [&](int) { progressed = true; });
    ASSERT_FALSE(scan.ok());
    EXPECT_EQ(scan.status().code(), absl::StatusCode::kDataLoss);
    EXPECT_FALSE(progressed);
  }
}

TEST_F(MlpAutomatonModelTest, ScanRejectsExecutorMismatch) {
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok()) << other.status();
  bool progressed = false;
  auto scan = ScanVocabulary(**other, *model_, kDimensions.vocab_size, 16,
                             [&](int) { progressed = true; });
  ASSERT_FALSE(scan.ok());
  EXPECT_EQ(scan.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_FALSE(progressed);
}

}  // namespace
}  // namespace pluto::llm::mlp_automaton
