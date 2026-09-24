#include "src/llm/layers/embedding.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "src/cuda/buffer.h"
#include "src/llm/layer.h"
#include "src/llm/layers/test_util.h"

namespace pluto::llm {
namespace {

template <class Element>
absl::Status WriteTestBuffer(cuda::Executor& executor,
                             const Buffer& destination,
                             const std::vector<Element>& values) {
  const size_t bytes = values.size() * sizeof(Element);
  if (destination.size_bytes() != bytes)
    return absl::InvalidArgumentError("test buffer has the wrong size");
  auto pinned = cuda::PageLockedHostArray<Element>::CopyFrom(executor, values);
  if (!pinned.ok())
    return pinned.status();
  auto status = cuda::CudaStatus(
      cudaMemcpyAsync(destination.data(), pinned->data(), bytes,
                      cudaMemcpyHostToDevice, executor.stream()),
      "copy determinism test input");
  if (!status.ok())
    return status;
  return executor.Synchronize();
}

template <class Element>
absl::StatusOr<Buffer> MakeTestBuffer(cuda::Executor& executor,
                                      const std::vector<Element>& values) {
  auto buffer = Buffer::Allocate(executor, values.size() * sizeof(Element));
  if (!buffer.ok())
    return buffer.status();
  auto status = WriteTestBuffer(executor, *buffer, values);
  if (!status.ok())
    return status;
  return std::move(*buffer);
}

absl::StatusOr<std::vector<uint32_t>> ReadTestFloatBits(
    cuda::Executor& executor, const Buffer& buffer) {
  if (buffer.size_bytes() % sizeof(uint32_t) != 0)
    return absl::InvalidArgumentError("test output is not an FP32 buffer");
  auto pinned = cuda::PageLockedHostArray<uint32_t>::Allocate(
      executor, buffer.size_bytes() / sizeof(uint32_t));
  if (!pinned.ok())
    return pinned.status();
  auto status = cuda::CudaStatus(
      cudaMemcpyAsync(pinned->data(), buffer.data(), buffer.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "read determinism test output");
  if (!status.ok())
    return status;
  status = executor.Synchronize();
  if (!status.ok())
    return status;
  return std::vector<uint32_t>(pinned->begin(), pinned->end());
}

std::vector<uint32_t> TestFloatBits(const std::vector<float>& values) {
  std::vector<uint32_t> result;
  result.reserve(values.size());
  for (float value : values)
    result.push_back(std::bit_cast<uint32_t>(value));
  return result;
}

std::vector<float> InitialTestGradient(size_t elements) {
  std::vector<float> result(elements);
  for (size_t index = 0; index < elements; ++index) {
    // Signed zeros also make unintended writes to untouched rows observable.
    result[index] = index % 11 == 0 ? -0.0f : (index % 17 + 1) * 0.125f;
  }
  return result;
}

float CancellationTestGradient(int occurrence, int column, int pass) {
  constexpr std::array<float, 6> kValues = {0x1p24f, 1.0f,  -0x1p24f,
                                            0.25f,   -0.5f, 0x1p-12f};
  return kValues[(occurrence + column + pass) % kValues.size()];
}

void AccumulateTestRows(const std::vector<int>& destinations,
                        const std::vector<float>& output_gradient, int width,
                        std::vector<float>* expected) {
  // This scalar row order is the contract, including the initial gradient.
  // Summing a batch first and adding it to the initial value later differs
  // for the cancellation-heavy inputs used below.
  for (size_t row = 0; row < destinations.size(); ++row) {
    for (int column = 0; column < width; ++column) {
      (*expected)[static_cast<size_t>(destinations[row]) * width + column] +=
          output_gradient[row * width + column];
    }
  }
}

TEST_F(LayersTest, RejectsFp8UntilScalingIsSpecified) {
  auto layer = EmbeddingLookupLayer::Create(*executor_, kTestVocabularySize,
                                            kTestModelWidth, DataType::FP8);
  EXPECT_FALSE(layer.ok());
  EXPECT_EQ(layer.status().code(), absl::StatusCode::kUnimplemented);
}

TEST_F(LayersTest, RejectsInvalidEmbeddingDimensions) {
  auto embedding = EmbeddingLookupLayer::Create(*executor_, 0, kTestModelWidth,
                                                DataType::FP16);
  EXPECT_FALSE(embedding.ok());
  EXPECT_EQ(embedding.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(LayersTest, LanguageModelingHeadUsesEmbeddingWeightTranspose) {
  auto embedding = EmbeddingLookupLayer::Create(
      *executor_, kTestVocabularySize, kTestModelWidth, DataType::FP16);
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  EXPECT_EQ((*embedding)->vocab_size(), kTestVocabularySize);
  EXPECT_EQ((*embedding)->embedding_dim(), kTestModelWidth);
  auto head = LanguageModelingHeadLayer::Create(embedding->get());
  ASSERT_TRUE(head.ok()) << head.status();
  ASSERT_EQ((*embedding)->weights().size(), 1u);
  ASSERT_EQ((*head)->weights().size(), 1u);
  EXPECT_EQ((*head)->weights().front().data(), (*embedding)->weight().data());

  std::vector<float> table(kTestVocabularySize * kTestModelWidth, 0.0f);
  table[3 * kTestModelWidth + 5] = 2.0f;
  table[7 * kTestModelWidth + 5] = 3.0f;
  const auto pinned_table = CopyToPageLockedHostArray(*executor_, table);
  ASSERT_EQ(cudaMemcpyAsync((*embedding)->weights().front().data(),
                            pinned_table.data(), table.size() * sizeof(float),
                            cudaMemcpyHostToDevice, executor_->stream()),
            cudaSuccess);
  std::vector<int> tokens(kTestTokenCount, 3);
  const auto pinned_tokens = CopyToPageLockedHostArray(*executor_, tokens);
  auto token_buffer = Buffer::Allocate(*executor_, tokens.size() * sizeof(int));
  ASSERT_TRUE(token_buffer.ok()) << token_buffer.status();
  ASSERT_EQ(cudaMemcpyAsync(token_buffer->data(), pinned_tokens.data(),
                            token_buffer->size_bytes(), cudaMemcpyHostToDevice,
                            executor_->stream()),
            cudaSuccess);

  BufferVec embedding_inputs = {*token_buffer};
  auto hidden = (*embedding)->fwd(*executor_, embedding_inputs);

  ASSERT_TRUE(hidden.ok()) << hidden.status();

  BufferVec head_inputs = {hidden->outputs[0]};
  auto logits = (*head)->fwd(*executor_, head_inputs);

  ASSERT_TRUE(logits.ok()) << logits.status();

  std::vector<float> output_gradient(kTestTokenCount * kTestVocabularySize,
                                     0.0f);
  output_gradient[7] = 1.0f;
  const auto pinned_output_gradient =
      CopyToPageLockedHostArray(*executor_, output_gradient);
  auto gradient_buffer =
      Buffer::Allocate(*executor_, output_gradient.size() * sizeof(float));
  ASSERT_TRUE(gradient_buffer.ok()) << gradient_buffer.status();
  ASSERT_EQ(
      cudaMemcpyAsync(gradient_buffer->data(), pinned_output_gradient.data(),
                      gradient_buffer->size_bytes(), cudaMemcpyHostToDevice,
                      executor_->stream()),
      cudaSuccess);
  BufferVec head_gradients = {*gradient_buffer};
  auto hidden_gradient =
      (*head)->bwd(*executor_, head_gradients, std::move(logits->state));
  ASSERT_TRUE(hidden_gradient.ok()) << hidden_gradient.status();
  ASSERT_EQ(hidden_gradient->size(), 1u);

  auto host_logits =
      AllocatePageLockedHostArray<float>(*executor_, kTestVocabularySize);
  auto host_hidden_gradient =
      AllocatePageLockedHostArray<float>(*executor_, kTestModelWidth);
  auto unchanged_table =
      AllocatePageLockedHostArray<float>(*executor_, table.size());
  auto table_gradient =
      AllocatePageLockedHostArray<float>(*executor_, table.size());
  ASSERT_EQ(cudaMemcpyAsync(host_logits.data(), logits->outputs[0].data(),
                            host_logits.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, executor_->stream()),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(host_hidden_gradient.data(),
                            hidden_gradient->front().data(),
                            host_hidden_gradient.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, executor_->stream()),
            cudaSuccess);
  ASSERT_EQ(
      cudaMemcpyAsync(unchanged_table.data(), (*embedding)->weight().data(),
                      unchanged_table.size() * sizeof(float),
                      cudaMemcpyDeviceToHost, executor_->stream()),
      cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(table_gradient.data(),
                            (*embedding)->gradients().front().data(),
                            table_gradient.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, executor_->stream()),
            cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());

  EXPECT_FLOAT_EQ(host_logits[3], 4.0f);
  EXPECT_FLOAT_EQ(host_logits[7], 6.0f);
  EXPECT_FLOAT_EQ(host_hidden_gradient[5], 3.0f);
  // Backward accumulates an FP32 tied-table gradient; the optimizer, not the
  // layer, owns parameter updates.
  EXPECT_FLOAT_EQ(unchanged_table[7 * kTestModelWidth + 5], 3.0f);
  EXPECT_FLOAT_EQ(table_gradient[7 * kTestModelWidth + 5], 2.0f);
}

TEST_F(LayersTest, LanguageModelingHeadRejectsNullEmbedding) {
  auto head = LanguageModelingHeadLayer::Create(nullptr);
  EXPECT_FALSE(head.ok());
  EXPECT_EQ(head.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(LayersTest, Bf16HeadMasksPhysicalVocabularyPadding) {
  constexpr int kLogicalVocabularySize = 17;
  constexpr int kPaddedVocabularySize = 32;
  auto embedding = EmbeddingLookupLayer::Create(
      *executor_, kLogicalVocabularySize, kTestModelWidth, DataType::BF16);
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  EXPECT_EQ((*embedding)->padded_vocab_size(), kPaddedVocabularySize);
  auto head = LanguageModelingHeadLayer::Create(embedding->get());
  ASSERT_TRUE(head.ok()) << head.status();

  std::vector<float> table(kPaddedVocabularySize * kTestModelWidth, 0.0f);
  table[3 * kTestModelWidth + 5] = 2.0f;
  const auto pinned_table = CopyToPageLockedHostArray(*executor_, table);
  ASSERT_EQ(cudaMemcpyAsync((*embedding)->weights().front().data(),
                            pinned_table.data(), table.size() * sizeof(float),
                            cudaMemcpyHostToDevice, executor_->stream()),
            cudaSuccess);
  std::vector<int> tokens(kTestTokenCount, 3);
  const auto pinned_tokens = CopyToPageLockedHostArray(*executor_, tokens);
  auto token_buffer = Buffer::Allocate(*executor_, tokens.size() * sizeof(int));
  ASSERT_TRUE(token_buffer.ok()) << token_buffer.status();
  ASSERT_EQ(cudaMemcpyAsync(token_buffer->data(), pinned_tokens.data(),
                            token_buffer->size_bytes(), cudaMemcpyHostToDevice,
                            executor_->stream()),
            cudaSuccess);

  BufferVec embedding_inputs = {*token_buffer};
  auto hidden = (*embedding)->fwd(*executor_, embedding_inputs);

  ASSERT_TRUE(hidden.ok()) << hidden.status();
  EXPECT_EQ(hidden->outputs[0].size_bytes(),
            kTestTokenCount * kTestModelWidth * sizeof(uint16_t));

  BufferVec head_inputs = {hidden->outputs[0]};
  auto logits = (*head)->fwd(*executor_, head_inputs);

  ASSERT_TRUE(logits.ok()) << logits.status();
  EXPECT_EQ(logits->outputs[0].size_bytes(),
            kTestTokenCount * kPaddedVocabularySize * sizeof(float));

  auto host_logits =
      AllocatePageLockedHostArray<float>(*executor_, kPaddedVocabularySize);
  ASSERT_EQ(cudaMemcpyAsync(host_logits.data(), logits->outputs[0].data(),
                            host_logits.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, executor_->stream()),
            cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());
  EXPECT_FLOAT_EQ(host_logits[3], 4.0f);
  for (int token = kLogicalVocabularySize; token < kPaddedVocabularySize;
       ++token) {
    EXPECT_LT(host_logits[token], -1e30f);
  }
}

TEST_F(LayersTest, PositionEmbeddingRepeatsAtRuntimeContextLength) {
  auto positions = PositionEmbeddingLayer::Create(
      *executor_, kTestContextLength, kTestModelWidth, DataType::FP16);
  ASSERT_TRUE(positions.ok()) << positions.status();

  std::vector<float> weight(kTestContextLength * kTestModelWidth, 0.0f);
  for (int position = 0; position < kTestContextLength; ++position)
    weight[position * kTestModelWidth] = static_cast<float>(position + 1);
  const auto pinned_weight = CopyToPageLockedHostArray(*executor_, weight);
  ASSERT_EQ(cudaMemcpyAsync((*positions)->weights().front().data(),
                            pinned_weight.data(), weight.size() * sizeof(float),
                            cudaMemcpyHostToDevice, executor_->stream()),
            cudaSuccess);
  std::vector<float> input(kTestTokenCount * kTestModelWidth, 0.0f);
  const auto pinned_input = CopyToPageLockedHostArray(*executor_, input);
  auto input_buffer =
      Buffer::Allocate(*executor_, input.size() * sizeof(float));
  ASSERT_TRUE(input_buffer.ok()) << input_buffer.status();
  ASSERT_EQ(cudaMemcpyAsync(input_buffer->data(), pinned_input.data(),
                            input_buffer->size_bytes(), cudaMemcpyHostToDevice,
                            executor_->stream()),
            cudaSuccess);

  BufferVec inputs = {*input_buffer};
  auto output = (*positions)->fwd(*executor_, inputs);

  ASSERT_TRUE(output.ok()) << output.status();
  auto host_output =
      AllocatePageLockedHostArray<float>(*executor_, input.size());
  ASSERT_EQ(cudaMemcpyAsync(host_output.data(), output->outputs[0].data(),
                            output->outputs[0].size_bytes(),
                            cudaMemcpyDeviceToHost, executor_->stream()),
            cudaSuccess);
  ASSERT_TRUE(executor_->Synchronize().ok());

  for (int row = 0; row < kTestTokenCount; ++row) {
    EXPECT_FLOAT_EQ(host_output[row * kTestModelWidth],
                    static_cast<float>(row % kTestContextLength + 1));
  }
}

TEST_F(LayersTest, LookupBackwardIsBitwiseRepeatableInOriginalRowOrder) {
  constexpr int kVocabulary = 17;
  constexpr std::array<int, 7> kTokenPattern = {16, 0, 7, 16, 7, 0, 16};
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    // Include both extremes of segment ownership: distinct tokens, and an
    // entire production-sized batch reduced into a single embedding row.
    for (const auto& [width, rows, pattern] :
         {std::tuple{1, 19, 0}, std::tuple{3, 19, 0}, std::tuple{8, 19, 0},
          std::tuple{15, 19, 0}, std::tuple{24, 19, 0}, std::tuple{16, 1, 0},
          std::tuple{16, 19, 0}, std::tuple{32, 255, 0}, std::tuple{32, 256, 0},
          std::tuple{32, 257, 0}, std::tuple{32, 511, 0},
          std::tuple{32, 512, 0}, std::tuple{32, 513, 0},
          std::tuple{32, 1025, 0}, std::tuple{32, 17, 1},
          std::tuple{32, 10240, 0}, std::tuple{32, 10240, 2}}) {
      SCOPED_TRACE(testing::Message()
                   << "type=" << static_cast<int>(type) << " width=" << width
                   << " rows=" << rows);
      auto embedding =
          EmbeddingLookupLayer::Create(*executor_, kVocabulary, width, type);
      ASSERT_TRUE(embedding.ok()) << embedding.status();
      const auto initial = InitialTestGradient(
          static_cast<size_t>((*embedding)->padded_vocab_size()) * width);
      std::array<std::vector<int>, 2> tokens;
      std::array<std::vector<float>, 2> output_gradients;
      for (int pass = 0; pass < 2; ++pass) {
        tokens[pass].resize(rows);
        output_gradients[pass].resize(static_cast<size_t>(rows) * width);
        std::array<int, kVocabulary> occurrences{};
        for (int row = 0; row < rows; ++row) {
          const int token =
              pattern == 2 ? 7
              : pattern == 1
                  ? (row + pass * 3) % kVocabulary
                  : kTokenPattern[(row + pass * 3) % kTokenPattern.size()];
          tokens[pass][row] = token;
          const int occurrence = occurrences[token]++;
          for (int column = 0; column < width; ++column) {
            output_gradients[pass][static_cast<size_t>(row) * width + column] =
                CancellationTestGradient(occurrence, column, pass);
          }
        }
      }

      std::array<std::vector<uint32_t>, 2> first_results;
      for (int repeat = 0; repeat < 4; ++repeat) {
        SCOPED_TRACE(testing::Message() << "repeat=" << repeat);
        ASSERT_TRUE(
            WriteTestBuffer(*executor_, (*embedding)->gradients()[0], initial)
                .ok());
        auto expected = initial;
        for (int pass = 0; pass < 2; ++pass) {
          SCOPED_TRACE(testing::Message() << "pass=" << pass);
          auto token_buffer = MakeTestBuffer(*executor_, tokens[pass]);
          auto gradient_buffer =
              MakeTestBuffer(*executor_, output_gradients[pass]);
          ASSERT_TRUE(token_buffer.ok()) << token_buffer.status();
          ASSERT_TRUE(gradient_buffer.ok()) << gradient_buffer.status();

          BufferVec inputs = {*token_buffer};
          auto output = (*embedding)->fwd(*executor_, inputs);

          ASSERT_TRUE(output.ok()) << output.status();
          BufferVec gradients = {*gradient_buffer};
          auto input_gradients =
              (*embedding)
                  ->bwd(*executor_, gradients, std::move(output->state));
          ASSERT_TRUE(input_gradients.ok()) << input_gradients.status();
          EXPECT_TRUE(input_gradients->empty());
          AccumulateTestRows(tokens[pass], output_gradients[pass], width,
                             &expected);
          auto actual =
              ReadTestFloatBits(*executor_, (*embedding)->gradients()[0]);
          ASSERT_TRUE(actual.ok()) << actual.status();
          // Compare the entire physical table: unobserved logical rows and
          // padding must preserve their original bytes as well.
          EXPECT_EQ(*actual, TestFloatBits(expected));
          if (repeat == 0)
            first_results[pass] = *actual;
          EXPECT_EQ(*actual, first_results[pass]);
        }
      }
    }
  }
}

TEST_F(LayersTest, LanguageModelingHeadIsBitwiseRepeatableWithRetainedStates) {
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (const auto& [vocab, width, rows] :
         {std::tuple{17, 16, 16}, std::tuple{65, 48, 48},
          std::tuple{257, 80, 80}, std::tuple{50257, 16, 16}}) {
      SCOPED_TRACE(testing::Message()
                   << "type=" << static_cast<int>(type) << " vocab=" << vocab
                   << " width=" << width << " rows=" << rows);
      auto embedding =
          EmbeddingLookupLayer::Create(*executor_, vocab, width, type);
      ASSERT_TRUE(embedding.ok()) << embedding.status();
      auto head = LanguageModelingHeadLayer::Create(embedding->get());
      ASSERT_TRUE(head.ok()) << head.status();
      const int padded = (*embedding)->padded_vocab_size();
      std::vector<float> table(static_cast<size_t>(padded) * width);
      for (size_t index = 0; index < table.size(); ++index)
        table[index] = 0.12f * std::sin(static_cast<float>(index) * 0.091f);
      ASSERT_TRUE(
          WriteTestBuffer(*executor_, (*embedding)->weight(), table).ok());

      std::array<BufferVec, 2> inputs;
      std::array<BufferVec, 2> gradients;
      std::array<FwdResult, 2> retained;
      std::array<std::vector<uint32_t>, 2> logits_bits;
      for (int pass = 0; pass < 2; ++pass) {
        std::vector<int> tokens(rows);
        for (int row = 0; row < rows; ++row)
          tokens[row] = (row * 5 + row / 3 + pass * 7) % vocab;
        auto token_buffer = MakeTestBuffer(*executor_, tokens);
        ASSERT_TRUE(token_buffer.ok()) << token_buffer.status();
        BufferVec token_inputs = {*token_buffer};
        auto hidden = (*embedding)->fwd(*executor_, token_inputs);
        ASSERT_TRUE(hidden.ok()) << hidden.status();
        inputs[pass] = std::move(hidden->outputs);
        std::vector<float> output_gradient(static_cast<size_t>(rows) * padded);
        for (size_t index = 0; index < output_gradient.size(); ++index)
          output_gradient[index] =
              0.08f * std::cos(static_cast<float>(index) * 0.07f + pass);
        auto gradient_buffer = MakeTestBuffer(*executor_, output_gradient);
        ASSERT_TRUE(gradient_buffer.ok()) << gradient_buffer.status();
        gradients[pass] = {*gradient_buffer};
        auto logits = (*head)->fwd(*executor_, inputs[pass]);
        ASSERT_TRUE(logits.ok()) << logits.status();
        auto bits = ReadTestFloatBits(*executor_, logits->outputs[0]);
        ASSERT_TRUE(bits.ok()) << bits.status();
        logits_bits[pass] = std::move(*bits);
        retained[pass] = std::move(*logits);
      }

      for (bool nonzero_initial : {false, true}) {
        SCOPED_TRACE(testing::Message()
                     << "nonzero initial=" << nonzero_initial);
        const auto initial = nonzero_initial
                                 ? InitialTestGradient(table.size())
                                 : std::vector<float>(table.size(), 0.0f);
        std::array<std::vector<uint32_t>, 2> first_input_gradients;
        std::array<std::vector<uint32_t>, 2> first_table_gradients;
        for (int repeat = 0; repeat < 4; ++repeat) {
          SCOPED_TRACE(testing::Message() << "repeat=" << repeat);
          ASSERT_TRUE(
              WriteTestBuffer(*executor_, (*embedding)->gradients()[0], initial)
                  .ok());
          for (int pass = 0; pass < 2; ++pass) {
            SCOPED_TRACE(testing::Message() << "pass=" << pass);
            // After the baseline run, make a different input the most recent
            // forward. Backward must still use its original retained input.
            const int forward_pass = repeat == 0 ? pass : 1 - pass;
            auto logits = (*head)->fwd(*executor_, inputs[forward_pass]);
            ASSERT_TRUE(logits.ok()) << logits.status();
            auto actual_logits =
                ReadTestFloatBits(*executor_, logits->outputs[0]);
            ASSERT_TRUE(actual_logits.ok()) << actual_logits.status();
            EXPECT_EQ(*actual_logits, logits_bits[forward_pass]);
            auto input_gradient =
                (*head)->bwd(*executor_, gradients[pass], retained[pass].state);
            ASSERT_TRUE(input_gradient.ok()) << input_gradient.status();
            ASSERT_EQ(input_gradient->size(), 1u);
            auto input_bits =
                ReadTestFloatBits(*executor_, input_gradient->front());
            auto table_bits =
                ReadTestFloatBits(*executor_, (*embedding)->gradients()[0]);
            ASSERT_TRUE(input_bits.ok()) << input_bits.status();
            ASSERT_TRUE(table_bits.ok()) << table_bits.status();
            if (repeat == 0) {
              first_input_gradients[pass] = *input_bits;
              first_table_gradients[pass] = *table_bits;
            }
            EXPECT_EQ(*input_bits, first_input_gradients[pass]);
            EXPECT_EQ(*table_bits, first_table_gradients[pass]);
          }
        }
      }
      for (int pass = 0; pass < 2; ++pass) {
        auto actual = ReadTestFloatBits(*executor_, retained[pass].outputs[0]);
        ASSERT_TRUE(actual.ok()) << actual.status();
        EXPECT_EQ(*actual, logits_bits[pass]);
      }
      auto unchanged_table =
          ReadTestFloatBits(*executor_, (*embedding)->weight());
      ASSERT_TRUE(unchanged_table.ok()) << unchanged_table.status();
      EXPECT_EQ(*unchanged_table, TestFloatBits(table));
    }
  }
}

TEST_F(LayersTest, LanguageModelingHeadCanonicalOrderPreservesExactRenaming) {
  // Prime vocabularies make the affine permutation a bijection, not its own
  // inverse. It crosses 64-wide MMA tiles and moves tokens into the tail tile.
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (bool pad_vocabulary : {false, true}) {
      for (const auto& [vocab, width, rows] :
           {std::tuple{137, 10, 27}, std::tuple{137, 73, 67}}) {
        SCOPED_TRACE(testing::Message()
                     << "type=" << static_cast<int>(type)
                     << " padded=" << pad_vocabulary << " width=" << width);
        auto original_embedding = EmbeddingLookupLayer::Create(
            *executor_, vocab, width, type, 1, pad_vocabulary);
        auto renamed_embedding = EmbeddingLookupLayer::Create(
            *executor_, vocab, width, type, 1, pad_vocabulary);
        ASSERT_TRUE(original_embedding.ok()) << original_embedding.status();
        ASSERT_TRUE(renamed_embedding.ok()) << renamed_embedding.status();
        const int stored = (*original_embedding)->stored_vocab_size();
        const int padded = (*original_embedding)->padded_vocab_size();
        std::vector<int32_t> permutation(vocab), identity(vocab);
        std::iota(identity.begin(), identity.end(), 0);
        for (int rank = 0; rank < vocab; ++rank)
          permutation[rank] = (rank * 37 + 19) % vocab;
        const auto saved_permutation = permutation;
        auto original_head =
            LanguageModelingHeadLayer::Create(original_embedding->get());
        auto identity_head = LanguageModelingHeadLayer::Create(
            original_embedding->get(), identity);
        auto renamed_head = LanguageModelingHeadLayer::Create(
            renamed_embedding->get(), permutation);
        ASSERT_TRUE(original_head.ok()) << original_head.status();
        ASSERT_TRUE(identity_head.ok()) << identity_head.status();
        ASSERT_TRUE(renamed_head.ok()) << renamed_head.status();
        // The host array is only an input to creation, not borrowed storage.
        std::fill(permutation.begin(), permutation.end(), -1);
        permutation.clear();
        permutation.shrink_to_fit();
        identity.clear();
        identity.shrink_to_fit();

        std::vector<float> table(static_cast<size_t>(stored) * width);
        std::vector<float> renamed_table(table.size());
        const auto initial_gradient = InitialTestGradient(table.size());
        std::vector<float> renamed_initial_gradient(table.size());
        for (int rank = 0; rank < stored; ++rank) {
          const int physical = rank < vocab ? saved_permutation[rank] : rank;
          for (int column = 0; column < width; ++column) {
            const size_t index = static_cast<size_t>(rank) * width + column;
            const size_t renamed_index =
                static_cast<size_t>(physical) * width + column;
            table[index] = 0.12f * std::sin(static_cast<float>(index) * 0.091f);
            renamed_table[renamed_index] = table[index];
            renamed_initial_gradient[renamed_index] = initial_gradient[index];
          }
        }
        ASSERT_TRUE(
            WriteTestBuffer(*executor_, (*original_embedding)->weight(), table)
                .ok());
        ASSERT_TRUE(WriteTestBuffer(*executor_, (*renamed_embedding)->weight(),
                                    renamed_table)
                        .ok());
        std::vector<int> tokens(rows);
        for (int row = 0; row < rows; ++row)
          tokens[row] = row * 7 % vocab;
        auto token_buffer = MakeTestBuffer(*executor_, tokens);
        ASSERT_TRUE(token_buffer.ok()) << token_buffer.status();
        auto hidden = (*original_embedding)->fwd(*executor_, {*token_buffer});
        ASSERT_TRUE(hidden.ok()) << hidden.status();
        std::vector<float> output_gradient(static_cast<size_t>(rows) * padded);
        std::vector<float> renamed_output_gradient(output_gradient.size());
        for (int row = 0; row < rows; ++row)
          for (int rank = 0; rank < padded; ++rank) {
            const int physical = rank < vocab ? saved_permutation[rank] : rank;
            const size_t index = static_cast<size_t>(row) * padded + rank;
            output_gradient[index] =
                0.08f * std::cos(static_cast<float>(index) * 0.07f);
            renamed_output_gradient[static_cast<size_t>(row) * padded +
                                    physical] = output_gradient[index];
          }
        auto gradient = MakeTestBuffer(*executor_, output_gradient);
        auto renamed_gradient =
            MakeTestBuffer(*executor_, renamed_output_gradient);
        ASSERT_TRUE(gradient.ok()) << gradient.status();
        ASSERT_TRUE(renamed_gradient.ok()) << renamed_gradient.status();

        std::array<std::vector<uint32_t>, 3> logits, d_hidden, d_table;
        const std::array<LanguageModelingHeadLayer*, 3> heads = {
            original_head->get(), identity_head->get(), renamed_head->get()};
        for (int pass = 0; pass < 3; ++pass) {
          ASSERT_TRUE(WriteTestBuffer(*executor_, heads[pass]->gradients()[0],
                                      pass == 2 ? renamed_initial_gradient
                                                : initial_gradient)
                          .ok());
          auto forward = heads[pass]->fwd(*executor_, hidden->outputs);
          ASSERT_TRUE(forward.ok()) << forward.status();
          auto backward = heads[pass]->bwd(
              *executor_, {pass == 2 ? *renamed_gradient : *gradient},
              std::move(forward->state));
          ASSERT_TRUE(backward.ok()) << backward.status();
          auto output_bits = ReadTestFloatBits(*executor_, forward->outputs[0]);
          auto hidden_bits = ReadTestFloatBits(*executor_, backward->front());
          auto table_bits =
              ReadTestFloatBits(*executor_, heads[pass]->gradients()[0]);
          ASSERT_TRUE(output_bits.ok()) << output_bits.status();
          ASSERT_TRUE(hidden_bits.ok()) << hidden_bits.status();
          ASSERT_TRUE(table_bits.ok()) << table_bits.status();
          logits[pass] = std::move(*output_bits);
          d_hidden[pass] = std::move(*hidden_bits);
          d_table[pass] = std::move(*table_bits);
        }
        // Align only physical vocabulary axes. Hidden gradients already share
        // coordinates and must match directly, not merely within a tolerance.
        auto aligned_logits = logits[2];
        auto aligned_table = d_table[2];
        for (int rank = 0; rank < vocab; ++rank) {
          for (int row = 0; row < rows; ++row)
            aligned_logits[static_cast<size_t>(row) * padded + rank] =
                logits[2][static_cast<size_t>(row) * padded +
                          saved_permutation[rank]];
          for (int column = 0; column < width; ++column)
            aligned_table[static_cast<size_t>(rank) * width + column] =
                d_table[2]
                       [static_cast<size_t>(saved_permutation[rank]) * width +
                        column];
        }
        EXPECT_EQ(logits[0], logits[1]);
        EXPECT_EQ(logits[0], aligned_logits);
        EXPECT_EQ(d_hidden[0], d_hidden[1]);
        EXPECT_EQ(d_hidden[0], d_hidden[2]);
        EXPECT_EQ(d_table[0], d_table[1]);
        EXPECT_EQ(d_table[0], aligned_table);
      }
    }
  }
}

TEST_F(LayersTest, LanguageModelingHeadRejectsMalformedTokenOrder) {
  auto embedding =
      EmbeddingLookupLayer::Create(*executor_, 3, 10, DataType::BF16);
  ASSERT_TRUE(embedding.ok()) << embedding.status();
  for (const auto& order : {std::vector<int32_t>{0, 1},
                            {0, 1, 1},
                            {0, 1, 3},
                            {0, 1, -1},
                            {0, 1, 2, 3}}) {
    auto head = LanguageModelingHeadLayer::Create(embedding->get(), order);
    EXPECT_EQ(head.status().code(), absl::StatusCode::kInvalidArgument);
  }
}

TEST_F(LayersTest, PositionBackwardIsBitwiseRepeatableWithPartialContexts) {
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    for (const auto& [context, width, rows] :
         {std::tuple{1, 16, 1}, std::tuple{7, 16, 3}, std::tuple{7, 32, 19},
          std::tuple{7, 32, 257}, std::tuple{1024, 32, 10240}}) {
      SCOPED_TRACE(testing::Message()
                   << "type=" << static_cast<int>(type) << " context="
                   << context << " width=" << width << " rows=" << rows);
      auto positions =
          PositionEmbeddingLayer::Create(*executor_, context, width, type);
      ASSERT_TRUE(positions.ok()) << positions.status();
      const auto initial =
          InitialTestGradient(static_cast<size_t>(context) * width);
      std::vector<int> destinations(rows);
      std::array<std::vector<float>, 2> output_gradients;
      for (int row = 0; row < rows; ++row)
        destinations[row] = row % context;
      for (int pass = 0; pass < 2; ++pass) {
        output_gradients[pass].resize(static_cast<size_t>(rows) * width);
        for (int row = 0; row < rows; ++row) {
          for (int column = 0; column < width; ++column) {
            output_gradients[pass][static_cast<size_t>(row) * width + column] =
                CancellationTestGradient(row / context, column, pass);
          }
        }
      }
      const size_t activation_bytes =
          type == DataType::BF16 ? 2 : sizeof(float);
      auto input = MakeTestBuffer(
          *executor_, std::vector<uint8_t>(static_cast<size_t>(rows) * width *
                                           activation_bytes));
      ASSERT_TRUE(input.ok()) << input.status();
      std::array<std::vector<uint32_t>, 2> first_results;
      for (int repeat = 0; repeat < 4; ++repeat) {
        SCOPED_TRACE(testing::Message() << "repeat=" << repeat);
        ASSERT_TRUE(
            WriteTestBuffer(*executor_, (*positions)->gradients()[0], initial)
                .ok());
        auto expected = initial;
        for (int pass = 0; pass < 2; ++pass) {
          SCOPED_TRACE(testing::Message() << "pass=" << pass);
          auto gradient_buffer =
              MakeTestBuffer(*executor_, output_gradients[pass]);
          ASSERT_TRUE(gradient_buffer.ok()) << gradient_buffer.status();

          BufferVec inputs = {*input};
          auto output = (*positions)->fwd(*executor_, inputs);

          ASSERT_TRUE(output.ok()) << output.status();
          BufferVec gradients = {*gradient_buffer};
          auto input_gradients =
              (*positions)
                  ->bwd(*executor_, gradients, std::move(output->state));
          ASSERT_TRUE(input_gradients.ok()) << input_gradients.status();
          ASSERT_EQ(input_gradients->size(), 1u);
          EXPECT_EQ(input_gradients->front().data(), gradient_buffer->data());
          auto returned =
              ReadTestFloatBits(*executor_, input_gradients->front());
          ASSERT_TRUE(returned.ok()) << returned.status();
          EXPECT_EQ(*returned, TestFloatBits(output_gradients[pass]));
          AccumulateTestRows(destinations, output_gradients[pass], width,
                             &expected);
          auto actual =
              ReadTestFloatBits(*executor_, (*positions)->gradients()[0]);
          ASSERT_TRUE(actual.ok()) << actual.status();
          EXPECT_EQ(*actual, TestFloatBits(expected));
          if (repeat == 0)
            first_results[pass] = *actual;
          EXPECT_EQ(*actual, first_results[pass]);
        }
      }
    }
  }
}

TEST_F(LayersTest, LookupBackwardPreservesRealTiedHeadGradientAcrossPasses) {
  constexpr int kVocabulary = 17;
  constexpr int kWidth = 16;
  constexpr int kRows = 32;
  for (DataType type : {DataType::FP16, DataType::BF16}) {
    SCOPED_TRACE(testing::Message() << "type=" << static_cast<int>(type));
    auto embedding =
        EmbeddingLookupLayer::Create(*executor_, kVocabulary, kWidth, type);
    ASSERT_TRUE(embedding.ok()) << embedding.status();
    auto head = LanguageModelingHeadLayer::Create(embedding->get());
    ASSERT_TRUE(head.ok()) << head.status();
    const int padded = (*embedding)->padded_vocab_size();
    std::vector<float> table(static_cast<size_t>(padded) * kWidth, 0.0f);
    table[3 * kWidth + 5] = 2.0f;
    table[7 * kWidth + 5] = 3.0f;
    ASSERT_TRUE(
        WriteTestBuffer(*executor_, (*embedding)->weight(), table).ok());
    std::vector<int> tokens(kRows);
    for (int row = 0; row < kRows; ++row)
      tokens[row] = row % 2 == 0 ? 3 : 7;
    auto token_buffer = MakeTestBuffer(*executor_, tokens);
    ASSERT_TRUE(token_buffer.ok()) << token_buffer.status();
    std::vector<float> logits_gradient(static_cast<size_t>(kRows) * padded,
                                       0.0f);
    logits_gradient[7] = 1.0f;
    logits_gradient[16 * padded + 3] = 0.5f;
    auto gradient_buffer = MakeTestBuffer(*executor_, logits_gradient);
    ASSERT_TRUE(gradient_buffer.ok()) << gradient_buffer.status();
    const std::vector<float> initial(table.size(), 0.0f);
    std::array<std::vector<uint32_t>, 2> first_results;
    for (int repeat = 0; repeat < 4; ++repeat) {
      SCOPED_TRACE(testing::Message() << "repeat=" << repeat);
      ASSERT_TRUE(
          WriteTestBuffer(*executor_, (*embedding)->gradients()[0], initial)
              .ok());
      auto expected = initial;
      for (int pass = 0; pass < 2; ++pass) {
        SCOPED_TRACE(testing::Message() << "pass=" << pass);

        BufferVec inputs = {*token_buffer};
        auto hidden = (*embedding)->fwd(*executor_, inputs);

        ASSERT_TRUE(hidden.ok()) << hidden.status();

        BufferVec head_inputs = {hidden->outputs[0]};
        auto logits = (*head)->fwd(*executor_, head_inputs);

        ASSERT_TRUE(logits.ok()) << logits.status();
        BufferVec gradients = {*gradient_buffer};
        auto hidden_gradient =
            (*head)->bwd(*executor_, gradients, std::move(logits->state));
        ASSERT_TRUE(hidden_gradient.ok()) << hidden_gradient.status();
        ASSERT_EQ(hidden_gradient->size(), 1u);
        // All products are exactly representable, avoiding an MMA-order
        // tolerance in this test of the shared gradient's ownership.
        expected[7 * kWidth + 5] += 2.0f;
        expected[3 * kWidth + 5] += 1.0f;
        auto after_head =
            ReadTestFloatBits(*executor_, (*embedding)->gradients()[0]);
        ASSERT_TRUE(after_head.ok()) << after_head.status();
        EXPECT_EQ(*after_head, TestFloatBits(expected));
        auto result =
            (*embedding)
                ->bwd(*executor_, *hidden_gradient, std::move(hidden->state));
        ASSERT_TRUE(result.ok()) << result.status();
        EXPECT_TRUE(result->empty());
        expected[3 * kWidth + 5] += 3.0f;
        expected[3 * kWidth + 5] += 1.0f;
        auto actual =
            ReadTestFloatBits(*executor_, (*embedding)->gradients()[0]);
        ASSERT_TRUE(actual.ok()) << actual.status();
        EXPECT_EQ(*actual, TestFloatBits(expected));
        if (repeat == 0)
          first_results[pass] = *actual;
        EXPECT_EQ(*actual, first_results[pass]);
      }
    }
  }
}

}  // namespace
}  // namespace pluto::llm
