#include "src/llm/qwen/embedding_algebra_table.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "src/util/status_macros.h"

namespace pluto::llm::qwen {
namespace {

// CPU bit conversion keeps the reference independent of CUDA conversion code.
uint16_t Bf16Bits(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fff + ((bits >> 16) & 1);
  return bits >> 16;
}
float Bf16(float value) {
  return std::bit_cast<float>(uint32_t(Bf16Bits(value)) << 16);
}

class EmbeddingAlgebraTableTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
  }
  void TearDown() override { EXPECT_TRUE(executor_->Synchronize().ok()); }

  absl::StatusOr<cuda::Buffer> Upload(const std::vector<float>& values) {
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<uint16_t>::Allocate(
                                    *executor_, values.size()));
    for (size_t i = 0; i < values.size(); ++i)
      host[i] = Bf16Bits(values[i]);
    ASSIGN_OR_RETURN(auto device,
                     cuda::Buffer::Allocate(*executor_, host.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(device.data(), host.data(), host.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "upload test embeddings"));
    return device;
  }
  absl::StatusOr<std::unique_ptr<EmbeddingAlgebraTable>> Table(
      const std::vector<float>& values, int rows, int width, int searchable) {
    ASSIGN_OR_RETURN(auto device, Upload(values));
    return EmbeddingAlgebraTable::Create(*executor_, std::move(device), rows,
                                         width, searchable);
  }

  std::unique_ptr<cuda::Executor> executor_;
};

TEST_F(EmbeddingAlgebraTableTest, SignedArithmeticRetainsFloat32Precision) {
  auto table = Table({1, 2, 3, .001953125f, -4, 8, .5, 1, -2}, 3, 3, 3);
  ASSERT_TRUE(table.ok()) << table.status();
  EXPECT_EQ((*table)->vocab_size(), 3);
  EXPECT_EQ((*table)->dimensions(), 3);
  auto output = (*table)->Evaluate({{0, 1}, {1, 1}, {2, -1}});
  ASSERT_TRUE(output.ok()) << output.status();
  EXPECT_EQ((*output)[0], .501953125f);
  EXPECT_EQ((*output)[1], -3);
  EXPECT_EQ((*output)[2], 13);
  auto precise = (*table)->Evaluate({{0, 1}, {1, 1}});
  ASSERT_TRUE(precise.ok()) << precise.status();
  EXPECT_EQ((*precise)[0], 1.001953125f);
  EXPECT_NE((*precise)[0], Bf16((*precise)[0]));
}

TEST_F(EmbeddingAlgebraTableTest, CosineRankingTiesDistancesAndPadding) {
  auto table =
      Table({1, 0, 0, 2, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 1, 1, 0}, 6, 3, 5);
  ASSERT_TRUE(table.ok()) << table.status();
  auto result = (*table)->Nearest({1, 1, 0});
  ASSERT_TRUE(result.ok()) << result.status();
  ASSERT_EQ(result->size(), 3u);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ((*result)[i].token_id, i);
    EXPECT_NEAR((*result)[i].cosine_similarity, 1 / std::sqrt(2.0), 1e-6);
  }
  EXPECT_FLOAT_EQ((*result)[0].l2_distance, 1);
  EXPECT_NEAR((*result)[1].l2_distance, std::sqrt(2.0), 1e-6);
  EXPECT_FLOAT_EQ((*result)[2].l2_distance, 1);
  auto all = (*table)->Nearest({1, 1, 0}, 100);
  ASSERT_TRUE(all.ok());
  // Omit the zero row and tokenizer-inaccessible padding.
  ASSERT_EQ(all->size(), 4u);
  EXPECT_EQ(all->back().token_id, 3);
  EXPECT_NEAR(all->back().cosine_similarity, -1 / std::sqrt(2.0), 1e-6);
  auto self = (*table)->Nearest({1, 0, 0}, 1);
  ASSERT_TRUE(self.ok());
  EXPECT_EQ(self->front().token_id, 0);
  EXPECT_EQ(self->front().cosine_similarity, 1);
  EXPECT_EQ(self->front().l2_distance, 0);
  EXPECT_FALSE((*table)->Evaluate({{5, 1}}).ok());
}

TEST_F(EmbeddingAlgebraTableTest,
       TailTilesAndProductionWidthMatchCpuReference) {
  for (int width : {1, 3, 259, 5120}) {
    SCOPED_TRACE(width);
    constexpr int kRows = 11;
    std::vector<float> values(kRows * width), query(width);
    for (int i = 0; i < kRows * width; ++i)
      values[i] = Bf16(std::sin(float(i + 3) * .137f) +
                       .4f * std::cos(float(i + 7) * .073f));
    for (int c = 0; c < width; ++c)
      query[c] = Bf16(std::cos(float(c + 5) * .17f));
    auto table = Table(values, kRows, width, kRows);
    ASSERT_TRUE(table.ok()) << table.status();
    auto output = (*table)->Evaluate({{3, 1}, {8, -1}, {5, 1}});
    ASSERT_TRUE(output.ok()) << output.status();
    for (int c = 0; c < width; ++c)
      EXPECT_FLOAT_EQ((*output)[c], values[3 * width + c] -
                                        values[8 * width + c] +
                                        values[5 * width + c]);
    auto actual = (*table)->Nearest(query, kRows);
    ASSERT_TRUE(actual.ok()) << actual.status();
    ASSERT_EQ(actual->size(), static_cast<size_t>(kRows));
    std::vector<EmbeddingMatch> expected;
    for (int r = 0; r < kRows; ++r) {
      double dot = 0, a2 = 0, b2 = 0, d2 = 0;
      for (int c = 0; c < width; ++c) {
        const double a = values[r * width + c], b = query[c];
        dot += a * b;
        a2 += a * a;
        b2 += b * b;
        d2 += (a - b) * (a - b);
      }
      expected.push_back({r, static_cast<float>(dot / std::sqrt(a2 * b2)),
                          static_cast<float>(std::sqrt(d2))});
    }
    std::sort(expected.begin(), expected.end(),
              [](const EmbeddingMatch& a, const EmbeddingMatch& b) {
                if (a.cosine_similarity != b.cosine_similarity)
                  return a.cosine_similarity > b.cosine_similarity;
                return a.token_id < b.token_id;
              });
    // Width one has mathematically tied +/-1 scores; small FP32 rounding can
    // perturb them, so the exact tie-order contract is tested separately above.
    for (const auto& match : *actual) {
      const auto found = std::find_if(expected.begin(), expected.end(),
                                      [&](const EmbeddingMatch& e) {
                                        return e.token_id == match.token_id;
                                      });
      ASSERT_NE(found, expected.end());
      EXPECT_NEAR(match.cosine_similarity, found->cosine_similarity, 3e-6);
      EXPECT_NEAR(match.l2_distance, found->l2_distance, 2e-4);
    }
    if (width == 1)
      continue;
    for (int r = 0; r < kRows; ++r)
      EXPECT_EQ((*actual)[r].token_id, expected[r].token_id);
  }
}

TEST_F(EmbeddingAlgebraTableTest, ZeroExpressionsAreValidButHaveNoCosine) {
  auto table = Table({1, 2, 3, 0, 0, 0}, 2, 3, 2);
  ASSERT_TRUE(table.ok());
  auto zero = (*table)->Evaluate({{0, 1}, {0, -1}});
  ASSERT_TRUE(zero.ok());
  for (float value : *zero)
    EXPECT_EQ(value, 0);
  EXPECT_FALSE((*table)->Nearest(zero->span()).ok());
  auto all_zero = Table({0, 0, 0}, 1, 3, 1);
  ASSERT_TRUE(all_zero.ok());
  auto empty = (*all_zero)->Nearest({1, 0, 0});
  ASSERT_TRUE(empty.ok());
  EXPECT_TRUE(empty->empty());
}

TEST_F(EmbeddingAlgebraTableTest, RejectsMalformedExpressionsAndQueries) {
  auto table = Table({1, 0, 0}, 1, 3, 1);
  ASSERT_TRUE(table.ok());
  EXPECT_FALSE((*table)->Evaluate({}).ok());
  EXPECT_FALSE((*table)->Evaluate({{-1, 1}}).ok());
  EXPECT_FALSE((*table)->Evaluate({{1, 1}}).ok());
  EXPECT_FALSE((*table)->Evaluate({{0, 0}}).ok());
  EXPECT_FALSE((*table)->Evaluate({{0, 2}}).ok());
  EXPECT_FALSE((*table)->Nearest({1, 0}).ok());
  EXPECT_FALSE((*table)->Nearest({1, 0, 0}, 0).ok());
  EXPECT_FALSE((*table)->Nearest({1, 0, 0}, -1).ok());
  EXPECT_FALSE(
      (*table)->Nearest({std::numeric_limits<float>::infinity(), 0, 0}).ok());
  EXPECT_FALSE(
      (*table)->Nearest({std::numeric_limits<float>::quiet_NaN(), 0, 0}).ok());
  EXPECT_FALSE(
      (*table)->Nearest({std::numeric_limits<float>::max(), 0, 0}).ok());
}

TEST_F(EmbeddingAlgebraTableTest, RejectsMalformedBuffersAndNonfiniteTables) {
  auto weights = Upload({1, 0, 0});
  ASSERT_TRUE(weights.ok());
  EXPECT_FALSE(
      EmbeddingAlgebraTable::Create(*executor_, *weights, 1, 3, 0).ok());
  EXPECT_FALSE(
      EmbeddingAlgebraTable::Create(*executor_, *weights, 1, 3, 2).ok());
  EXPECT_FALSE(
      EmbeddingAlgebraTable::Create(*executor_, *weights, 0, 3, 1).ok());
  EXPECT_FALSE(
      EmbeddingAlgebraTable::Create(*executor_, *weights, 1, -3, 1).ok());
  EXPECT_FALSE(
      EmbeddingAlgebraTable::Create(*executor_, *weights, 1, 4, 1).ok());
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok());
  EXPECT_FALSE(EmbeddingAlgebraTable::Create(**other, *weights, 1, 3, 1).ok());
  EXPECT_FALSE(
      Table({1, std::numeric_limits<float>::infinity()}, 1, 2, 1).ok());
  EXPECT_FALSE(
      Table({1, std::numeric_limits<float>::quiet_NaN()}, 1, 2, 1).ok());
}

// This checkpoint intentionally has no decoder, head, or norm tensors. Loading
// it successfully verifies that embedding algebra has no dependency on them.
constexpr char kConfig[] = R"({
  "model_type":"qwen3_5", "text_config": {
    "model_type":"qwen3_5_text", "dtype":"bfloat16", "hidden_act":"silu",
    "hidden_size":8, "intermediate_size":8, "num_hidden_layers":2,
    "num_attention_heads":1, "num_key_value_heads":1, "head_dim":8,
    "vocab_size":4, "max_position_embeddings":8,
    "linear_conv_kernel_dim":2, "linear_key_head_dim":4,
    "linear_num_key_heads":1, "linear_num_value_heads":1,
    "linear_value_head_dim":4, "full_attention_interval":2,
    "bos_token_id":0, "eos_token_id":3, "rms_norm_eps":0.000001,
    "attn_output_gate":true, "tie_word_embeddings":false,
    "output_gate_type":"swish", "attention_bias":false,
    "attention_dropout":0, "partial_rotary_factor":0.25,
    "layer_types":["linear_attention", "full_attention"],
    "rope_parameters":{"rope_type":"default", "rope_theta":10000000,
      "partial_rotary_factor":0.25, "mrope_interleaved":true,
      "mrope_section":[1,0,0]}
  }, "quantization_config":{"quant_method":"fp8", "fmt":"e4m3",
    "activation_scheme":"dynamic", "weight_block_size":[128,128]}
})";

TEST_F(EmbeddingAlgebraTableTest, LoadsOnlyEmbeddingAndChecksShape) {
  std::string pattern = testing::TempDir() + "/qwen-embedding-XXXXXX";
  char* created = mkdtemp(pattern.data());
  ASSERT_NE(created, nullptr);
  const std::filesystem::path directory(created);
  std::ofstream(directory / "config.json") << kConfig;
  auto write_tensor = [&](const std::string& dtype, int rows, int columns) {
    const int bytes = rows * columns * (dtype == "BF16" ? 2 : 4);
    std::string header = absl::StrCat(
        "{\"model.language_model.embed_tokens.weight\":{\"dtype\":\"", dtype,
        "\",\"shape\":[", rows, ",", columns, "],\"data_offsets\":[0,", bytes,
        "]}}");
    header.append((8 - header.size() % 8) % 8, ' ');
    std::ofstream output(directory / "model.safetensors", std::ios::binary);
    const uint64_t length = header.size();
    for (int b = 0; b < 8; ++b)
      output.put((length >> (8 * b)) & 255);
    output << header;
    for (int b = 0; b < bytes; ++b)
      output.put(0);
  };
  write_tensor("BF16", 4, 8);
  {
    auto table = EmbeddingAlgebraTable::Load(*executor_, directory, 4);
    ASSERT_TRUE(table.ok()) << table.status();
    EXPECT_EQ((*table)->dimensions(), 8);
    EXPECT_EQ((*table)->vocab_size(), 4);
  }
  EXPECT_FALSE(EmbeddingAlgebraTable::Load(*executor_, directory, 5).ok());
  write_tensor("BF16", 8, 4);
  EXPECT_FALSE(EmbeddingAlgebraTable::Load(*executor_, directory, 4).ok());
  write_tensor("F32", 4, 8);
  EXPECT_FALSE(EmbeddingAlgebraTable::Load(*executor_, directory, 4).ok());
  std::error_code error;
  std::filesystem::remove_all(directory, error);
  EXPECT_FALSE(error) << error.message();
}

}  // namespace
}  // namespace pluto::llm::qwen
