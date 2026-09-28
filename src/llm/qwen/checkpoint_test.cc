#include "src/llm/qwen/checkpoint.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"

namespace pluto::llm::qwen {
namespace {

constexpr char kConfig[] = R"({
  "model_type": "qwen3_5",
  "text_config": {
    "model_type": "qwen3_5_text", "dtype": "bfloat16", "hidden_act": "silu",
    "hidden_size": 5120, "intermediate_size": 17408, "num_hidden_layers": 4,
    "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
    "vocab_size": 248320, "max_position_embeddings": 262144,
    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
    "linear_num_key_heads": 16, "linear_num_value_heads": 48,
    "linear_value_head_dim": 128, "full_attention_interval": 4,
    "bos_token_id": 248044, "eos_token_id": 248044, "rms_norm_eps": 1e-6,
    "attn_output_gate": true, "tie_word_embeddings": false,
    "output_gate_type": "swish", "attention_bias": false,
    "attention_dropout": 0.0, "partial_rotary_factor": 0.25,
    "layer_types": ["linear_attention", "linear_attention",
                    "linear_attention", "full_attention"],
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
      "partial_rotary_factor": 0.25, "mrope_interleaved": true,
      "mrope_section": [11, 11, 10]}
  },
  "quantization_config": {"quant_method": "fp8", "fmt": "e4m3",
    "activation_scheme": "dynamic", "weight_block_size": [128, 128]}
})";

std::string Replace(std::string input, const std::string& from,
                    const std::string& to) {
  const size_t offset = input.find(from);
  EXPECT_NE(offset, std::string::npos);
  if (offset != std::string::npos)
    input.replace(offset, from.size(), to);
  return input;
}

class QwenCheckpointTest : public testing::Test {
 protected:
  void SetUp() override {
    std::string pattern = testing::TempDir() + "/qwen-checkpoint-XXXXXX";
    char* result = mkdtemp(pattern.data());
    ASSERT_NE(result, nullptr);
    directory_ = result;
  }

  void TearDown() override {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
    EXPECT_FALSE(error) << error.message();
  }

  void Write(const std::string& name, const std::string& contents) {
    std::ofstream out(directory_ / name, std::ios::binary);
    out.write(contents.data(), contents.size());
    ASSERT_TRUE(out.good());
  }

  void Shard(const std::string& header, const std::string& payload,
             const std::string& name = "model.safetensors") {
    std::string padded = header;
    padded.append((8 - (padded.size() % 8)) % 8, ' ');
    const uint64_t length = padded.size();
    std::string file;
    for (int i = 0; i < 8; ++i)
      file.push_back((length >> (8 * i)) & 255);
    file += padded;
    file += payload;
    Write(name, file);
  }

  std::filesystem::path directory_;
};

TEST(QwenConfigTest, ReadsHybridDecoderAndFp8Settings) {
  auto config = ParseConfig(kConfig);
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->hidden_size, 5120);
  EXPECT_EQ(config->intermediate_size, 17408);
  EXPECT_EQ(config->num_attention_heads, 24);
  EXPECT_EQ(config->head_dim, 256);
  EXPECT_EQ(config->linear_num_value_heads, 48);
  EXPECT_EQ(config->linear_num_key_heads, 16);
  EXPECT_EQ(config->linear_conv_kernel_dim, 4);
  EXPECT_EQ(config->eos_token_id, 248044);
  EXPECT_EQ(config->layer_types,
            (std::vector<LayerType>{
                LayerType::kLinearAttention, LayerType::kLinearAttention,
                LayerType::kLinearAttention, LayerType::kFullAttention}));
  EXPECT_DOUBLE_EQ(config->rms_norm_eps, 1e-6);
  EXPECT_DOUBLE_EQ(config->rope_theta, 10000000);
  EXPECT_DOUBLE_EQ(config->partial_rotary_factor, 0.25);
  EXPECT_EQ(config->output_gate_type, "swish");
  EXPECT_EQ(config->weight_block_size, (std::array<int, 2>{128, 128}));
  EXPECT_EQ(config->mrope_section, (std::array<int, 3>{11, 11, 10}));
  EXPECT_TRUE(config->mrope_interleaved);
  EXPECT_TRUE(config->attn_output_gate);
  EXPECT_FALSE(config->tie_word_embeddings);
}

TEST(QwenConfigTest, RejectsInvalidDimensionsAndUnsupportedModes) {
  for (const auto& [from, to] :
       std::vector<std::pair<std::string, std::string>>{
           {"\"hidden_size\": 5120", "\"hidden_size\": 0"},
           {"\"num_hidden_layers\": 4", "\"num_hidden_layers\": 64"},
           {"\"num_key_value_heads\": 4", "\"num_key_value_heads\": 5"},
           {"\"linear_num_key_heads\": 16", "\"linear_num_key_heads\": 17"},
           {"\"head_dim\": 256", "\"head_dim\": 257"},
           {"\"rms_norm_eps\": 1e-6", "\"rms_norm_eps\": -1"},
           {"\"rms_norm_eps\": 1e-6", "\"rms_norm_eps\": 1e999"},
           {"\"eos_token_id\": 248044", "\"eos_token_id\": 248320"},
           {"\"eos_token_id\": 248044", "\"eos_token_id\": 2.0"},
           {"\"weight_block_size\": [128, 128]",
            "\"weight_block_size\": [0, 128]"},
           {"\"attention_bias\": false", "\"attention_bias\": true"},
           {"\"attn_output_gate\": true", "\"attn_output_gate\": false"},
           {"\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true"},
           {"\"output_gate_type\": \"swish\"",
            "\"output_gate_type\": \"sigmoid\""},
           {"\"rope_type\": \"default\"", "\"rope_type\": \"yarn\""},
           {"\"output_gate_type\": \"swish\"",
            "\"output_gate_type\": \"unknown\""},
           {"\"fmt\": \"e4m3\"", "\"fmt\": \"e5m2\""}}) {
    SCOPED_TRACE(to);
    EXPECT_FALSE(ParseConfig(Replace(kConfig, from, to)).ok());
  }
}

TEST(QwenConfigTest, RejectsMalformedJsonAndDuplicateKeys) {
  for (const std::string& json :
       {std::string(kConfig) + " x", Replace(kConfig, "5120", "05120"),
        Replace(kConfig, "5120", "5120, \"hidden_size\": 1"),
        Replace(kConfig, "5120", "5120."), Replace(kConfig, "5120", "+5120"),
        Replace(kConfig, "\"silu\"", "\"\\uD800\""),
        Replace(kConfig, "\"silu\"", std::string("\"\xc0\x80\"")),
        Replace(kConfig, "\"silu\"", "truegarbage")}) {
    EXPECT_FALSE(ParseConfig(json).ok());
  }
  std::string deep = std::string(100, '[') + "0" + std::string(100, ']');
  EXPECT_FALSE(ParseConfig(deep).ok());
}

TEST_F(QwenCheckpointTest, LoadsConfigurationFromDirectory) {
  Write("config.json", kConfig);
  auto config = LoadConfig(directory_);
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->hidden_size, 5120);
}

TEST_F(QwenCheckpointTest, PreservesFp8Bfloat16AndFloat32Bytes) {
  Shard(R"({"__metadata__":{"format":"pt"},
    "weight":{"dtype":"F8_E4M3","shape":[2,2],"data_offsets":[0,4]},
    "scale":{"dtype":"BF16","shape":[1,1],"data_offsets":[4,6]},
    "scalar":{"dtype":"F32","shape":[],"data_offsets":[6,10]}})",
        std::string("\x00\x38\x7e\xff\x80\x3f\x00\x00\x80\x3f", 10));
  auto checkpoint = SafetensorsCheckpoint::Open(directory_);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  EXPECT_EQ((*checkpoint)->tensor_names(),
            (std::vector<std::string>{"scalar", "scale", "weight"}));
  auto weight = (*checkpoint)->Tensor("weight");
  ASSERT_TRUE(weight.ok()) << weight.status();
  EXPECT_EQ(weight->dtype, TensorDType::kF8E4M3);
  EXPECT_EQ(weight->shape, (std::vector<int64_t>{2, 2}));
  EXPECT_EQ(std::vector<uint8_t>(weight->bytes.begin(), weight->bytes.end()),
            (std::vector<uint8_t>{0, 0x38, 0x7e, 0xff}));
  auto scale = (*checkpoint)->Tensor("scale");
  ASSERT_TRUE(scale.ok()) << scale.status();
  EXPECT_EQ(scale->dtype, TensorDType::kBF16);
  EXPECT_EQ(scale->bytes.size(), 2u);
  auto scalar = (*checkpoint)->Tensor("scalar");
  ASSERT_TRUE(scalar.ok()) << scalar.status();
  EXPECT_EQ(scalar->dtype, TensorDType::kF32);
  EXPECT_TRUE(scalar->shape.empty());
  EXPECT_EQ((*checkpoint)->Tensor("missing").status().code(),
            absl::StatusCode::kNotFound);
  auto again = (*checkpoint)->Tensor("weight");
  ASSERT_TRUE(again.ok());
  EXPECT_EQ(again->bytes.data(), weight->bytes.data());
}

TEST_F(QwenCheckpointTest, LoadsShardsLazilyAndKeepsEarlierViewsValid) {
  Write("model.safetensors.index.json", R"({"weight_map":{
    "first":"part1.safetensors","second":"part2.safetensors"}})");
  Shard(R"({"first":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[0,1]}})",
        "a", "part1.safetensors");
  auto checkpoint = SafetensorsCheckpoint::Open(directory_);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  auto first = (*checkpoint)->Tensor("first");
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_EQ((*checkpoint)->Tensor("second").status().code(),
            absl::StatusCode::kNotFound);
  Shard(R"({"second":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[0,1]}})",
        "b", "part2.safetensors");
  auto second = (*checkpoint)->Tensor("second");
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(second->bytes[0], 'b');
  EXPECT_EQ(first->bytes[0], 'a');
}

TEST_F(QwenCheckpointTest, AcceptsEmptyTensorsAndEscapedNames) {
  Shard(R"({"empty":{"dtype":"BF16","shape":[999,0],"data_offsets":[0,0]},
    "\u0061\uD83D\uDE00":{"dtype":"F32","shape":[],"data_offsets":[0,4]}})",
        "abcd");
  auto checkpoint = SafetensorsCheckpoint::Open(directory_);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  auto empty = (*checkpoint)->Tensor("empty");
  ASSERT_TRUE(empty.ok());
  EXPECT_TRUE(empty->bytes.empty());
  auto unicode = (*checkpoint)->Tensor("a\xf0\x9f\x98\x80");
  ASSERT_TRUE(unicode.ok()) << unicode.status();
  EXPECT_EQ(unicode->bytes.size(), 4u);
}

TEST_F(QwenCheckpointTest, RejectsTruncatedAndOversizedHeaderLengths) {
  for (const std::string& file : {std::string("short"), std::string(8, '\xff'),
                                  std::string("\x20\0\0\0\0\0\0\0{}", 10)}) {
    Write("model.safetensors", file);
    EXPECT_FALSE(SafetensorsCheckpoint::Open(directory_).ok());
  }
}

TEST_F(QwenCheckpointTest, RejectsBadByteRangesShapesAndMetadata) {
  for (
      const char* header :
      {R"({"w":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[0,1]}})",
       R"({"w":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[0,2]}})",
       R"({"w":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[1,0]}})",
       R"({"w":{"dtype":"F8_E4M3","shape":[0],"data_offsets":[1,1]}})",
       R"({"w":{"dtype":"F8_E4M3","shape":[-1],"data_offsets":[0,1]}})",
       R"({"w":{"dtype":"F8_E4M3","shape":[1.0],"data_offsets":[0,1]}})",
       R"({"w":{"dtype":"F8_E4M3","shape":[9223372036854775807,3],"data_offsets":[0,1]}})",
       R"({"w":{"dtype":"F8_E4M3","shape":[18446744073709551616],"data_offsets":[0,1]}})",
       R"({"w":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[0,1e0]}})",
       R"({"w":{"dtype":"F16","shape":[1],"data_offsets":[0,1]}})",
       R"({"w":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[0,1]},
         "x":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[0,1]}})",
       R"({"w":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[0,1]},"w":{}})",
       R"({"__metadata__":{"format":1}})", R"({})"}) {
    SCOPED_TRACE(header);
    Shard(header, "x");
    EXPECT_FALSE(SafetensorsCheckpoint::Open(directory_).ok());
  }
}

TEST_F(QwenCheckpointTest, RejectsIndexTraversalAndMalformedIndex) {
  for (const char* index :
       {R"({"weight_map":{"w":"../outside.safetensors"}})",
        R"({"weight_map":{"w":"/tmp/outside.safetensors"}})",
        R"({"weight_map":{"w":"dir/file.safetensors"}})",
        R"({"weight_map":{"w":"file\u0000.safetensors"}})",
        R"({"weight_map":{"w":"file.safetensors","w":"other.safetensors"}})",
        R"({"weight_map":{}})", R"({"weight_map":{"w":null}})", "malformed"}) {
    SCOPED_TRACE(index);
    Write("model.safetensors.index.json", index);
    EXPECT_FALSE(SafetensorsCheckpoint::Open(directory_).ok());
  }
}

TEST_F(QwenCheckpointTest, ChecksAllShardEntriesAgainstIndexBeforeReturning) {
  Shard(R"({"w":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[0,1]}})", "x",
        "part.safetensors");
  for (
      const char* index :
      {R"({"weight_map":{"w":"part.safetensors","missing":"part.safetensors"}})",
       R"({"weight_map":{"other":"part.safetensors"}})"}) {
    Write("model.safetensors.index.json", index);
    auto checkpoint = SafetensorsCheckpoint::Open(directory_);
    ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
    EXPECT_FALSE(
        (*checkpoint)->Tensor((*checkpoint)->tensor_names().front()).ok());
  }
}

// Opt-in metadata validation of the real checkpoint. It maps the shards but
// never scans or copies the tens of gigabytes of tensor payloads.
TEST(QwenCheckpointIntegrationTest, ValidatesOfficialCheckpointMetadata) {
  const char* directory = std::getenv("PLUTO_QWEN_CHECKPOINT_DIR");
  if (directory == nullptr || *directory == '\0')
    GTEST_SKIP() << "Set PLUTO_QWEN_CHECKPOINT_DIR to Qwen3.8-27B-FP8";
  auto config = LoadConfig(directory);
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->num_hidden_layers, 64);
  EXPECT_EQ(config->hidden_size, 5120);
  auto checkpoint = SafetensorsCheckpoint::Open(directory);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status();
  int fp8_count = 0;
  int bf16_count = 0;
  for (const std::string& name : (*checkpoint)->tensor_names()) {
    auto tensor = (*checkpoint)->Tensor(name);
    ASSERT_TRUE(tensor.ok()) << name << ": " << tensor.status();
    fp8_count += tensor->dtype == TensorDType::kF8E4M3;
    bf16_count += tensor->dtype == TensorDType::kBF16;
  }
  EXPECT_GT(fp8_count, 0);
  EXPECT_GT(bf16_count, 0);
}

}  // namespace
}  // namespace pluto::llm::qwen
