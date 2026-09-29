#include "src/llm/qwen/model.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layer.h"
#include "src/llm/layer_hooks.h"
#include "src/util/status_macros.h"

namespace pluto::llm::qwen {
namespace {

constexpr char kTinyConfig[] = R"({
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

struct SavedTensor {
  std::string dtype;
  std::vector<int64_t> shape;
  std::string bytes;
  size_t device_bytes;
  std::vector<float> values;
};

float Bfloat16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fff + ((bits >> 16) & 1);
  return std::bit_cast<float>(bits & 0xffff0000);
}

float Fp8Value(uint8_t bits) {
  const int exponent = (bits >> 3) & 15;
  const int mantissa = bits & 7;
  const float magnitude =
      exponent == 0 ? std::ldexp(float(mantissa), -9)
                    : std::ldexp(1.0f + mantissa / 8.0f, exponent - 7);
  return bits & 128 ? -magnitude : magnitude;
}

// Enumerating the tiny finite codebook makes this independent of CUDA casts.
uint8_t Fp8Bits(float value) {
  const float magnitude = std::abs(value);
  int best = 0;
  float distance = magnitude;
  for (int code = 1; code <= 126; ++code) {
    const float candidate = std::abs(magnitude - Fp8Value(code));
    if (candidate < distance || (candidate == distance && !(code & 1))) {
      best = code;
      distance = candidate;
    }
  }
  return best | (std::signbit(value) ? 128 : 0);
}

float Sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }
float Silu(float value) { return value * Sigmoid(value); }

struct TinyHistory {
  std::vector<float> previous_qkv = std::vector<float>(12);
  std::vector<float> recurrent = std::vector<float>(16);
  std::vector<std::vector<float>> keys;
  std::vector<std::vector<float>> values;
};

void AppendLittleEndian(uint32_t bits, int bytes, std::string* output) {
  for (int byte = 0; byte < bytes; ++byte)
    output->push_back((bits >> (8 * byte)) & 255);
}

// Every attention/MLP output is zero, so each block is an identity residual.
// The embeddings and head are one-hot; their logits have a known value after
// final RMS normalization. FP8 zero matrices still exercise scale loading and
// dynamic activation quantization through both types of attention block.
class QwenModelTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
    std::string pattern = testing::TempDir() + "/qwen-model-XXXXXX";
    char* result = mkdtemp(pattern.data());
    ASSERT_NE(result, nullptr);
    directory_ = result;
    std::ofstream(directory_ / "config.json") << kTinyConfig;
    std::vector<float> identity(4 * 8, 0);
    for (int token = 0; token < 4; ++token)
      identity[token * 8 + token] = 1;
    Add("model.language_model.embed_tokens.weight", {4, 8}, "BF16", identity,
        true);
    Add("lm_head.weight", {4, 8}, "BF16", identity, true);
    Constant("model.language_model.norm.weight", {8});
    for (int layer = 0; layer < 2; ++layer) {
      const std::string prefix =
          absl::StrCat("model.language_model.layers.", layer, ".");
      Constant(prefix + "input_layernorm.weight", {8});
      Constant(prefix + "post_attention_layernorm.weight", {8});
      Fp8(prefix + "mlp.gate_proj.weight", 8, 8);
      Fp8(prefix + "mlp.up_proj.weight", 8, 8);
      Fp8(prefix + "mlp.down_proj.weight", 8, 8);
      if (layer == 0) {
        Fp8(prefix + "linear_attn.in_proj_qkv.weight", 12, 8);
        Fp8(prefix + "linear_attn.in_proj_z.weight", 4, 8);
        Constant(prefix + "linear_attn.in_proj_a.weight", {1, 8}, 0, true);
        Constant(prefix + "linear_attn.in_proj_b.weight", {1, 8}, 0, true);
        Fp8(prefix + "linear_attn.out_proj.weight", 8, 4);
        Constant(prefix + "linear_attn.conv1d.weight", {12, 1, 2});
        Add(prefix + "linear_attn.A_log", {1}, "F32", {0}, false);
        Add(prefix + "linear_attn.dt_bias", {1}, "F32", {0}, false);
        Constant(prefix + "linear_attn.norm.weight", {4}, 1);
      } else {
        Fp8(prefix + "self_attn.q_proj.weight", 16, 8);
        Fp8(prefix + "self_attn.k_proj.weight", 8, 8);
        Fp8(prefix + "self_attn.v_proj.weight", 8, 8);
        Fp8(prefix + "self_attn.o_proj.weight", 8, 8);
        Constant(prefix + "self_attn.q_norm.weight", {8});
        Constant(prefix + "self_attn.k_norm.weight", {8});
      }
    }
    WriteCheckpoint();
  }

  void TearDown() override {
    if (executor_ != nullptr) {
      EXPECT_TRUE(executor_->Synchronize().ok());
      executor_.reset();
    }
    if (!directory_.empty()) {
      std::error_code error;
      std::filesystem::remove_all(directory_, error);
      EXPECT_FALSE(error) << error.message();
    }
  }

  void Add(std::string name, std::vector<int64_t> shape, std::string dtype,
           const std::vector<float>& values, bool matrix) {
    std::string bytes;
    std::vector<float> decoded;
    for (float value : values) {
      const uint32_t bits = std::bit_cast<uint32_t>(value);
      if (dtype == "BF16") {
        AppendLittleEndian(bits >> 16, 2, &bytes);
        decoded.push_back(std::bit_cast<float>(bits & 0xffff0000));
      } else if (dtype == "F32") {
        AppendLittleEndian(bits, 4, &bytes);
        decoded.push_back(value);
      } else {
        const uint8_t code = Fp8Bits(value);
        bytes.push_back(code);
        decoded.push_back(Fp8Value(code));
      }
    }
    const size_t device_bytes =
        matrix ? bytes.size() : values.size() * sizeof(float);
    tensors_.insert_or_assign(
        std::move(name),
        SavedTensor{std::move(dtype), std::move(shape), std::move(bytes),
                    device_bytes, std::move(decoded)});
  }

  void Constant(const std::string& name, std::vector<int64_t> shape,
                float value = 0, bool matrix = false) {
    size_t count = 1;
    for (int64_t dimension : shape)
      count *= dimension;
    Add(name, std::move(shape), "BF16", std::vector<float>(count, value),
        matrix);
  }

  void Fp8(const std::string& name, int rows, int columns) {
    Add(name, {rows, columns}, "F8_E4M3", std::vector<float>(rows * columns),
        true);
    Constant(name.substr(0, name.size() - 7) + ".weight_scale_inv", {1, 1}, 1);
  }

  void WriteCheckpoint() {
    std::string header = "{";
    std::string payload;
    for (const auto& [name, tensor] : tensors_) {
      if (header.size() > 1)
        header += ",";
      absl::StrAppend(&header, "\"", name, "\":{\"dtype\":\"", tensor.dtype,
                      "\",\"shape\":[");
      for (size_t i = 0; i < tensor.shape.size(); ++i) {
        if (i > 0)
          header += ",";
        absl::StrAppend(&header, tensor.shape[i]);
      }
      absl::StrAppend(&header, "],\"data_offsets\":[", payload.size(), ",",
                      payload.size() + tensor.bytes.size(), "]}");
      payload += tensor.bytes;
    }
    header += "}";
    header.append((8 - (header.size() % 8)) % 8, ' ');
    std::ofstream output(directory_ / "model.safetensors", std::ios::binary);
    const uint64_t size = header.size();
    for (int byte = 0; byte < 8; ++byte)
      output.put((size >> (byte * 8)) & 255);
    output.write(header.data(), header.size());
    output.write(payload.data(), payload.size());
    ASSERT_TRUE(output.good());
  }

  std::vector<float> CopyLogits(Model& model) {
    auto logits = model.Logits();
    EXPECT_TRUE(logits.ok()) << logits.status();
    if (!logits.ok())
      return {};
    auto output = cuda::PageLockedHostArray<float>::Allocate(
        *executor_, logits->size_bytes() / sizeof(float));
    EXPECT_TRUE(output.ok()) << output.status();
    if (!output.ok())
      return {};
    EXPECT_EQ(
        cudaMemcpyAsync(output->data(), logits->data(), logits->size_bytes(),
                        cudaMemcpyDeviceToHost, executor_->stream()),
        cudaSuccess);
    EXPECT_TRUE(executor_->Synchronize().ok());
    return std::vector<float>(output->begin(), output->end());
  }

  // Unlike the identity fixture, every decoder branch changes the residual.
  // Two query heads share one KV head, and both convolution taps are nonzero.
  void MakeNonzeroHybrid() {
    std::string config = kTinyConfig;
    for (const auto& [before, after] :
         std::vector<std::pair<std::string, std::string>>{
             {"\"num_attention_heads\":1", "\"num_attention_heads\":2"},
             {"\"head_dim\":8", "\"head_dim\":4"},
             {"\"partial_rotary_factor\":0.25",
              "\"partial_rotary_factor\":0.5"}}) {
      size_t position = 0;
      while ((position = config.find(before, position)) != std::string::npos) {
        config.replace(position, before.size(), after);
        position += after.size();
      }
    }
    std::ofstream(directory_ / "config.json") << config;
    const std::string full = "model.language_model.layers.1.self_attn.";
    Fp8(full + "k_proj.weight", 4, 8);
    Fp8(full + "v_proj.weight", 4, 8);
    Constant(full + "q_norm.weight", {4}, 0.125f);
    Constant(full + "k_norm.weight", {4}, -0.125f);
    int matrix = 0;
    for (auto& [name, tensor] : tensors_) {
      if (tensor.dtype != "F8_E4M3")
        continue;
      ++matrix;
      for (size_t i = 0; i < tensor.values.size(); ++i) {
        const int row = i / tensor.shape[1];
        const int column = i % tensor.shape[1];
        const float value = ((row * 7 + column * 3 + matrix) % 9 - 4) / 16.0f;
        const uint8_t code = Fp8Bits(value);
        tensor.bytes[i] = code;
        tensor.values[i] = Fp8Value(code);
      }
    }
    const std::string delta = "model.language_model.layers.0.linear_attn.";
    std::vector<float> conv(24);
    for (int channel = 0; channel < 12; ++channel) {
      conv[2 * channel] = 0.25f;
      conv[2 * channel + 1] = 0.75f;
    }
    Add(delta + "conv1d.weight", {12, 1, 2}, "BF16", conv, false);
    Constant(delta + "in_proj_a.weight", {1, 8}, 0.125f, true);
    Constant(delta + "in_proj_b.weight", {1, 8}, -0.125f, true);
    Constant(delta + "dt_bias", {1}, 0.25f);
    Constant(delta + "norm.weight", {4}, 0.75f);
    WriteCheckpoint();
  }

  std::vector<float> Normalize(const std::vector<float>& input,
                               const std::string& name) {
    float sum = 0;
    for (float x : input)
      sum += x * x;
    const float inverse = 1 / std::sqrt(sum / input.size() + 1e-6f);
    const auto& weight = tensors_.at(name).values;
    std::vector<float> result(input.size());
    for (size_t i = 0; i < input.size(); ++i)
      result[i] = Bfloat16(input[i] * inverse * (1 + weight[i]));
    return result;
  }

  std::vector<float> Project(const std::string& name,
                             std::vector<float> input) {
    const auto& tensor = tensors_.at(name);
    if (tensor.dtype == "F8_E4M3") {
      float maximum = 0;
      for (float x : input)
        maximum = std::max(maximum, std::abs(x));
      const float scale = maximum / 448;
      for (float& x : input)
        x = Fp8Value(Fp8Bits(x / std::max(scale, 1e-12f))) * scale;
    }
    std::vector<float> result(tensor.shape[0]);
    for (size_t row = 0; row < result.size(); ++row) {
      float sum = 0;
      for (size_t column = 0; column < input.size(); ++column)
        sum += tensor.values[row * input.size() + column] * input[column];
      result[row] = Bfloat16(sum);
    }
    return result;
  }

  // Scalar end-to-end reference for this deliberately tiny configuration.
  // It uses complete softmax and explicit 4x4 recurrent matrices, without any
  // GPU operators or the implementations used by attention_ops_test.
  std::vector<float> ReferenceStep(int token, TinyHistory* history) {
    const std::string root = "model.language_model.";
    const auto& embedding = tensors_.at(root + "embed_tokens.weight").values;
    std::vector<float> hidden(embedding.begin() + token * 8,
                              embedding.begin() + (token + 1) * 8);
    for (int layer = 0; layer < 2; ++layer) {
      const std::string prefix = absl::StrCat(root, "layers.", layer, ".");
      const auto input = Normalize(hidden, prefix + "input_layernorm.weight");
      std::vector<float> attention;
      if (layer == 0) {
        const std::string p = prefix + "linear_attn.";
        const auto qkv = Project(p + "in_proj_qkv.weight", input);
        const auto z = Project(p + "in_proj_z.weight", input);
        const float a = Project(p + "in_proj_a.weight", input)[0];
        const float b = Project(p + "in_proj_b.weight", input)[0];
        const auto& conv_weight = tensors_.at(p + "conv1d.weight").values;
        std::vector<float> conv(12);
        for (int i = 0; i < 12; ++i)
          conv[i] = Bfloat16(
              Silu(Bfloat16(history->previous_qkv[i] * conv_weight[2 * i] +
                            qkv[i] * conv_weight[2 * i + 1])));
        history->previous_qkv = qkv;
        float q_square = 1e-6f, k_square = 1e-6f;
        for (int i = 0; i < 4; ++i) {
          q_square += conv[i] * conv[i];
          k_square += conv[4 + i] * conv[4 + i];
        }
        std::vector<float> q(4), k(4), core(4);
        for (int i = 0; i < 4; ++i) {
          q[i] = conv[i] / std::sqrt(4 * q_square);
          k[i] = conv[4 + i] / std::sqrt(k_square);
        }
        const float alpha = a + tensors_.at(p + "dt_bias").values[0];
        const float decay = std::exp(
            -std::exp(tensors_.at(p + "A_log").values[0]) *
            (std::max(alpha, 0.0f) + std::log1p(std::exp(-std::abs(alpha)))));
        for (float& x : history->recurrent)
          x *= decay;
        for (int v = 0; v < 4; ++v) {
          float prediction = 0;
          for (int d = 0; d < 4; ++d)
            prediction += history->recurrent[d * 4 + v] * k[d];
          const float update =
              (conv[8 + v] - prediction) * Bfloat16(Sigmoid(b));
          for (int d = 0; d < 4; ++d) {
            history->recurrent[d * 4 + v] += k[d] * update;
            core[v] += history->recurrent[d * 4 + v] * q[d];
          }
          core[v] = Bfloat16(core[v]);
        }
        float square = 0;
        for (float x : core)
          square += x * x;
        const float inverse = 1 / std::sqrt(square / 4 + 1e-6f);
        const auto& weight = tensors_.at(p + "norm.weight").values;
        for (int i = 0; i < 4; ++i)
          core[i] = Bfloat16(Bfloat16(Bfloat16(core[i] * inverse) * weight[i]) *
                             Silu(z[i]));
        attention = Project(p + "out_proj.weight", core);
      } else {
        const std::string p = prefix + "self_attn.";
        const auto q_gate = Project(p + "q_proj.weight", input);
        auto key = Project(p + "k_proj.weight", input);
        const auto value = Project(p + "v_proj.weight", input);
        const int position = history->keys.size();
        auto rotate = [&](std::vector<float> x, const std::string& norm) {
          x = Normalize(x, p + norm);
          const float cosine = Bfloat16(std::cos(float(position)));
          const float sine = Bfloat16(std::sin(float(position)));
          const float first = x[0], second = x[1];
          x[0] = Bfloat16(Bfloat16(first * cosine) - Bfloat16(second * sine));
          x[1] = Bfloat16(Bfloat16(second * cosine) + Bfloat16(first * sine));
          return x;
        };
        history->keys.push_back(rotate(key, "k_norm.weight"));
        history->values.push_back(value);
        std::vector<float> core(8);
        for (int head = 0; head < 2; ++head) {
          const auto query =
              rotate(std::vector<float>(q_gate.begin() + head * 8,
                                        q_gate.begin() + head * 8 + 4),
                     "q_norm.weight");
          std::vector<float> probability(history->keys.size());
          for (size_t t = 0; t < probability.size(); ++t)
            for (int d = 0; d < 4; ++d)
              probability[t] += query[d] * history->keys[t][d] / 2;
          const float maximum =
              *std::max_element(probability.begin(), probability.end());
          float total = 0;
          for (float& x : probability) {
            x = std::exp(x - maximum);
            total += x;
          }
          for (int d = 0; d < 4; ++d) {
            float sum = 0;
            for (size_t t = 0; t < probability.size(); ++t)
              sum += probability[t] / total * history->values[t][d];
            core[head * 4 + d] = Bfloat16(
                Bfloat16(sum) * Bfloat16(Sigmoid(q_gate[head * 8 + 4 + d])));
          }
        }
        attention = Project(p + "o_proj.weight", core);
      }
      for (int i = 0; i < 8; ++i)
        hidden[i] = Bfloat16(hidden[i] + attention[i]);
      const auto post =
          Normalize(hidden, prefix + "post_attention_layernorm.weight");
      auto gate = Project(prefix + "mlp.gate_proj.weight", post);
      const auto up = Project(prefix + "mlp.up_proj.weight", post);
      for (int i = 0; i < 8; ++i)
        gate[i] = Bfloat16(Bfloat16(Silu(gate[i])) * up[i]);
      const auto down = Project(prefix + "mlp.down_proj.weight", gate);
      for (int i = 0; i < 8; ++i)
        hidden[i] = Bfloat16(hidden[i] + down[i]);
    }
    return Project("lm_head.weight", Normalize(hidden, root + "norm.weight"));
  }

  void ExpectOneHotLogits(const std::vector<float>& logits, int token) {
    ASSERT_EQ(logits.size(), 4u);
    // sqrt(8) rounded to BF16 at the normalization boundary.
    for (int i = 0; i < 4; ++i)
      EXPECT_FLOAT_EQ(logits[i], i == token ? 2.828125f : 0.0f);
  }

  std::unique_ptr<cuda::Executor> executor_;
  std::filesystem::path directory_;
  std::map<std::string, SavedTensor> tensors_;
};

TEST_F(QwenModelTest, LoadsHybridFp8ModelAndProducesKnownLogits) {
  InferenceOptions options;
  options.context_length = 3;
  std::vector<std::pair<int, int>> progress;
  options.load_progress = [&](int loaded, int total) {
    progress.emplace_back(loaded, total);
  };
  auto model = Model::Load(*executor_, directory_, options);
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ(progress, (std::vector<std::pair<int, int>>{{1, 2}, {2, 2}}));
  EXPECT_EQ((*model)->config().layer_types,
            (std::vector<LayerType>{LayerType::kLinearAttention,
                                    LayerType::kFullAttention}));
  size_t expected_bytes = 0;
  for (const auto& [name, tensor] : tensors_)
    expected_bytes += tensor.device_bytes;
  EXPECT_EQ((*model)->weight_bytes(), expected_bytes);
  EXPECT_EQ((*model)->position(), 0);
  EXPECT_EQ((*model)->Logits().status().code(),
            absl::StatusCode::kFailedPrecondition);
  ASSERT_TRUE((*model)->Step(2).ok());
  EXPECT_EQ((*model)->position(), 1);
  ExpectOneHotLogits(CopyLogits(**model), 2);
  ExpectOneHotLogits(CopyLogits(**model), 2);
  EXPECT_EQ((*model)->position(), 1);
  ASSERT_TRUE((*model)->Step(1).ok());
  ExpectOneHotLogits(CopyLogits(**model), 1);
  EXPECT_EQ((*model)->position(), 2);
}

TEST_F(QwenModelTest, EnforcesCapacityAndResetRestoresBothCacheTypes) {
  InferenceOptions options;
  options.context_length = 3;
  auto model = Model::Load(*executor_, directory_, options);
  ASSERT_TRUE(model.ok()) << model.status();
  EXPECT_EQ((*model)->Step(-1).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ((*model)->Step(4).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ((*model)->position(), 0);
  for (int repetition = 0; repetition < 2; ++repetition) {
    for (int token = 0; token < 3; ++token) {
      ASSERT_TRUE((*model)->Step(token).ok());
      ExpectOneHotLogits(CopyLogits(**model), token);
    }
    EXPECT_EQ((*model)->position(), 3);
    EXPECT_EQ((*model)->Step(0).code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ((*model)->position(), 3);
    ASSERT_TRUE((*model)->Reset().ok());
    EXPECT_EQ((*model)->position(), 0);
    EXPECT_EQ((*model)->Logits().status().code(),
              absl::StatusCode::kFailedPrecondition);
  }
}

TEST_F(QwenModelTest, NonzeroHybridMatchesCpuAcrossHistoryAndReset) {
  MakeNonzeroHybrid();
  InferenceOptions options;
  options.context_length = 8;
  auto model = Model::Load(*executor_, directory_, options);
  ASSERT_TRUE(model.ok()) << model.status();
  const std::vector<int> tokens = {0, 1, 2, 0};
  std::vector<std::vector<float>> first_run;
  for (int repetition = 0; repetition < 2; ++repetition) {
    TinyHistory reference;
    for (size_t position = 0; position < tokens.size(); ++position) {
      SCOPED_TRACE(
          absl::StrCat("repetition=", repetition, " position=", position));
      const auto expected = ReferenceStep(tokens[position], &reference);
      ASSERT_TRUE((*model)->Step(tokens[position]).ok());
      const auto actual = CopyLogits(**model);
      ASSERT_EQ(actual.size(), expected.size());
      for (size_t token = 0; token < actual.size(); ++token)
        EXPECT_NEAR(actual[token], expected[token], 0.02f) << "logit " << token;
      // Repeated logits must not mutate either attention cache.
      EXPECT_EQ(CopyLogits(**model), actual);
      if (repetition == 0)
        first_run.push_back(actual);
      else
        EXPECT_EQ(actual, first_run[position]);
    }
    ASSERT_TRUE((*model)->Reset().ok());
    EXPECT_EQ((*model)->position(), 0);
  }
  // Token zero appears both fresh and after three earlier tokens. The fixture
  // must make history observable, so this cannot pass with identity residuals.
  float history_difference = 0;
  float branch_difference = 0;
  for (size_t token = 0; token < first_run.front().size(); ++token) {
    history_difference =
        std::max(history_difference,
                 std::abs(first_run.front()[token] - first_run.back()[token]));
    branch_difference = std::max(
        branch_difference,
        std::abs(first_run.front()[token] - (token == 0 ? 2.828125f : 0.0f)));
  }
  EXPECT_GT(history_difference, 0.05f);
  EXPECT_GT(branch_difference, 0.05f);
}

TEST_F(QwenModelTest, RunsNativeLayerGraphWithBalancedScopesAndTypedHooks) {
  MakeNonzeroHybrid();
  InferenceOptions options;
  options.context_length = 8;
  auto model = Model::Load(*executor_, directory_, options);
  ASSERT_TRUE(model.ok()) << model.status();
  LayerHooks hooks;
  std::vector<std::string> scopes;
  std::map<std::string, int> activations;
  int probability_calls = 0;
  hooks.enter_combinator = [&](cuda::Executor&, absl::string_view name) {
    scopes.emplace_back(name);
    return absl::OkStatus();
  };
  hooks.exit_combinator = [&](cuda::Executor&) {
    EXPECT_FALSE(scopes.empty());
    if (!scopes.empty())
      scopes.pop_back();
    return absl::OkStatus();
  };
  hooks.activation_hook = [&](cuda::Executor& executor, absl::string_view name,
                              absl::Span<const ActivationType> types,
                              absl::Span<Buffer> buffers) {
    EXPECT_EQ(&executor, executor_.get());
    EXPECT_EQ(types.size(), buffers.size());
    for (size_t i = 0; i < types.size(); ++i) {
      EXPECT_EQ(types[i].data_type(), DataType::BF16);
      EXPECT_EQ(types[i].dimensions()[0], ActivationType::kBatchDimension);
      EXPECT_EQ(types[i].dimensions()[1], 1);
      EXPECT_EQ(buffers[i].size_bytes(), size_t(types[i].dimensions()[2]) * 2);
    }
    ++activations[std::string(name)];
    return absl::OkStatus();
  };
  hooks.attention_probabilities_hook =
      [&](cuda::Executor&, absl::string_view name, const ActivationType& type,
          const Buffer& probabilities) {
        EXPECT_EQ(name, "QwenAttentionLayer");
        EXPECT_EQ(type, ActivationType(DataType::FP32, {1, 2, 1, 1}));
        EXPECT_EQ(probabilities.size_bytes(), 2 * sizeof(float));
        EXPECT_NE(std::find(scopes.begin(), scopes.end(), "QwenBlock1"),
                  scopes.end());
        ++probability_calls;
        return absl::OkStatus();
      };
  ASSERT_TRUE((*model)->Step(0, &hooks).ok());
  ASSERT_TRUE((*model)->Logits(&hooks).ok());
  EXPECT_TRUE(scopes.empty());
  EXPECT_EQ(activations["QwenDecoder"], 1);
  EXPECT_EQ(activations["QwenBlock0"], 1);
  EXPECT_EQ(activations["QwenBlock1"], 1);
  EXPECT_EQ(activations["ResidualLayer"], 4);
  EXPECT_EQ(activations["DeltaNetLayer"], 1);
  EXPECT_EQ(activations["QwenAttentionLayer"], 1);
  EXPECT_EQ(activations["EmbeddingLookupLayer"], 1);
  EXPECT_EQ(activations["FullyConnectedLayer"], 16);
  EXPECT_EQ(activations["SwiGluLayer"], 2);
  EXPECT_EQ(activations["RmsNormLayer"], 5);
  EXPECT_EQ(activations["LanguageModelingHead"], 1);
  EXPECT_EQ(probability_calls, 1);
  TinyHistory reference;
  auto expected = ReferenceStep(0, &reference);
  auto actual = CopyLogits(**model);
  for (size_t i = 0; i < expected.size(); ++i)
    EXPECT_NEAR(actual[i], expected[i], 0.02);
}

TEST_F(QwenModelTest, ActivationInterventionsAffectDownstreamLayers) {
  InferenceOptions options;
  options.context_length = 2;
  auto model = Model::Load(*executor_, directory_, options);
  ASSERT_TRUE(model.ok()) << model.status();
  LayerHooks hooks;
  hooks.activation_hook = [&](cuda::Executor& executor, absl::string_view name,
                              absl::Span<const ActivationType>,
                              absl::Span<Buffer> buffers) -> absl::Status {
    if (name != "EmbeddingLookupLayer")
      return absl::OkStatus();
    ASSIGN_OR_RETURN(auto zeros,
                     Buffer::Allocate(executor, buffers[0].size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemsetAsync(zeros.data(), 0, zeros.size_bytes(), executor.stream()),
        "zero replacement embedding"));
    buffers[0] = std::move(zeros);
    return absl::OkStatus();
  };
  ASSERT_TRUE((*model)->Step(2, &hooks).ok());
  EXPECT_EQ(CopyLogits(**model), std::vector<float>(4, 0));
  ASSERT_TRUE((*model)->Reset().ok());
  ASSERT_TRUE((*model)->Step(2).ok());
  ExpectOneHotLogits(CopyLogits(**model), 2);
}

TEST_F(QwenModelTest, FailedHooksRequireResetAndDoNotPublishMixedHistory) {
  MakeNonzeroHybrid();
  InferenceOptions options;
  options.context_length = 8;
  auto model = Model::Load(*executor_, directory_, options);
  ASSERT_TRUE(model.ok()) << model.status();
  LayerHooks hooks;
  int depth = 0;
  hooks.enter_combinator = [&](cuda::Executor&, absl::string_view) {
    ++depth;
    return absl::OkStatus();
  };
  hooks.exit_combinator = [&](cuda::Executor&) {
    --depth;
    return absl::OkStatus();
  };
  hooks.activation_hook = [&](cuda::Executor&, absl::string_view name,
                              absl::Span<const ActivationType>,
                              absl::Span<Buffer>) {
    // The DeltaNet cache has advanced, but full attention has not.
    return name == "DeltaNetLayer"
               ? absl::AbortedError("intentional hook failure")
               : absl::OkStatus();
  };
  EXPECT_EQ((*model)->Step(0, &hooks).code(), absl::StatusCode::kAborted);
  EXPECT_EQ(depth, 0);
  EXPECT_EQ((*model)->position(), 0);
  EXPECT_EQ((*model)->Step(1).code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ((*model)->Logits().status().code(),
            absl::StatusCode::kFailedPrecondition);
  ASSERT_TRUE((*model)->Reset().ok());
  ASSERT_TRUE((*model)->Step(1).ok());
  TinyHistory reference;
  const auto expected = ReferenceStep(1, &reference);
  const auto actual = CopyLogits(**model);
  for (size_t i = 0; i < actual.size(); ++i)
    EXPECT_NEAR(actual[i], expected[i], 0.02);
}

TEST_F(QwenModelTest, RejectsInvalidContextBeforeUploadingWeights) {
  for (int context : {-1, 0, 9}) {
    InferenceOptions options;
    options.context_length = context;
    EXPECT_EQ(Model::Load(*executor_, directory_, options).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST_F(QwenModelTest,
       RejectsNonBFloat16EmbeddingInsteadOfChangingItsPrecision) {
  Add("model.language_model.embed_tokens.weight", {4, 8}, "F32",
      std::vector<float>(32, 1.001f), true);
  WriteCheckpoint();
  InferenceOptions options;
  options.context_length = 8;
  EXPECT_EQ(Model::Load(*executor_, directory_, options).status().code(),
            absl::StatusCode::kUnimplemented);
}

TEST_F(QwenModelTest, RejectsMissingFp8Scale) {
  tensors_.erase(
      "model.language_model.layers.0.mlp.gate_proj.weight_scale_inv");
  WriteCheckpoint();
  InferenceOptions options;
  options.context_length = 3;
  auto model = Model::Load(*executor_, directory_, options);
  ASSERT_FALSE(model.ok());
  EXPECT_EQ(model.status().code(), absl::StatusCode::kNotFound);
}

TEST_F(QwenModelTest, RejectsNonpositiveFp8Scale) {
  InferenceOptions options;
  options.context_length = 3;
  for (float scale : {0.0f, -1.0f}) {
    SCOPED_TRACE(scale);
    Constant("model.language_model.layers.0.mlp.gate_proj.weight_scale_inv",
             {1, 1}, scale);
    WriteCheckpoint();
    auto model = Model::Load(*executor_, directory_, options);
    ASSERT_FALSE(model.ok());
    EXPECT_EQ(model.status().code(), absl::StatusCode::kInvalidArgument);
  }
}

TEST_F(QwenModelTest, RejectsShapeMismatch) {
  // Same byte count keeps the file valid while violating the model contract.
  tensors_.at("lm_head.weight").shape = {8, 4};
  WriteCheckpoint();
  InferenceOptions options;
  options.context_length = 3;
  auto model = Model::Load(*executor_, directory_, options);
  ASSERT_FALSE(model.ok());
  EXPECT_EQ(model.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(QwenModelTest, RejectsNonFiniteNormalizationWeight) {
  Add("model.language_model.norm.weight", {8}, "F32",
      std::vector<float>(8, std::numeric_limits<float>::infinity()), false);
  WriteCheckpoint();
  InferenceOptions options;
  options.context_length = 3;
  auto model = Model::Load(*executor_, directory_, options);
  ASSERT_FALSE(model.ok());
  EXPECT_EQ(model.status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace pluto::llm::qwen
