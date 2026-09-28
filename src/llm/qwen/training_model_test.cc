#include "src/llm/qwen/training_model.h"

#include <cuda_runtime_api.h>

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
#include <vector>

#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/llm/qwen/model.h"
#include "src/util/status_macros.h"

namespace pluto::llm::qwen {
namespace {

// A nonzero, tiny hybrid decoder exercises both temporal backward operators
// without downloading a checkpoint or consuming production-model memory.
constexpr char kConfig[] = R"({
  "model_type":"qwen3_5", "text_config": {
    "model_type":"qwen3_5_text", "dtype":"bfloat16", "hidden_act":"silu",
    "hidden_size":8, "intermediate_size":8, "num_hidden_layers":2,
    "num_attention_heads":1, "num_key_value_heads":1, "head_dim":8,
    "vocab_size":16, "max_position_embeddings":8,
    "linear_conv_kernel_dim":2, "linear_key_head_dim":4,
    "linear_num_key_heads":1, "linear_num_value_heads":1,
    "linear_value_head_dim":4, "full_attention_interval":2,
    "bos_token_id":0, "eos_token_id":15, "rms_norm_eps":0.000001,
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

struct Tensor {
  std::vector<int64_t> shape;
  std::string bytes;
};

class TrainingModelTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
    std::string pattern = testing::TempDir() + "/qwen-training-XXXXXX";
    char* path = mkdtemp(pattern.data());
    ASSERT_NE(path, nullptr);
    directory_ = path;
    std::ofstream(directory_ / "config.json") << kConfig;
    Add("model.language_model.embed_tokens.weight", {16, 8});
    Add("lm_head.weight", {16, 8});
    Add("model.language_model.norm.weight", {8}, 0);
    for (int layer = 0; layer < 2; ++layer) {
      const std::string root =
          absl::StrCat("model.language_model.layers.", layer, ".");
      Add(root + "input_layernorm.weight", {8}, 0);
      Add(root + "post_attention_layernorm.weight", {8}, 0);
      Add(root + "mlp.gate_proj.weight", {8, 8});
      Add(root + "mlp.up_proj.weight", {8, 8});
      Add(root + "mlp.down_proj.weight", {8, 8});
      if (layer == 0) {
        Add(root + "linear_attn.in_proj_qkv.weight", {12, 8});
        Add(root + "linear_attn.in_proj_z.weight", {4, 8});
        Add(root + "linear_attn.in_proj_a.weight", {1, 8});
        Add(root + "linear_attn.in_proj_b.weight", {1, 8});
        Add(root + "linear_attn.out_proj.weight", {8, 4});
        Add(root + "linear_attn.conv1d.weight", {12, 1, 2}, .5);
        Add(root + "linear_attn.A_log", {1}, -1);
        Add(root + "linear_attn.dt_bias", {1}, .25);
        Add(root + "linear_attn.norm.weight", {4}, 1);
      } else {
        Add(root + "self_attn.q_proj.weight", {16, 8});
        Add(root + "self_attn.k_proj.weight", {8, 8});
        Add(root + "self_attn.v_proj.weight", {8, 8});
        Add(root + "self_attn.o_proj.weight", {8, 8});
        Add(root + "self_attn.q_norm.weight", {8}, .125);
        Add(root + "self_attn.k_norm.weight", {8}, -.125);
      }
    }
    WriteCheckpoint();
  }

  void TearDown() override {
    EXPECT_TRUE(executor_->Synchronize().ok());
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
    EXPECT_FALSE(error) << error.message();
  }

  // Omitted constant means a deterministic, nonzero matrix. Store BF16 so
  // training and cached inference use the same imported values, not FP8's
  // additional inference-only dynamic activation quantization.
  void Add(const std::string& name, std::vector<int64_t> shape,
           float constant = std::numeric_limits<float>::quiet_NaN()) {
    size_t size = 1;
    for (auto dimension : shape)
      size *= dimension;
    Tensor tensor{std::move(shape), {}};
    for (size_t i = 0; i < size; ++i) {
      float value = std::isnan(constant)
                        ? .18f * std::sin(.41f * i + .31f * tensors_.size())
                        : constant;
      uint32_t bits = std::bit_cast<uint32_t>(value);
      bits += 0x7fff + ((bits >> 16) & 1);
      tensor.bytes.push_back((bits >> 16) & 255);
      tensor.bytes.push_back((bits >> 24) & 255);
    }
    tensors_.emplace(name, std::move(tensor));
  }

  void WriteCheckpoint() {
    std::string header = "{", payload;
    for (const auto& [name, tensor] : tensors_) {
      if (header.size() > 1)
        header += ",";
      absl::StrAppend(&header, "\"", name,
                      "\":{\"dtype\":\"BF16\",\"shape\":[");
      for (size_t i = 0; i < tensor.shape.size(); ++i) {
        if (i)
          header += ",";
        absl::StrAppend(&header, tensor.shape[i]);
      }
      absl::StrAppend(&header, "],\"data_offsets\":[", payload.size(), ",",
                      payload.size() + tensor.bytes.size(), "]}");
      payload += tensor.bytes;
    }
    header += "}";
    header.append((8 - header.size() % 8) % 8, ' ');
    std::ofstream output(directory_ / "model.safetensors", std::ios::binary);
    uint64_t header_size = header.size();
    for (int i = 0; i < 8; ++i)
      output.put((header_size >> (i * 8)) & 255);
    output.write(header.data(), header.size());
    output.write(payload.data(), payload.size());
    ASSERT_TRUE(output.good());
  }

  template <class T>
  absl::StatusOr<Buffer> Upload(const std::vector<T>& values) {
    ASSIGN_OR_RETURN(auto host,
                     cuda::PageLockedHostArray<T>::CopyFrom(*executor_, values));
    ASSIGN_OR_RETURN(auto result,
                     Buffer::Allocate(*executor_, values.size() * sizeof(T)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(result.data(), host.data(), result.size_bytes(),
                        cudaMemcpyHostToDevice, executor_->stream()),
        "upload training model test"));
    RETURN_IF_ERROR(executor_->Synchronize());
    return result;
  }

  template <class T>
  std::vector<T> Read(const Buffer& buffer) {
    auto host = cuda::PageLockedHostArray<T>::Allocate(
        *executor_, buffer.size_bytes() / sizeof(T));
    EXPECT_TRUE(host.ok()) << host.status();
    if (!host.ok())
      return {};
    EXPECT_EQ(cudaMemcpyAsync(host->data(), buffer.data(), buffer.size_bytes(),
                              cudaMemcpyDeviceToHost, executor_->stream()),
              cudaSuccess);
    EXPECT_TRUE(executor_->Synchronize().ok());
    return std::vector<T>(host->begin(), host->end());
  }

  using Snapshot = std::vector<std::vector<std::vector<uint8_t>>>;
  Snapshot Weights(const TrainingModel& model) {
    Snapshot result;
    for (const auto& block : model.parameter_blocks()) {
      result.emplace_back();
      for (const auto& parameter : block.parameters)
        result.back().push_back(Read<uint8_t>(parameter->value()));
    }
    return result;
  }

  std::unique_ptr<cuda::Executor> executor_;
  std::filesystem::path directory_;
  std::map<std::string, Tensor> tensors_;
};

TEST_F(TrainingModelTest, HybridForwardMatchesCachedInference) {
  TrainingModelOptions options;
  options.sequence_length = 3;
  int progress_calls = 0;
  options.load_progress = [&](int loaded, int total) {
    EXPECT_EQ(total, 2);
    EXPECT_EQ(loaded, ++progress_calls);
  };
  auto train = TrainingModel::Load(*executor_, directory_, options);
  ASSERT_TRUE(train.ok()) << train.status();
  EXPECT_EQ(progress_calls, 2);
  EXPECT_EQ((*train)->input_types()[0],
            ActivationType(DataType::INT32, {-2, 3}));
  EXPECT_EQ((*train)->output_types()[0],
            ActivationType(DataType::FP32, {-2, 3, 16}));
  auto inference = Model::Load(*executor_, directory_, {.context_length = 3});
  ASSERT_TRUE(inference.ok()) << inference.status();
  const std::vector<int32_t> tokens{0, 1, 2};
  auto input = Upload(tokens);
  ASSERT_TRUE(input.ok());
  auto result = (*train)->fwd(*executor_, {*input});
  ASSERT_TRUE(result.ok()) << result.status();
  auto logits = Read<float>(result->outputs[0]);
  for (int t = 0; t < 3; ++t) {
    ASSERT_TRUE((*inference)->Step(tokens[t]).ok());
    auto output = (*inference)->Logits();
    ASSERT_TRUE(output.ok());
    auto expected = Read<float>(*output);
    for (int v = 0; v < 16; ++v) {
      EXPECT_TRUE(std::isfinite(logits[t * 16 + v]));
      // Training's head produces FP32 logits; inference rounds its head
      // output to BF16. Permit one BF16 ulp plus reduction-order rounding.
      EXPECT_NEAR(logits[t * 16 + v], expected[v], .01f);
    }
  }
}

TEST_F(TrainingModelTest, BAdamCyclesAllBlocksWithoutChangingFrozenWeights) {
  auto model =
      TrainingModel::Load(*executor_, directory_, {.sequence_length = 3});
  ASSERT_TRUE(model.ok()) << model.status();
  ASSERT_EQ((*model)->parameter_blocks().size(), 4);
  auto inputs = Upload<int32_t>({0, 1, 2});
  auto targets = Upload<int32_t>({1, 2, 3});
  ASSERT_TRUE(inputs.ok());
  ASSERT_TRUE(targets.ok());
  auto loss = CrossEntropyLossLayer::Create(*executor_, 16, DataType::BF16, 3);
  ASSERT_TRUE(loss.ok());
  BAdamConfig config;
  config.switch_every = 1;
  config.adam.learning_rate = .01f;
  config.adam.weight_decay = 0;
  auto optimizer =
      BAdamOptimizer::Create(*executor_, (*model)->parameter_blocks(), config);
  ASSERT_TRUE(optimizer.ok()) << optimizer.status();
  double initial_loss = 0, last_loss = 0;
  std::vector<bool> visited(4, false);
  for (int step = 0; step < 12; ++step) {
    ASSERT_TRUE((*optimizer)->ZeroGrad().ok());
    const int active = (*optimizer)->active_block();
    EXPECT_EQ(active, 3 - step % 4);
    visited[active] = true;
    ASSERT_TRUE((*model)->SetBackwardStart(active).ok());
    for (int block = 0; block < 4; ++block)
      for (const auto& parameter :
           (*model)->parameter_blocks()[block].parameters)
        EXPECT_EQ(parameter->active(), block == active);
    auto before = Weights(**model);
    auto result = (*model)->fwd(*executor_, {*inputs});
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(result->state.children.size(), 4 - active);
    auto loss_result = (*loss)->fwd(*executor_, {result->outputs[0], *targets});
    ASSERT_TRUE(loss_result.ok()) << loss_result.status();
    auto losses = Read<float>(loss_result->outputs[0]);
    last_loss = 0;
    for (float value : losses) {
      EXPECT_TRUE(std::isfinite(value));
      last_loss += value / 3.;
    }
    if (step == 0)
      initial_loss = last_loss;
    auto gradients =
        (*loss)->bwd(*executor_, {}, std::move(loss_result->state));
    ASSERT_TRUE(gradients.ok()) << gradients.status();
    auto backward =
        (*model)->bwd(*executor_, *gradients, std::move(result->state));
    ASSERT_TRUE(backward.ok()) << backward.status();
    EXPECT_TRUE(backward->empty());
    ASSERT_TRUE((*optimizer)->ApplyStep().ok());
    auto after = Weights(**model);
    for (int block = 0; block < 4; ++block)
      if (block == active)
        EXPECT_NE(before[block], after[block]) << "active block " << block;
      else
        EXPECT_EQ(before[block], after[block]) << "frozen block " << block;
  }
  EXPECT_LT(last_loss, initial_loss);
  EXPECT_EQ((*optimizer)->step(), 12);
  for (bool seen : visited)
    EXPECT_TRUE(seen);
}

TEST_F(TrainingModelTest, RejectsInvalidSequenceBoundaryAndExecutor) {
  for (int sequence : {0, -1, 9, 129})
    EXPECT_FALSE(TrainingModel::Load(*executor_, directory_,
                                     {.sequence_length = sequence})
                     .ok());
  auto model =
      TrainingModel::Load(*executor_, directory_, {.sequence_length = 3});
  ASSERT_TRUE(model.ok());
  EXPECT_FALSE((*model)->SetBackwardStart(-1).ok());
  EXPECT_FALSE((*model)->SetBackwardStart(4).ok());
  auto short_input = Upload<int32_t>({0, 1});
  ASSERT_TRUE(short_input.ok());
  EXPECT_FALSE((*model)->fwd(*executor_, {*short_input}).ok());
  auto input = Upload<int32_t>({0, 1, 2});
  ASSERT_TRUE(input.ok());
  auto other = cuda::Executor::Create();
  ASSERT_TRUE(other.ok());
  EXPECT_FALSE((*model)->fwd(**other, {*input}).ok());
}

}  // namespace
}  // namespace pluto::llm::qwen
