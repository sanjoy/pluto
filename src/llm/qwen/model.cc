#include "src/llm/qwen/model.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/qwen/attention_ops.h"
#include "src/llm/qwen/ops.h"
#include "src/util/status_macros.h"

namespace pluto::llm::qwen {
namespace {

using cuda::Buffer;

struct DeviceTensor {
  // Raw matrix storage (BF16/FP8), or FP32 storage for small scalar tensors.
  Buffer data;
  // FP32 multiplicative scales for FP8 matrices; absent for other storage.
  std::optional<Buffer> scales;
  MatrixStorage storage;
  int rows;     // Matrix output width; zero for non-matrix tensors.
  int columns;  // Matrix input width; zero for non-matrix tensors.
  const float* floats() const { return static_cast<const float*>(data.data()); }
};

// One shared, bounded staging allocation avoids page-locking an entire 31 GB
// checkpoint. Synchronizing before reusing it is necessary: unlike freeing a
// stream-ordered allocation, overwriting host memory is not stream ordered.
class Uploader {
 public:
  Uploader(cuda::Executor& executor, cuda::PageLockedHostArray<uint8_t> staging)
      : executor_(executor), staging_(std::move(staging)) {}

  absl::StatusOr<Buffer> Copy(absl::Span<const uint8_t> bytes) {
    ASSIGN_OR_RETURN(auto result, Buffer::Allocate(executor_, bytes.size()));
    for (size_t offset = 0; offset < bytes.size(); offset += staging_.size()) {
      size_t count = std::min(staging_.size(), bytes.size() - offset);
      std::memcpy(staging_.data(), bytes.data() + offset, count);
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpyAsync(static_cast<uint8_t*>(result.data()) + offset,
                          staging_.data(), count, cudaMemcpyHostToDevice,
                          executor_.stream()),
          "upload Qwen tensor"));
      RETURN_IF_ERROR(executor_.Synchronize());
    }
    weight_bytes += bytes.size();
    return result;
  }

  absl::StatusOr<Buffer> FloatTensor(const TensorView& tensor,
                                     bool positive_scale = false) {
    if (tensor.dtype == TensorDType::kF8E4M3)
      return absl::InvalidArgumentError(
          "small tensors/scales must be BF16 or FP32");
    size_t count =
        tensor.bytes.size() / (tensor.dtype == TensorDType::kBF16 ? 2 : 4);
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
      if (tensor.dtype == TensorDType::kBF16) {
        uint32_t bits = (uint32_t(tensor.bytes[2 * i]) |
                         (uint32_t(tensor.bytes[2 * i + 1]) << 8))
                        << 16;
        values[i] = std::bit_cast<float>(bits);
      } else {
        std::memcpy(&values[i], tensor.bytes.data() + 4 * i, 4);
      }
      if (!std::isfinite(values[i]))
        return absl::InvalidArgumentError(
            "non-finite Qwen scalar weight or scale");
      if (positive_scale && values[i] <= 0)
        return absl::InvalidArgumentError("FP8 block scales must be positive");
    }
    return Copy(absl::Span<const uint8_t>(
        reinterpret_cast<const uint8_t*>(values.data()),
        count * sizeof(float)));
  }

  size_t weight_bytes = 0;

 private:
  cuda::Executor& executor_;
  cuda::PageLockedHostArray<uint8_t> staging_;
};

absl::Status Shape(const TensorView& tensor, absl::Span<const int64_t> expected,
                   const std::string& name) {
  if (!std::equal(tensor.shape.begin(), tensor.shape.end(), expected.begin(),
                  expected.end()))
    return absl::InvalidArgumentError(
        absl::StrCat("unexpected checkpoint shape: ", name));
  return absl::OkStatus();
}

absl::StatusOr<DeviceTensor> LoadTensor(SafetensorsCheckpoint& checkpoint,
                                        Uploader& uploader,
                                        const std::string& name,
                                        absl::Span<const int64_t> shape,
                                        bool matrix) {
  ASSIGN_OR_RETURN(auto tensor, checkpoint.Tensor(name));
  RETURN_IF_ERROR(Shape(tensor, shape, name));
  if (!matrix) {
    ASSIGN_OR_RETURN(auto data, uploader.FloatTensor(tensor));
    return DeviceTensor{std::move(data), std::nullopt, MatrixStorage::kFloat32,
                        0, 0};
  }
  MatrixStorage storage =
      tensor.dtype == TensorDType::kBF16     ? MatrixStorage::kBFloat16
      : tensor.dtype == TensorDType::kF8E4M3 ? MatrixStorage::kFp8E4M3
                                             : MatrixStorage::kFloat32;
  ASSIGN_OR_RETURN(auto data, uploader.Copy(tensor.bytes));
  DeviceTensor result{std::move(data), std::nullopt, storage,
                      static_cast<int>(shape[0]), static_cast<int>(shape[1])};
  if (storage == MatrixStorage::kFp8E4M3) {
    const std::string scale_name =
        name.substr(0, name.size() - 7) + ".weight_scale_inv";
    ASSIGN_OR_RETURN(auto scale, checkpoint.Tensor(scale_name));
    RETURN_IF_ERROR(Shape(
        scale, {(shape[0] + 127) / 128, (shape[1] + 127) / 128}, scale_name));
    ASSIGN_OR_RETURN(auto scales, uploader.FloatTensor(scale, true));
    result.scales = std::move(scales);
  }
  return result;
}

// Loaded tensors and exactly one stateful attention implementation per block.
struct Block {
  absl::flat_hash_map<std::string, DeviceTensor> weights;
  std::unique_ptr<FullAttentionState> full_attention;
  std::unique_ptr<DeltaNetState> delta;
};

}  // namespace

struct Model::Impl {
  Impl(cuda::Executor& executor, Config config, int capacity)
      : executor(executor), config(std::move(config)), capacity(capacity) {}
  cuda::Executor& executor;
  Config config;
  int capacity;
  int position = 0;
  bool poisoned = false;
  size_t weight_bytes = 0;
  absl::flat_hash_map<std::string, DeviceTensor> outer;
  std::vector<Block> blocks;
  // Reused single-token scratch: hidden, normalized, quantized, then six
  // projection slots. Each slot fits the largest non-vocabulary projection.
  std::vector<Buffer> scratch;
  float* S(int slot) { return static_cast<float*>(scratch[slot].data()); }

  absl::Status Project(const DeviceTensor& weight, const float* input,
                       float* output) {
    if (weight.storage == MatrixStorage::kFp8E4M3) {
      RETURN_IF_ERROR(QuantizeFp8Input(executor, input, weight.columns, S(2)));
      input = S(2);
    }
    return MatVec(executor, weight.data.data(), weight.storage,
                  weight.scales
                      ? static_cast<const float*>(weight.scales->data())
                      : nullptr,
                  input, weight.columns, weight.rows, output);
  }
};

Model::Model(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Model::~Model() = default;

absl::StatusOr<std::unique_ptr<Model>> Model::Load(
    cuda::Executor& executor, const std::filesystem::path& directory,
    const InferenceOptions& options) {
  ASSIGN_OR_RETURN(auto config, LoadConfig(directory));
  if (options.context_length <= 0 ||
      options.context_length > config.max_position_embeddings)
    return absl::InvalidArgumentError(
        "context_length exceeds checkpoint limit or is nonpositive");
  if (config.hidden_size > 16384 || config.head_dim > 256 ||
      config.linear_key_head_dim > 128 || config.linear_value_head_dim > 256)
    return absl::UnimplementedError(
        "Qwen dimensions exceed current operator tile limits");
  ASSIGN_OR_RETURN(auto checkpoint, SafetensorsCheckpoint::Open(directory));
  ASSIGN_OR_RETURN(auto staging, cuda::PageLockedHostArray<uint8_t>::Allocate(
                                     executor, 32 * 1024 * 1024));
  Uploader uploader(executor, std::move(staging));
  auto impl = absl::make_unique<Impl>(executor, config, options.context_length);
  const std::string root = "model.language_model.";
  const int d = config.hidden_size;
  const int f = config.intermediate_size;
  const int q = config.num_attention_heads * config.head_dim;
  const int kv = config.num_key_value_heads * config.head_dim;
  const int lk = config.linear_num_key_heads * config.linear_key_head_dim;
  const int lv = config.linear_num_value_heads * config.linear_value_head_dim;
  if (std::max({d, f, 2 * q, kv, 2 * lk + lv, config.vocab_size}) > 1048576 ||
      config.linear_conv_kernel_dim > 32)
    return absl::UnimplementedError(
        "Qwen dimensions exceed operator indexing limits");
  auto load_outer = [&](const std::string& key, const std::string& name,
                        absl::Span<const int64_t> shape,
                        bool matrix) -> absl::Status {
    ASSIGN_OR_RETURN(auto weight,
                     LoadTensor(*checkpoint, uploader, name, shape, matrix));
    impl->outer.emplace(key, std::move(weight));
    return absl::OkStatus();
  };
  RETURN_IF_ERROR(load_outer("embedding", root + "embed_tokens.weight",
                             {config.vocab_size, d}, true));
  RETURN_IF_ERROR(
      load_outer("head", "lm_head.weight", {config.vocab_size, d}, true));
  RETURN_IF_ERROR(load_outer("norm", root + "norm.weight", {d}, false));
  if (impl->outer.at("embedding").storage == MatrixStorage::kFp8E4M3)
    return absl::UnimplementedError("Qwen token embeddings must not be FP8");
  for (int layer = 0; layer < config.num_hidden_layers; ++layer) {
    Block block;
    std::string prefix = absl::StrCat(root, "layers.", layer, ".");
    auto load = [&](const std::string& name, absl::Span<const int64_t> shape,
                    bool matrix = true) -> absl::Status {
      ASSIGN_OR_RETURN(auto weight, LoadTensor(*checkpoint, uploader,
                                               prefix + name, shape, matrix));
      block.weights.emplace(name, std::move(weight));
      return absl::OkStatus();
    };
    RETURN_IF_ERROR(load("input_layernorm.weight", {d}, false));
    RETURN_IF_ERROR(load("post_attention_layernorm.weight", {d}, false));
    RETURN_IF_ERROR(load("mlp.gate_proj.weight", {f, d}));
    RETURN_IF_ERROR(load("mlp.up_proj.weight", {f, d}));
    RETURN_IF_ERROR(load("mlp.down_proj.weight", {d, f}));
    if (config.layer_types[layer] == LayerType::kFullAttention) {
      RETURN_IF_ERROR(load("self_attn.q_proj.weight", {2 * q, d}));
      RETURN_IF_ERROR(load("self_attn.k_proj.weight", {kv, d}));
      RETURN_IF_ERROR(load("self_attn.v_proj.weight", {kv, d}));
      RETURN_IF_ERROR(load("self_attn.o_proj.weight", {d, q}));
      RETURN_IF_ERROR(
          load("self_attn.q_norm.weight", {config.head_dim}, false));
      RETURN_IF_ERROR(
          load("self_attn.k_norm.weight", {config.head_dim}, false));
      FullAttentionParameters p;
      p.query_heads = config.num_attention_heads;
      p.key_value_heads = config.num_key_value_heads;
      p.head_dim = config.head_dim;
      p.rotary_dim =
          static_cast<int>(config.head_dim * config.partial_rotary_factor);
      p.capacity = options.context_length;
      p.rms_norm_epsilon = config.rms_norm_eps;
      p.rope_theta = config.rope_theta;
      ASSIGN_OR_RETURN(block.full_attention,
                       FullAttentionState::Create(executor, p));
    } else {
      RETURN_IF_ERROR(load("linear_attn.in_proj_qkv.weight", {2 * lk + lv, d}));
      RETURN_IF_ERROR(load("linear_attn.in_proj_z.weight", {lv, d}));
      RETURN_IF_ERROR(load("linear_attn.in_proj_a.weight",
                           {config.linear_num_value_heads, d}));
      RETURN_IF_ERROR(load("linear_attn.in_proj_b.weight",
                           {config.linear_num_value_heads, d}));
      RETURN_IF_ERROR(load("linear_attn.out_proj.weight", {d, lv}));
      RETURN_IF_ERROR(load("linear_attn.conv1d.weight",
                           {2 * lk + lv, 1, config.linear_conv_kernel_dim},
                           false));
      RETURN_IF_ERROR(
          load("linear_attn.A_log", {config.linear_num_value_heads}, false));
      RETURN_IF_ERROR(
          load("linear_attn.dt_bias", {config.linear_num_value_heads}, false));
      RETURN_IF_ERROR(load("linear_attn.norm.weight",
                           {config.linear_value_head_dim}, false));
      DeltaNetParameters p;
      p.key_heads = config.linear_num_key_heads;
      p.value_heads = config.linear_num_value_heads;
      p.key_head_dim = config.linear_key_head_dim;
      p.value_head_dim = config.linear_value_head_dim;
      p.conv_kernel_dim = config.linear_conv_kernel_dim;
      p.rms_norm_epsilon = config.rms_norm_eps;
      ASSIGN_OR_RETURN(block.delta, DeltaNetState::Create(executor, p));
    }
    impl->blocks.push_back(std::move(block));
    if (options.load_progress)
      options.load_progress(layer + 1, config.num_hidden_layers);
  }
  impl->weight_bytes = uploader.weight_bytes;
  const int max_width =
      std::max({d, f, 2 * q, 2 * lk + lv, config.linear_num_value_heads});
  for (int i = 0; i < 9; ++i) {
    ASSIGN_OR_RETURN(auto buffer, Buffer::Allocate(executor, size_t(max_width) *
                                                                 sizeof(float)));
    impl->scratch.push_back(std::move(buffer));
  }
  RETURN_IF_ERROR(executor.Synchronize());
  return absl::WrapUnique(new Model(std::move(impl)));
}

absl::Status Model::Step(int token) {
  auto& m = *impl_;
  const auto& c = m.config;
  if (m.poisoned)
    return absl::FailedPreconditionError(
        "Reset Qwen caches after a failed Step");
  if (token < 0 || token >= c.vocab_size || m.position >= m.capacity)
    return absl::InvalidArgumentError(
        "invalid Qwen token or context capacity exhausted");
  m.poisoned = true;
  const auto& embedding = m.outer.at("embedding");
  RETURN_IF_ERROR(EmbeddingLookup(m.executor, embedding.data.data(),
                                  embedding.storage, token, c.vocab_size,
                                  c.hidden_size, m.S(0)));
  for (auto& block : m.blocks) {
    auto scalar = [&](const char* name) {
      return block.weights.at(name).floats();
    };
    auto project = [&](const char* name, const float* input, int output) {
      return m.Project(block.weights.at(name), input, m.S(output));
    };
    RETURN_IF_ERROR(RmsNorm(m.executor, m.S(0),
                            scalar("input_layernorm.weight"), c.hidden_size,
                            c.rms_norm_eps, m.S(1)));
    if (block.full_attention) {
      RETURN_IF_ERROR(project("self_attn.q_proj.weight", m.S(1), 3));
      RETURN_IF_ERROR(project("self_attn.k_proj.weight", m.S(1), 4));
      RETURN_IF_ERROR(project("self_attn.v_proj.weight", m.S(1), 5));
      RETURN_IF_ERROR(block.full_attention->Step(
          m.S(3), m.S(4), m.S(5), scalar("self_attn.q_norm.weight"),
          scalar("self_attn.k_norm.weight"), m.S(7)));
      RETURN_IF_ERROR(project("self_attn.o_proj.weight", m.S(7), 8));
    } else {
      RETURN_IF_ERROR(project("linear_attn.in_proj_qkv.weight", m.S(1), 3));
      RETURN_IF_ERROR(project("linear_attn.in_proj_z.weight", m.S(1), 4));
      RETURN_IF_ERROR(project("linear_attn.in_proj_a.weight", m.S(1), 5));
      RETURN_IF_ERROR(project("linear_attn.in_proj_b.weight", m.S(1), 6));
      RETURN_IF_ERROR(block.delta->Step(
          m.S(3), m.S(4), m.S(5), m.S(6), scalar("linear_attn.conv1d.weight"),
          scalar("linear_attn.A_log"), scalar("linear_attn.dt_bias"),
          scalar("linear_attn.norm.weight"), m.S(7)));
      RETURN_IF_ERROR(project("linear_attn.out_proj.weight", m.S(7), 8));
    }
    RETURN_IF_ERROR(
        ResidualAdd(m.executor, m.S(0), m.S(8), c.hidden_size, m.S(0)));
    RETURN_IF_ERROR(RmsNorm(m.executor, m.S(0),
                            scalar("post_attention_layernorm.weight"),
                            c.hidden_size, c.rms_norm_eps, m.S(1)));
    RETURN_IF_ERROR(project("mlp.gate_proj.weight", m.S(1), 3));
    RETURN_IF_ERROR(project("mlp.up_proj.weight", m.S(1), 4));
    RETURN_IF_ERROR(
        SwiGlu(m.executor, m.S(3), m.S(4), c.intermediate_size, m.S(5)));
    RETURN_IF_ERROR(project("mlp.down_proj.weight", m.S(5), 8));
    RETURN_IF_ERROR(
        ResidualAdd(m.executor, m.S(0), m.S(8), c.hidden_size, m.S(0)));
  }
  ++m.position;
  m.poisoned = false;
  return absl::OkStatus();
}

absl::StatusOr<Buffer> Model::Logits() {
  auto& m = *impl_;
  if (m.position == 0 || m.poisoned)
    return absl::FailedPreconditionError(
        "Qwen logits require a successful Step");
  RETURN_IF_ERROR(RmsNorm(m.executor, m.S(0), m.outer.at("norm").floats(),
                          m.config.hidden_size, m.config.rms_norm_eps, m.S(1)));
  ASSIGN_OR_RETURN(
      auto logits,
      Buffer::Allocate(m.executor, size_t(m.config.vocab_size) * sizeof(float)));
  RETURN_IF_ERROR(m.Project(m.outer.at("head"), m.S(1),
                            static_cast<float*>(logits.data())));
  return logits;
}

absl::Status Model::Reset() {
  // A failed reset can leave only a prefix of the caches cleared. Do not let
  // that mixed history be consumed as if it were a valid sequence.
  impl_->poisoned = true;
  for (auto& block : impl_->blocks)
    if (block.full_attention)
      RETURN_IF_ERROR(block.full_attention->Reset());
    else
      RETURN_IF_ERROR(block.delta->Reset());
  impl_->position = 0;
  impl_->poisoned = false;
  return absl::OkStatus();
}
const Config& Model::config() const { return impl_->config; }
int Model::position() const { return impl_->position; }
size_t Model::weight_bytes() const { return impl_->weight_bytes; }

}  // namespace pluto::llm::qwen
