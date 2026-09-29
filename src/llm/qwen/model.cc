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

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/delta_net.h"
#include "src/llm/layers/embedding.h"
#include "src/llm/layers/fully_connected.h"
#include "src/llm/layers/qwen_attention.h"
#include "src/llm/layers/rms_norm.h"
#include "src/llm/layers/swiglu.h"
#include "src/llm/layers/util.h"
#include "src/util/status_macros.h"

namespace pluto::llm::qwen {
namespace {

using cuda::Buffer;
using ::pluto::llm::MatrixStorage;

struct DeviceTensor {
  // Raw matrix storage (BF16/FP8), or FP32 storage for small scalar tensors.
  Buffer data;
  // FP32 multiplicative scales for FP8 matrices; absent for other storage.
  std::optional<Buffer> scales;
  MatrixStorage storage;
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
    return DeviceTensor{std::move(data), std::nullopt, MatrixStorage::kFloat32};
  }
  MatrixStorage storage =
      tensor.dtype == TensorDType::kBF16     ? MatrixStorage::kBFloat16
      : tensor.dtype == TensorDType::kF8E4M3 ? MatrixStorage::kFp8E4M3
                                             : MatrixStorage::kFloat32;
  ASSIGN_OR_RETURN(auto data, uploader.Copy(tensor.bytes));
  DeviceTensor result{std::move(data), std::nullopt, storage};
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

// Checkpoint names/layouts belong here; execution belongs to ordinary Layers.
// Uploaded weights are shared into layers without FP32 masters or gradients.
class LayerLoader {
 public:
  LayerLoader(cuda::Executor& executor, SafetensorsCheckpoint& checkpoint,
              Uploader& uploader)
      : executor_(executor), checkpoint_(checkpoint), uploader_(uploader) {}

  absl::StatusOr<std::unique_ptr<FullyConnectedLayer>> Linear(
      const std::string& name, int input_dim, int output_dim) {
    ASSIGN_OR_RETURN(auto tensor, LoadTensor(checkpoint_, uploader_, name,
                                             {output_dim, input_dim}, true));
    return FullyConnectedLayer::Create(executor_, std::move(tensor.data),
                                       tensor.storage, input_dim, output_dim,
                                       std::move(tensor.scales));
  }

  absl::StatusOr<Buffer> Scalar(const std::string& name,
                                absl::Span<const int64_t> shape) {
    ASSIGN_OR_RETURN(auto tensor,
                     LoadTensor(checkpoint_, uploader_, name, shape, false));
    return std::move(tensor.data);
  }

  absl::StatusOr<std::unique_ptr<RmsNormLayer>> Norm(const std::string& name,
                                                     int width, float epsilon) {
    ASSIGN_OR_RETURN(auto weight, Scalar(name, {width}));
    return RmsNormLayer::Create(executor_, width, std::move(weight), epsilon);
  }

 private:
  cuda::Executor& executor_;
  SafetensorsCheckpoint& checkpoint_;
  Uploader& uploader_;
};

// Both variants expose the same residual-stream signature. Projections are
// distinct layers: normal LayerHooks can inspect/intervene at each boundary.
absl::StatusOr<std::unique_ptr<ComposedLayer>> CreateAttentionBranch(
    cuda::Executor& executor, const Config& config, int index, int capacity,
    LayerLoader& loader, std::vector<QwenAttentionLayer*>& full_attention,
    std::vector<DeltaNetLayer*>& delta_net) {
  const std::string prefix =
      absl::StrCat("model.language_model.layers.", index, ".");
  const int d = config.hidden_size;
  ComposedLayerBuilder builder;
  RETURN_IF_ERROR(builder.add(
      loader.Norm(prefix + "input_layernorm.weight", d, config.rms_norm_eps)));
  std::vector<std::unique_ptr<Layer>> projections;
  if (config.layer_types[index] == LayerType::kFullAttention) {
    const int q = config.num_attention_heads * config.head_dim;
    const int kv = config.num_key_value_heads * config.head_dim;
    ASSIGN_OR_RETURN(
        auto query, loader.Linear(prefix + "self_attn.q_proj.weight", d, 2 * q));
    ASSIGN_OR_RETURN(auto key,
                     loader.Linear(prefix + "self_attn.k_proj.weight", d, kv));
    ASSIGN_OR_RETURN(auto value,
                     loader.Linear(prefix + "self_attn.v_proj.weight", d, kv));
    projections.push_back(std::move(query));
    projections.push_back(std::move(key));
    projections.push_back(std::move(value));
    RETURN_IF_ERROR(builder.add(
        ParallelLayer::Create("QueryGateKeyValue", std::move(projections))));
    ASSIGN_OR_RETURN(
        auto q_norm,
        loader.Scalar(prefix + "self_attn.q_norm.weight", {config.head_dim}));
    ASSIGN_OR_RETURN(
        auto k_norm,
        loader.Scalar(prefix + "self_attn.k_norm.weight", {config.head_dim}));
    cached_attention_ops::FullAttentionParameters p;
    p.query_heads = config.num_attention_heads;
    p.key_value_heads = config.num_key_value_heads;
    p.head_dim = config.head_dim;
    p.rotary_dim =
        static_cast<int>(config.head_dim * config.partial_rotary_factor);
    p.capacity = capacity;
    p.rms_norm_epsilon = config.rms_norm_eps;
    p.rope_theta = config.rope_theta;
    ASSIGN_OR_RETURN(auto attention,
                     QwenAttentionLayer::Create(executor, p, std::move(q_norm),
                                                std::move(k_norm)));
    full_attention.push_back(attention.get());
    RETURN_IF_ERROR(builder.add(std::move(attention)));
    RETURN_IF_ERROR(
        builder.add(loader.Linear(prefix + "self_attn.o_proj.weight", q, d)));
  } else {
    const int k = config.linear_num_key_heads * config.linear_key_head_dim;
    const int v = config.linear_num_value_heads * config.linear_value_head_dim;
    ASSIGN_OR_RETURN(
        auto qkv,
        loader.Linear(prefix + "linear_attn.in_proj_qkv.weight", d, 2 * k + v));
    ASSIGN_OR_RETURN(
        auto z, loader.Linear(prefix + "linear_attn.in_proj_z.weight", d, v));
    ASSIGN_OR_RETURN(auto a,
                     loader.Linear(prefix + "linear_attn.in_proj_a.weight", d,
                                   config.linear_num_value_heads));
    ASSIGN_OR_RETURN(auto b,
                     loader.Linear(prefix + "linear_attn.in_proj_b.weight", d,
                                   config.linear_num_value_heads));
    projections.push_back(std::move(qkv));
    projections.push_back(std::move(z));
    projections.push_back(std::move(a));
    projections.push_back(std::move(b));
    RETURN_IF_ERROR(builder.add(
        ParallelLayer::Create("DeltaNetProjections", std::move(projections))));
    ASSIGN_OR_RETURN(
        auto conv, loader.Scalar(prefix + "linear_attn.conv1d.weight",
                                 {2 * k + v, 1, config.linear_conv_kernel_dim}));
    ASSIGN_OR_RETURN(auto a_log, loader.Scalar(prefix + "linear_attn.A_log",
                                               {config.linear_num_value_heads}));
    ASSIGN_OR_RETURN(auto dt_bias,
                     loader.Scalar(prefix + "linear_attn.dt_bias",
                                   {config.linear_num_value_heads}));
    ASSIGN_OR_RETURN(auto norm,
                     loader.Scalar(prefix + "linear_attn.norm.weight",
                                   {config.linear_value_head_dim}));
    cached_attention_ops::DeltaNetParameters p;
    p.key_heads = config.linear_num_key_heads;
    p.value_heads = config.linear_num_value_heads;
    p.key_head_dim = config.linear_key_head_dim;
    p.value_head_dim = config.linear_value_head_dim;
    p.conv_kernel_dim = config.linear_conv_kernel_dim;
    p.rms_norm_epsilon = config.rms_norm_eps;
    ASSIGN_OR_RETURN(
        auto attention,
        DeltaNetLayer::Create(executor, p, std::move(conv), std::move(a_log),
                              std::move(dt_bias), std::move(norm)));
    delta_net.push_back(attention.get());
    RETURN_IF_ERROR(builder.add(std::move(attention)));
    RETURN_IF_ERROR(builder.add(
        loader.Linear(prefix + "linear_attn.out_proj.weight", v, d)));
  }
  return builder.create("AttentionBranch");
}

// Pre-norm decoder: Residual(norm -> attention -> output projection), then
// Residual(norm -> parallel gate/up projections -> SwiGLU -> down projection).
// Existing combinators check shapes, route hooks, and perform BF16 additions.
absl::StatusOr<std::unique_ptr<ComposedLayer>> CreateTransformerBlock(
    cuda::Executor& executor, const Config& config, int index, int capacity,
    LayerLoader& loader, std::vector<QwenAttentionLayer*>& full_attention,
    std::vector<DeltaNetLayer*>& delta_net) {
  ASSIGN_OR_RETURN(auto attention,
                   CreateAttentionBranch(executor, config, index, capacity,
                                         loader, full_attention, delta_net));
  ComposedLayerBuilder block;
  RETURN_IF_ERROR(block.add(ResidualLayer::Create(std::move(attention))));
  const std::string prefix =
      absl::StrCat("model.language_model.layers.", index, ".");
  const int d = config.hidden_size;
  const int f = config.intermediate_size;
  ComposedLayerBuilder mlp;
  RETURN_IF_ERROR(mlp.add(loader.Norm(
      prefix + "post_attention_layernorm.weight", d, config.rms_norm_eps)));
  std::vector<std::unique_ptr<Layer>> projections;
  ASSIGN_OR_RETURN(auto gate,
                   loader.Linear(prefix + "mlp.gate_proj.weight", d, f));
  ASSIGN_OR_RETURN(auto up, loader.Linear(prefix + "mlp.up_proj.weight", d, f));
  projections.push_back(std::move(gate));
  projections.push_back(std::move(up));
  RETURN_IF_ERROR(mlp.add(
      ParallelLayer::Create("GateAndUpProjections", std::move(projections))));
  RETURN_IF_ERROR(mlp.add(SwiGluLayer::Create(f)));
  RETURN_IF_ERROR(
      mlp.add(loader.Linear(prefix + "mlp.down_proj.weight", f, d)));
  ASSIGN_OR_RETURN(auto mlp_branch, mlp.create("FeedForward"));
  RETURN_IF_ERROR(block.add(ResidualLayer::Create(std::move(mlp_branch))));
  return block.create(absl::StrCat("QwenBlock", index));
}

}  // namespace

struct Model::Impl {
  Impl(cuda::Executor& executor, Config config, int capacity, Buffer token)
      : executor(executor),
        config(std::move(config)),
        capacity(capacity),
        token(std::move(token)) {}
  cuda::Executor& executor;
  Config config;
  int capacity;
  int position = 0;
  bool poisoned = false;
  size_t weight_bytes = 0;
  // Graphs own the parameters and caches. Raw pointers are only reset handles;
  // execution always runs through the ordinary Layer graph.
  std::unique_ptr<ComposedLayer> decoder;
  std::unique_ptr<ComposedLayer> head;
  std::vector<QwenAttentionLayer*> full_attention;
  std::vector<DeltaNetLayer*> delta_net;
  Buffer token;
  std::optional<Buffer> hidden;
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
  ASSIGN_OR_RETURN(auto checkpoint, SafetensorsCheckpoint::Open(directory));
  ASSIGN_OR_RETURN(auto staging, cuda::PageLockedHostArray<uint8_t>::Allocate(
                                     executor, 32 * 1024 * 1024));
  Uploader uploader(executor, std::move(staging));
  LayerLoader loader(executor, *checkpoint, uploader);
  ASSIGN_OR_RETURN(auto token, Buffer::Allocate(executor, sizeof(int32_t)));
  auto impl = absl::make_unique<Impl>(executor, config, options.context_length,
                                      std::move(token));
  const std::string root = "model.language_model.";
  ASSIGN_OR_RETURN(auto embedding, LoadTensor(*checkpoint, uploader,
                                              root + "embed_tokens.weight",
                                              {config.vocab_size, d}, true));
  // The graph has BF16 residual boundaries, matching this checkpoint's
  // declared compute dtype. Do not silently round an arbitrary FP32 table
  // whose low bits the previous FP32-storage path would have preserved.
  if (embedding.storage != MatrixStorage::kBFloat16)
    return absl::UnimplementedError("Qwen token embeddings must be BF16");
  ComposedLayerBuilder decoder;
  RETURN_IF_ERROR(decoder.add(
      EmbeddingLookupLayer::Create(executor, std::move(embedding.data),
                                   embedding.storage, config.vocab_size, d)));
  ComposedLayerBuilder head;
  RETURN_IF_ERROR(
      head.add(loader.Norm(root + "norm.weight", d, config.rms_norm_eps)));
  RETURN_IF_ERROR(
      head.add(loader.Linear("lm_head.weight", d, config.vocab_size)));
  ASSIGN_OR_RETURN(impl->head, head.create("LanguageModelingHead"));
  for (int i = 0; i < config.num_hidden_layers; ++i) {
    RETURN_IF_ERROR(decoder.add(
        CreateTransformerBlock(executor, config, i, options.context_length,
                               loader, impl->full_attention, impl->delta_net)));
    if (options.load_progress)
      options.load_progress(i + 1, config.num_hidden_layers);
  }
  ASSIGN_OR_RETURN(impl->decoder, decoder.create("QwenDecoder"));
  impl->weight_bytes = uploader.weight_bytes;
  RETURN_IF_ERROR(executor.Synchronize());
  return absl::WrapUnique(new Model(std::move(impl)));
}

absl::Status Model::Step(int token, LayerHooks* hooks) {
  auto& m = *impl_;
  if (m.poisoned)
    return absl::FailedPreconditionError(
        "Reset Qwen caches after a failed Step");
  if (token < 0 || token >= m.config.vocab_size || m.position >= m.capacity)
    return absl::InvalidArgumentError(
        "invalid Qwen token or context capacity exhausted");
  // Hooks or later layers can fail after some caches advance. Never consume
  // that mixed history until Reset has restored every cache.
  m.poisoned = true;
  ASSIGN_OR_RETURN(auto staging,
                   cuda::PageLockedHostArray<int32_t>::Allocate(m.executor, 1));
  staging[0] = token;
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(m.token.data(), staging.data(), sizeof(int32_t),
                      cudaMemcpyHostToDevice, m.executor.stream()),
      "upload Qwen token"));
  ASSIGN_OR_RETURN(auto result, m.decoder->fwd(m.executor, {m.token}, hooks));
  m.hidden = std::move(result.outputs[0]);
  ++m.position;
  m.poisoned = false;
  return absl::OkStatus();
}

absl::StatusOr<Buffer> Model::Logits(LayerHooks* hooks) {
  auto& m = *impl_;
  if (!m.hidden || m.poisoned)
    return absl::FailedPreconditionError(
        "Qwen logits require a successful Step");
  ASSIGN_OR_RETURN(auto result, m.head->fwd(m.executor, {*m.hidden}, hooks));
  // Keep the public FP32-logits API. Graph boundaries are physical BF16 so
  // the existing residual combinator rounds additions exactly as before.
  return ::pluto::llm::internal::ToFloat(m.executor, result.outputs[0],
                                         m.config.vocab_size);
}

absl::Status Model::Reset() {
  // A failed reset can leave only a prefix of the caches cleared.
  impl_->poisoned = true;
  for (auto* attention : impl_->full_attention)
    RETURN_IF_ERROR(attention->Reset());
  for (auto* delta : impl_->delta_net)
    RETURN_IF_ERROR(delta->Reset());
  impl_->hidden.reset();
  impl_->position = 0;
  impl_->poisoned = false;
  return absl::OkStatus();
}
const Config& Model::config() const { return impl_->config; }
int Model::position() const { return impl_->position; }
size_t Model::weight_bytes() const { return impl_->weight_bytes; }

}  // namespace pluto::llm::qwen
