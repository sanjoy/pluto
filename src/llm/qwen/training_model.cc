#include "src/llm/qwen/training_model.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <optional>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/llm/layers/block_training.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/sequence_delta_net.h"
#include "src/llm/layers/sequence_full_attention.h"
#include "src/util/status_macros.h"

namespace pluto::llm::qwen {
namespace {

// Import one parameter at a time with bounded pinned staging. Compressed GPU
// buffers are freed after conversion, not retained beside the BF16 model.
class TrainingLoader {
 public:
  TrainingLoader(cuda::Executor& executor, SafetensorsCheckpoint& checkpoint,
                 cuda::PageLockedHostArray<uint8_t> staging, int sequence)
      : executor_(executor),
        checkpoint_(checkpoint),
        staging_(std::move(staging)),
        sequence_(sequence) {}
  void BeginBlock(std::string name) {
    blocks_.push_back({std::move(name), {}});
  }
  std::vector<BAdamBlock> TakeBlocks() { return std::move(blocks_); }

  // Safetensors bytes are pageable, so copy through pinned staging and wait
  // only before its next overwrite. No whole-checkpoint host copy is needed.
  absl::StatusOr<Buffer> Upload(absl::Span<const uint8_t> bytes) {
    ASSIGN_OR_RETURN(auto result, Buffer::Allocate(executor_, bytes.size()));
    for (size_t offset = 0; offset < bytes.size(); offset += staging_.size()) {
      const size_t count = std::min(staging_.size(), bytes.size() - offset);
      std::memcpy(staging_.data(), bytes.data() + offset, count);
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemcpyAsync(static_cast<uint8_t*>(result.data()) + offset,
                          staging_.data(), count, cudaMemcpyHostToDevice,
                          executor_.stream()),
          "upload Qwen training weight"));
      RETURN_IF_ERROR(executor_.Synchronize());
    }
    return result;
  }

  // Norms and recurrence coefficients stay FP32, including imported BF16
  // scalars. These small arrays do not determine the model's memory footprint.
  absl::StatusOr<Buffer> FloatTensor(const TensorView& tensor,
                                     bool scale = false) {
    if (tensor.dtype == TensorDType::kF8E4M3)
      return absl::InvalidArgumentError("FP8 scalar weights are unsupported");
    const size_t bytes = tensor.dtype == TensorDType::kBF16 ? 2 : 4;
    std::vector<float> values(tensor.bytes.size() / bytes);
    for (size_t i = 0; i < values.size(); ++i) {
      if (bytes == 2) {
        const uint32_t bits = uint32_t(tensor.bytes[2 * i]) |
                              (uint32_t(tensor.bytes[2 * i + 1]) << 8);
        values[i] = std::bit_cast<float>(bits << 16);
      } else {
        std::memcpy(&values[i], tensor.bytes.data() + i * 4, 4);
      }
      if (!std::isfinite(values[i]) || (scale && values[i] <= 0))
        return absl::InvalidArgumentError("invalid Qwen scalar weight/scale");
    }
    return Upload({reinterpret_cast<const uint8_t*>(values.data()),
                   values.size() * sizeof(float)});
  }

  // Shapes and FP8 scale layouts are checkpoint contracts, not inferred from
  // byte counts. Registered resident buffers are never replaced by BAdam.
  absl::StatusOr<std::shared_ptr<BlockParameter>> Parameter(
      const std::string& name, absl::Span<const int64_t> shape, bool matrix) {
    ASSIGN_OR_RETURN(auto tensor, checkpoint_.Tensor(name));
    if (!std::equal(shape.begin(), shape.end(), tensor.shape.begin(),
                    tensor.shape.end()))
      return absl::InvalidArgumentError(
          absl::StrCat("weight shape mismatch: ", name));
    std::optional<Buffer> resident;
    if (matrix) {
      ASSIGN_OR_RETURN(auto raw, Upload(tensor.bytes));
      using inference_ops::MatrixStorage;
      const auto storage =
          tensor.dtype == TensorDType::kBF16  ? MatrixStorage::kBFloat16
          : tensor.dtype == TensorDType::kF32 ? MatrixStorage::kFloat32
                                              : MatrixStorage::kFp8E4M3;
      std::optional<Buffer> scales;
      if (tensor.dtype == TensorDType::kF8E4M3) {
        const auto scale_name =
            name.substr(0, name.size() - 7) + ".weight_scale_inv";
        ASSIGN_OR_RETURN(auto scale, checkpoint_.Tensor(scale_name));
        const std::vector<int64_t> expected{(shape[0] + 127) / 128,
                                            (shape[1] + 127) / 128};
        if (scale.shape != expected)
          return absl::InvalidArgumentError("incorrect FP8 scale shape");
        ASSIGN_OR_RETURN(scales, FloatTensor(scale, true));
      }
      ASSIGN_OR_RETURN(resident, DequantizeMatrix(executor_, raw, storage,
                                                  scales, shape[0], shape[1]));
    } else {
      ASSIGN_OR_RETURN(resident, FloatTensor(tensor));
    }
    ASSIGN_OR_RETURN(
        auto parameter,
        BlockParameter::Create(executor_, std::move(*resident),
                               matrix ? DataType::BF16 : DataType::FP32));
    blocks_.back().parameters.push_back(parameter);
    return parameter;
  }

  absl::StatusOr<std::unique_ptr<BlockLinearLayer>> Linear(
      const std::string& name, int input, int output, bool logits = false) {
    ASSIGN_OR_RETURN(auto parameter, Parameter(name, {output, input}, true));
    return BlockLinearLayer::Create(executor_, std::move(parameter), input,
                                    output, sequence_, logits);
  }
  absl::StatusOr<std::unique_ptr<BlockRmsNormLayer>> Norm(
      const std::string& name, int width, float epsilon) {
    ASSIGN_OR_RETURN(auto parameter, Parameter(name, {width}, false));
    return BlockRmsNormLayer::Create(executor_, std::move(parameter), width,
                                     sequence_, epsilon);
  }

 private:
  cuda::Executor& executor_;
  SafetensorsCheckpoint& checkpoint_;
  cuda::PageLockedHostArray<uint8_t> staging_;
  int sequence_;
  std::vector<BAdamBlock> blocks_;
};

// Build the same pre-norm/residual decoder recipe as inference, but with
// stateless sequence attention and complete backward through the causal past.
absl::StatusOr<std::unique_ptr<Layer>> DecoderBlock(cuda::Executor& executor,
                                                    const Config& c, int index,
                                                    int sequence,
                                                    TrainingLoader& loader) {
  const std::string root =
      absl::StrCat("model.language_model.layers.", index, ".");
  const int d = c.hidden_size;
  ComposedLayerBuilder attention;
  RETURN_IF_ERROR(attention.add(
      loader.Norm(root + "input_layernorm.weight", d, c.rms_norm_eps)));
  std::vector<std::unique_ptr<Layer>> projections;
  if (c.layer_types[index] == LayerType::kFullAttention) {
    const int q = c.num_attention_heads * c.head_dim;
    const int kv = c.num_key_value_heads * c.head_dim;
    ASSIGN_OR_RETURN(auto query,
                     loader.Linear(root + "self_attn.q_proj.weight", d, 2 * q));
    ASSIGN_OR_RETURN(auto key,
                     loader.Linear(root + "self_attn.k_proj.weight", d, kv));
    ASSIGN_OR_RETURN(auto value,
                     loader.Linear(root + "self_attn.v_proj.weight", d, kv));
    projections.push_back(std::move(query));
    projections.push_back(std::move(key));
    projections.push_back(std::move(value));
    RETURN_IF_ERROR(attention.add(
        ParallelLayer::Create("QueryGateKeyValue", std::move(projections))));
    ASSIGN_OR_RETURN(
        auto q_norm,
        loader.Parameter(root + "self_attn.q_norm.weight", {c.head_dim}, false));
    ASSIGN_OR_RETURN(
        auto k_norm,
        loader.Parameter(root + "self_attn.k_norm.weight", {c.head_dim}, false));
    cached_attention_ops::FullAttentionParameters p;
    p.query_heads = c.num_attention_heads;
    p.key_value_heads = c.num_key_value_heads;
    p.head_dim = c.head_dim;
    p.rotary_dim = static_cast<int>(c.head_dim * c.partial_rotary_factor);
    p.capacity = sequence;
    p.rms_norm_epsilon = c.rms_norm_eps;
    p.rope_theta = c.rope_theta;
    RETURN_IF_ERROR(attention.add(SequenceFullAttentionLayer::Create(
        executor, p, q_norm, k_norm, sequence)));
    RETURN_IF_ERROR(
        attention.add(loader.Linear(root + "self_attn.o_proj.weight", q, d)));
  } else {
    const int k = c.linear_num_key_heads * c.linear_key_head_dim;
    const int v = c.linear_num_value_heads * c.linear_value_head_dim;
    ASSIGN_OR_RETURN(
        auto qkv,
        loader.Linear(root + "linear_attn.in_proj_qkv.weight", d, 2 * k + v));
    ASSIGN_OR_RETURN(auto z,
                     loader.Linear(root + "linear_attn.in_proj_z.weight", d, v));
    ASSIGN_OR_RETURN(auto a,
                     loader.Linear(root + "linear_attn.in_proj_a.weight", d,
                                   c.linear_num_value_heads));
    ASSIGN_OR_RETURN(auto b,
                     loader.Linear(root + "linear_attn.in_proj_b.weight", d,
                                   c.linear_num_value_heads));
    projections.push_back(std::move(qkv));
    projections.push_back(std::move(z));
    projections.push_back(std::move(a));
    projections.push_back(std::move(b));
    RETURN_IF_ERROR(attention.add(
        ParallelLayer::Create("DeltaNetProjections", std::move(projections))));
    ASSIGN_OR_RETURN(
        auto conv,
        loader.Parameter(root + "linear_attn.conv1d.weight",
                         {2 * k + v, 1, c.linear_conv_kernel_dim}, false));
    ASSIGN_OR_RETURN(auto a_log,
                     loader.Parameter(root + "linear_attn.A_log",
                                      {c.linear_num_value_heads}, false));
    ASSIGN_OR_RETURN(auto dt_bias,
                     loader.Parameter(root + "linear_attn.dt_bias",
                                      {c.linear_num_value_heads}, false));
    ASSIGN_OR_RETURN(auto norm,
                     loader.Parameter(root + "linear_attn.norm.weight",
                                      {c.linear_value_head_dim}, false));
    cached_attention_ops::DeltaNetParameters p;
    p.key_heads = c.linear_num_key_heads;
    p.value_heads = c.linear_num_value_heads;
    p.key_head_dim = c.linear_key_head_dim;
    p.value_head_dim = c.linear_value_head_dim;
    p.conv_kernel_dim = c.linear_conv_kernel_dim;
    p.rms_norm_epsilon = c.rms_norm_eps;
    RETURN_IF_ERROR(attention.add(SequenceDeltaNetLayer::Create(
        executor, p, conv, a_log, dt_bias, norm, sequence)));
    RETURN_IF_ERROR(attention.add(
        loader.Linear(root + "linear_attn.out_proj.weight", v, d)));
  }
  ASSIGN_OR_RETURN(auto attention_branch, attention.create("AttentionBranch"));
  ComposedLayerBuilder block;
  RETURN_IF_ERROR(
      block.add(ResidualLayer::Create(std::move(attention_branch))));
  ComposedLayerBuilder mlp;
  RETURN_IF_ERROR(mlp.add(loader.Norm(root + "post_attention_layernorm.weight",
                                      d, c.rms_norm_eps)));
  std::vector<std::unique_ptr<Layer>> gate_up;
  ASSIGN_OR_RETURN(auto gate, loader.Linear(root + "mlp.gate_proj.weight", d,
                                            c.intermediate_size));
  ASSIGN_OR_RETURN(auto up, loader.Linear(root + "mlp.up_proj.weight", d,
                                          c.intermediate_size));
  gate_up.push_back(std::move(gate));
  gate_up.push_back(std::move(up));
  RETURN_IF_ERROR(mlp.add(
      ParallelLayer::Create("GateAndUpProjections", std::move(gate_up))));
  RETURN_IF_ERROR(
      mlp.add(BlockSwiGluLayer::Create(c.intermediate_size, sequence)));
  RETURN_IF_ERROR(mlp.add(
      loader.Linear(root + "mlp.down_proj.weight", c.intermediate_size, d)));
  ASSIGN_OR_RETURN(auto feed_forward, mlp.create("FeedForward"));
  RETURN_IF_ERROR(block.add(ResidualLayer::Create(std::move(feed_forward))));
  return block.create(absl::StrCat("QwenBlock", index));
}

}  // namespace

TrainingModel::TrainingModel(cuda::Executor& executor, Config config,
                             std::vector<std::unique_ptr<Layer>> blocks,
                             std::vector<BAdamBlock> parameters)
    : executor_(executor),
      config_(std::move(config)),
      blocks_(std::move(blocks)),
      parameters_(std::move(parameters)) {
  for (const auto& block : parameters_)
    for (const auto& p : block.parameters)
      weights_.push_back(p->value());
}

// The training loader dequantizes matrices once and registers every text weight
// exactly once. It never loads vision or MTP tensors or allocates dense
// moments.
absl::StatusOr<std::unique_ptr<TrainingModel>> TrainingModel::Load(
    cuda::Executor& executor, const std::filesystem::path& directory,
    const TrainingModelOptions& options) {
  ASSIGN_OR_RETURN(auto config, LoadConfig(directory));
  const int sequence = options.sequence_length;
  if (sequence <= 0 || sequence > 128 ||
      sequence > config.max_position_embeddings)
    return absl::InvalidArgumentError(
        "training sequence_length must be in [1,128] and within the checkpoint "
        "context");
  if (config.vocab_size % 16 != 0)
    return absl::UnimplementedError(
        "training requires vocabulary padded to 16");
  ASSIGN_OR_RETURN(auto checkpoint, SafetensorsCheckpoint::Open(directory));
  ASSIGN_OR_RETURN(auto staging, cuda::PageLockedHostArray<uint8_t>::Allocate(
                                     executor, 32 * 1024 * 1024));
  TrainingLoader loader(executor, *checkpoint, std::move(staging), sequence);
  std::vector<std::unique_ptr<Layer>> blocks;
  loader.BeginBlock("TokenEmbedding");
  ASSIGN_OR_RETURN(
      auto embedding,
      loader.Parameter("model.language_model.embed_tokens.weight",
                       {config.vocab_size, config.hidden_size}, true));
  ASSIGN_OR_RETURN(
      auto embedding_layer,
      BlockEmbeddingLayer::Create(executor, embedding, config.vocab_size,
                                  config.hidden_size, sequence));
  blocks.push_back(std::move(embedding_layer));
  for (int i = 0; i < config.num_hidden_layers; ++i) {
    loader.BeginBlock(absl::StrCat("QwenBlock", i));
    ASSIGN_OR_RETURN(auto block,
                     DecoderBlock(executor, config, i, sequence, loader));
    blocks.push_back(std::move(block));
    if (options.load_progress)
      options.load_progress(i + 1, config.num_hidden_layers);
  }
  loader.BeginBlock("LanguageModelingHead");
  ComposedLayerBuilder head;
  RETURN_IF_ERROR(
      head.add(loader.Norm("model.language_model.norm.weight",
                           config.hidden_size, config.rms_norm_eps)));
  RETURN_IF_ERROR(head.add(loader.Linear("lm_head.weight", config.hidden_size,
                                         config.vocab_size, true)));
  ASSIGN_OR_RETURN(auto output, head.create("LanguageModelingHead"));
  blocks.push_back(std::move(output));
  RETURN_IF_ERROR(executor.Synchronize());
  return absl::WrapUnique(new TrainingModel(
      executor, std::move(config), std::move(blocks), loader.TakeBlocks()));
}

absl::Span<const ActivationType> TrainingModel::input_types() const {
  return blocks_.front()->input_types();
}
absl::Span<const ActivationType> TrainingModel::output_types() const {
  return blocks_.back()->output_types();
}
size_t TrainingModel::resident_weight_bytes() const {
  size_t total = 0;
  for (const auto& weight : weights_)
    total += weight.size_bytes();
  return total;
}
absl::Status TrainingModel::SetBackwardStart(int block) {
  if (block < 0 || block >= static_cast<int>(blocks_.size()))
    return absl::InvalidArgumentError("invalid Qwen backward start block");
  backward_start_ = block;
  return absl::OkStatus();
}

// Dropping prefix saved state is safe because BAdam never asks for that
// prefix's gradients. Later frozen blocks still save state for the chain rule.
absl::StatusOr<FwdResult> TrainingModel::fwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> inputs,
    LayerHooks* hooks) const {
  if (&executor != &executor_)
    return absl::InvalidArgumentError("Qwen training executor mismatch");
  BufferVec values(inputs.begin(), inputs.end());
  BackwardState state;
  for (size_t i = 0; i < blocks_.size(); ++i) {
    ASSIGN_OR_RETURN(auto result, blocks_[i]->fwd(executor, values, hooks));
    values = std::move(result.outputs);
    if (static_cast<int>(i) >= backward_start_)
      state.children.push_back(std::move(result.state));
  }
  return FwdResult{std::move(values), std::move(state)};
}

// The retained suffix fixes the backward boundary at forward time. Never
// detach intermediate frozen layers; stop only before the earliest active one.
absl::StatusOr<BufferVec> TrainingModel::bwd_impl(
    cuda::Executor& executor, absl::Span<const Buffer> gradients,
    BackwardState state, LayerHooks* hooks) {
  if (&executor != &executor_ || state.children.empty() ||
      state.children.size() > blocks_.size())
    return absl::InvalidArgumentError("invalid Qwen backward state/executor");
  BufferVec values(gradients.begin(), gradients.end());
  const size_t first = blocks_.size() - state.children.size();
  for (size_t i = blocks_.size(); i > first; --i) {
    ASSIGN_OR_RETURN(
        values,
        blocks_[i - 1]->bwd(executor, values,
                            std::move(state.children[i - 1 - first]), hooks));
  }
  // The model's public inputs are integer token IDs, not differentiable floats.
  return BufferVec{};
}

}  // namespace pluto::llm::qwen
