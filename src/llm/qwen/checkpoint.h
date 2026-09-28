#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace pluto::llm::qwen {

enum class TensorDType { kBF16, kF32, kF8E4M3 };

// An unconverted, little-endian view of checkpoint storage. The originating
// checkpoint must outlive this view. Files must not change while mapped.
struct TensorView {
  TensorDType dtype;
  std::vector<int64_t> shape;
  absl::Span<const uint8_t> bytes;
};

// Reads a Hugging Face sharded index, or a single model.safetensors file.
// Shards are mapped and fully validated on first access, without copying or
// dequantizing their tensor payloads. Instances are not thread-safe.
class SafetensorsCheckpoint {
 public:
  static absl::StatusOr<std::unique_ptr<SafetensorsCheckpoint>> Open(
      const std::filesystem::path& directory);
  ~SafetensorsCheckpoint();

  SafetensorsCheckpoint(const SafetensorsCheckpoint&) = delete;
  SafetensorsCheckpoint& operator=(const SafetensorsCheckpoint&) = delete;

  absl::StatusOr<TensorView> Tensor(absl::string_view name);
  const std::vector<std::string>& tensor_names() const;

 private:
  struct Impl;
  explicit SafetensorsCheckpoint(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

enum class LayerType { kLinearAttention, kFullAttention };

// The text decoder and quantization settings used by Qwen3.8-27B-FP8.
// Vision and multi-token-prediction configuration is intentionally ignored.
struct Config {
  int hidden_size = 0;
  int intermediate_size = 0;
  int num_hidden_layers = 0;
  int num_attention_heads = 0;
  int num_key_value_heads = 0;
  int head_dim = 0;
  int vocab_size = 0;
  int max_position_embeddings = 0;
  int linear_conv_kernel_dim = 0;
  int linear_key_head_dim = 0;
  int linear_num_key_heads = 0;
  int linear_num_value_heads = 0;
  int linear_value_head_dim = 0;
  int full_attention_interval = 0;
  int bos_token_id = -1;
  int eos_token_id = -1;
  double rms_norm_eps = 0;
  double rope_theta = 0;
  double partial_rotary_factor = 0;
  bool attn_output_gate = false;
  bool tie_word_embeddings = false;
  bool mrope_interleaved = false;
  std::string output_gate_type;
  std::array<int, 3> mrope_section = {};
  std::array<int, 2> weight_block_size = {};
  std::vector<LayerType> layer_types;
};

absl::StatusOr<Config> ParseConfig(absl::string_view json);
absl::StatusOr<Config> LoadConfig(const std::filesystem::path& directory);

}  // namespace pluto::llm::qwen
