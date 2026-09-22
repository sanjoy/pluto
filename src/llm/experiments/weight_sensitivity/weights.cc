#include "src/llm/experiments/weight_sensitivity/weights.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/util/status_macros.h"

namespace pluto::llm::weight_sensitivity {
namespace {

absl::Status ValidateTarget(const WeightTarget& target) {
  // Division-based checks avoid overflowing on malformed caller-owned metadata.
  if (target.rows == 0 || target.columns == 0 ||
      target.row_stride < target.columns ||
      target.offset >= target.tensor_elements ||
      target.columns > target.tensor_elements - target.offset ||
      target.rows - 1 >
          (target.tensor_elements - target.offset - target.columns) /
              target.row_stride)
    return absl::InvalidArgumentError("invalid weight target slice");
  return absl::OkStatus();
}

// Standardized engine bits plus an explicit Box-Muller transform avoid the
// implementation-dependent sampling algorithm of std::normal_distribution.
// Transcendental rounding can still differ across platforms/libm versions.
double Gaussian(std::mt19937_64& engine) {
  const double u = (static_cast<double>(engine() >> 11) + 0.5) * 0x1p-53;
  const double v = (static_cast<double>(engine() >> 11) + 0.5) * 0x1p-53;
  return std::sqrt(-2.0 * std::log(u)) * std::cos(2.0 * std::numbers::pi * v);
}

}  // namespace

absl::StatusOr<std::vector<WeightTarget>> DescribeWeights(
    const Gpt2Config& config) {
  RETURN_IF_ERROR(config.Validate());
  if (config.transformer_block_count >
      (std::numeric_limits<int>::max() - 4) / 12)
    return absl::InvalidArgumentError("too many checkpoint weight indices");
  const size_t width = config.model_width;
  const size_t ff_width = config.feed_forward_width;
  const size_t vocab =
      config.pad_vocabulary
          ? (static_cast<size_t>(config.vocabulary_size) + 15) / 16 * 16
          : config.vocabulary_size;
  std::vector<WeightTarget> result;
  result.reserve(static_cast<size_t>(config.transformer_block_count) * 16 + 4);
  int tensor = 0;
  int block = -1;
  auto whole = [&](std::string name, std::vector<int64_t> shape, size_t rows,
                   size_t columns) {
    result.push_back({std::move(name), tensor++, block, std::move(shape),
                      rows * columns, 0, rows, columns, columns});
  };
  auto vector = [&](std::string name, size_t size) {
    whole(std::move(name), {static_cast<int64_t>(size)}, 1, size);
  };
  auto matrix = [&](std::string name, size_t rows, size_t columns) {
    whole(std::move(name),
          {static_cast<int64_t>(rows), static_cast<int64_t>(columns)}, rows,
          columns);
  };
  matrix("Token embedding / tied LM head", vocab, width);
  matrix("Position embedding", kGpt2ContextLength, width);
  for (block = 0; block < config.transformer_block_count; ++block) {
    const std::string attention = absl::StrCat("Attn ", block);
    const std::string mlp = absl::StrCat("MLP ", block);
    vector(absl::StrCat(attention, " LayerNorm scale"), width);
    vector(absl::StrCat(attention, " LayerNorm bias"), width);
    // FullyConnected stores [input, output]; Q/K/V are interleaved by input
    // row, not three contiguous width-by-width matrices.
    constexpr const char* components[] = {"Q", "K", "V"};
    for (int component = 0; component < 3; ++component)
      result.push_back(
          {absl::StrCat(attention, " ", components[component], " matrix"),
           tensor,
           block,
           {config.model_width, config.model_width},
           3 * width * width,
           component * width,
           width,
           width,
           3 * width});
    ++tensor;
    for (int component = 0; component < 3; ++component)
      result.push_back(
          {absl::StrCat(attention, " ", components[component], " bias"),
           tensor,
           block,
           {config.model_width},
           3 * width,
           component * width,
           1,
           width,
           3 * width});
    ++tensor;
    matrix(absl::StrCat(attention, " output matrix"), width, width);
    vector(absl::StrCat(attention, " output bias"), width);
    vector(absl::StrCat(mlp, " LayerNorm scale"), width);
    vector(absl::StrCat(mlp, " LayerNorm bias"), width);
    matrix(absl::StrCat(mlp, " expansion matrix"), width, ff_width);
    vector(absl::StrCat(mlp, " expansion bias"), ff_width);
    matrix(absl::StrCat(mlp, " output matrix"), ff_width, width);
    vector(absl::StrCat(mlp, " output bias"), width);
  }
  block = -1;
  vector("Final LayerNorm scale", width);
  vector("Final LayerNorm bias", width);
  return result;
}

absl::Status ValidateWeights(const Gpt2Config& config,
                             absl::Span<const Buffer> unique_weights) {
  ASSIGN_OR_RETURN(auto targets, DescribeWeights(config));
  const size_t expected_count =
      static_cast<size_t>(config.transformer_block_count) * 12 + 4;
  if (unique_weights.size() != expected_count)
    return absl::InvalidArgumentError(absl::StrCat("expected ", expected_count,
                                                   " unique weights, found ",
                                                   unique_weights.size()));
  absl::flat_hash_set<const void*> pointers;
  for (const Buffer& weight : unique_weights)
    if (!pointers.insert(weight.data()).second)
      return absl::InvalidArgumentError("unique weights contain an alias");
  for (const WeightTarget& target : targets)
    if (unique_weights[target.checkpoint_index].size_bytes() !=
        target.tensor_elements * sizeof(float))
      return absl::InvalidArgumentError(
          absl::StrCat("weight ", target.checkpoint_index, " (", target.name,
                       ") has wrong byte size; expected ",
                       target.tensor_elements * sizeof(float)));
  return absl::OkStatus();
}

absl::StatusOr<double> ReplaceWithNoise(const WeightTarget& target,
                                        absl::Span<const float> original,
                                        absl::Span<float> destination,
                                        uint64_t seed, double noise_scale,
                                        double zero_rms_stddev) {
  RETURN_IF_ERROR(ValidateTarget(target));
  if (original.size() != target.tensor_elements ||
      destination.size() != target.tensor_elements)
    return absl::InvalidArgumentError(
        "noise replacement needs the complete physical tensor");
  if (!std::isfinite(noise_scale) || noise_scale <= 0 ||
      !std::isfinite(zero_rms_stddev) || zero_rms_stddev <= 0)
    return absl::InvalidArgumentError(
        "noise scale and zero-RMS standard deviation must be finite and "
        "positive");
  for (const float value : original)
    if (!std::isfinite(value))
      return absl::InvalidArgumentError("original weight is not finite");
  double squared_sum = 0;
  for (size_t row = 0; row < target.rows; ++row)
    for (size_t column = 0; column < target.columns; ++column) {
      const double value =
          original[target.offset + row * target.row_stride + column];
      squared_sum += value * value;
    }
  const double rms = std::sqrt(squared_sum / (target.rows * target.columns));
  const double stddev = noise_scale * (rms == 0 ? zero_rms_stddev : rms);
  if (!std::isfinite(stddev) || stddev <= 0)
    return absl::InvalidArgumentError(
        "noise standard deviation overflowed or underflowed");
  if (original.data() != destination.data())
    std::copy(original.begin(), original.end(), destination.begin());
  std::mt19937_64 engine(seed);
  for (size_t row = 0; row < target.rows; ++row)
    for (size_t column = 0; column < target.columns; ++column) {
      const double value = stddev * Gaussian(engine);
      if (!std::isfinite(value) ||
          std::abs(value) > std::numeric_limits<float>::max())
        return absl::InvalidArgumentError("noise exceeds FP32 range");
      destination[target.offset + row * target.row_stride + column] = value;
    }
  return stddev;
}

}  // namespace pluto::llm::weight_sensitivity
