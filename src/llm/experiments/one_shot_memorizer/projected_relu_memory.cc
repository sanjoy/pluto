#include "src/llm/experiments/one_shot_memorizer/projected_relu_memory.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "src/llm/experiments/one_shot_memorizer/split_mix64.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {
constexpr uint64_t kCoefficientBound = uint64_t{1} << 20;
constexpr uint32_t kAttempts = 256;
constexpr absl::string_view kMagic = "PLUTO_PROJECTED_RELU_V1\n";
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
using Key = std::array<int, kProjectedReluWindow>;

absl::Status Metadata(const ProjectedReluMemory& model) {
  if (model.vocabulary_size <= 0 || model.vocabulary_size > 65536 ||
      model.eos_token_id < 0 || model.eos_token_id >= model.vocabulary_size)
    return absl::InvalidArgumentError("invalid projected ReLU vocabulary/EOS");
  return absl::OkStatus();
}

absl::Status Prefix(const ProjectedReluMemory& model,
                    absl::Span<const int> prefix) {
  if (prefix.size() < kProjectedReluPrompt)
    return absl::InvalidArgumentError(
        "projected ReLU needs at least five tokens");
  for (int token : prefix)
    if (token < 0 || token >= model.vocabulary_size ||
        token == model.eos_token_id)
      return absl::InvalidArgumentError("invalid projected ReLU input token");
  return absl::OkStatus();
}

Key MakeKey(absl::Span<const int> prefix) {
  Key key;
  for (size_t offset = 0; offset < key.size(); ++offset)
    key[key.size() - 1 - offset] =
        offset < prefix.size() ? prefix[prefix.size() - 1 - offset] : -1;
  return key;
}

double Project(const ProjectedReluMemory& model, const Key& key) {
  double hash = 0;
  // Each product and partial sum is an integer of magnitude below 2^40. Thus
  // FP64 multiplication/addition are exact, not a floating-point hash guess.
  for (size_t j = 0; j < key.size(); ++j)
    hash += model.projection[j] * static_cast<double>(key[j]);
  return hash;
}

absl::StatusOr<int> Evaluate(const ProjectedReluMemory& model,
                             absl::Span<const int> prefix) {
  const double hash = Project(model, MakeKey(prefix));
  double output = 0;
  size_t active = 0;
  for (const auto& unit : model.units) {
    const double left = std::max(0.0, hash + unit.negative_bias);
    const double right = std::max(0.0, -hash + unit.positive_bias);
    const double hidden = std::max(0.0, 1.0 - left - right);
    output += hidden * unit.output_weight;
    active += hidden == 1;
  }
  if (active == 0)
    return absl::NotFoundError("no projected ReLU hash match for this suffix");
  if (active != 1 || !std::isfinite(output) || std::trunc(output) != output ||
      output < 0 || output >= model.vocabulary_size)
    return absl::DataLossError("projected ReLU output is not one valid token");
  return static_cast<int>(output);
}

void Append(uint64_t value, int width, std::string& bytes) {
  for (int i = 0; i < width; ++i)
    bytes.push_back(static_cast<char>((value >> (8 * i)) & 255));
}

absl::StatusOr<uint64_t> Read(int width, absl::string_view& bytes) {
  if (bytes.size() < static_cast<size_t>(width))
    return absl::DataLossError("truncated projected ReLU artifact");
  uint64_t value = 0;
  for (int i = 0; i < width; ++i)
    value |= uint64_t{static_cast<unsigned char>(bytes[i])} << (8 * i);
  bytes.remove_prefix(width);
  return value;
}
}  // namespace

absl::StatusOr<ProjectedReluMemory> BuildProjectedReluMemory(
    const std::vector<std::vector<int>>& sentences, int vocabulary_size,
    int eos_token_id) {
  ProjectedReluMemory model{.vocabulary_size = vocabulary_size,
                            .eos_token_id = eos_token_id};
  RETURN_IF_ERROR(Metadata(model));
  if (sentences.empty())
    return absl::InvalidArgumentError("projected ReLU corpus is empty");
  std::map<Key, int> targets;
  for (const auto& sentence : sentences) {
    RETURN_IF_ERROR(Prefix(model, sentence));
    for (size_t position = kProjectedReluPrompt; position <= sentence.size();
         ++position) {
      const Key key = MakeKey(absl::MakeConstSpan(sentence).first(position));
      const int target =
          position == sentence.size() ? eos_token_id : sentence[position];
      const auto [it, inserted] = targets.emplace(key, target);
      if (!inserted && it->second != target)
        return absl::InvalidArgumentError(
            "conflicting labels for the same nine-token suffix");
    }
  }
  internal::SplitMix64 random(0);
  for (uint32_t attempt = 1; attempt <= kAttempts; ++attempt) {
    for (double& weight : model.projection)
      weight = static_cast<double>(1 + random.Bounded(kCoefficientBound));
    absl::flat_hash_set<int64_t> hashes;
    bool collision = false;
    for (const auto& [key, target] : targets)
      if (!hashes.insert(static_cast<int64_t>(Project(model, key))).second) {
        collision = true;
        break;
      }
    if (collision)
      continue;
    model.projection_attempts = attempt;
    model.units.reserve(targets.size());
    for (const auto& [key, target] : targets) {
      const double hash = Project(model, key);
      model.units.push_back({-hash, hash, static_cast<double>(target)});
    }
    std::sort(model.units.begin(), model.units.end(),
              [](const auto& a, const auto& b) {
                return a.positive_bias < b.positive_bias;
              });
    RETURN_IF_ERROR(ValidateProjectedReluMemory(model));
    return model;
  }
  return absl::ResourceExhaustedError(
      "all 256 fixed projected ReLU coefficient draws had collisions");
}

absl::Status ValidateProjectedReluMemory(const ProjectedReluMemory& model) {
  RETURN_IF_ERROR(Metadata(model));
  if (model.projection_attempts == 0 || model.projection_attempts > kAttempts ||
      model.units.empty())
    return absl::InvalidArgumentError("invalid projected ReLU attempts/units");
  double coefficient_sum = 0;
  for (double value : model.projection) {
    if (!std::isfinite(value) || value < 1 || value > kCoefficientBound ||
        std::trunc(value) != value)
      return absl::InvalidArgumentError("invalid integer projection weight");
    coefficient_sum += value;
  }
  // The artifact names a fixed seed and draw number, so verify those claims.
  // Original corpus keys are intentionally absent: this cannot certify that
  // earlier draws collided, only the identity of the stored chosen draw.
  internal::SplitMix64 random(0);
  for (uint32_t attempt = 1; attempt <= model.projection_attempts; ++attempt)
    for (double value : model.projection) {
      const double expected =
          static_cast<double>(1 + random.Bounded(kCoefficientBound));
      if (attempt == model.projection_attempts && value != expected)
        return absl::InvalidArgumentError(
            "projection weights disagree with seed/draw metadata");
    }
  const double minimum = -coefficient_sum;
  const double maximum = (model.vocabulary_size - 1.0) * coefficient_sum;
  // All legal inputs and biases lie in this interval. Distances and the +1
  // threshold stay exactly representable integers, comfortably below 2^53.
  if (maximum - minimum + 1 >= 0x1p53)
    return absl::InvalidArgumentError("projection exceeds exact FP64 range");
  double previous = -std::numeric_limits<double>::infinity();
  for (const auto& unit : model.units) {
    const double hash = unit.positive_bias;
    if (!std::isfinite(hash) || !std::isfinite(unit.negative_bias) ||
        unit.negative_bias != -hash || std::trunc(hash) != hash ||
        hash < minimum || hash > maximum || hash <= previous)
      return absl::InvalidArgumentError(
          "invalid or duplicate projected biases");
    if (!std::isfinite(unit.output_weight) ||
        std::trunc(unit.output_weight) != unit.output_weight ||
        unit.output_weight < 0 || unit.output_weight >= model.vocabulary_size)
      return absl::InvalidArgumentError("invalid projected output weight");
    previous = hash;
  }
  return absl::OkStatus();
}

absl::StatusOr<int> ProjectedReluNextToken(const ProjectedReluMemory& model,
                                           absl::Span<const int> prefix) {
  RETURN_IF_ERROR(ValidateProjectedReluMemory(model));
  RETURN_IF_ERROR(Prefix(model, prefix));
  return Evaluate(model, prefix);
}

absl::StatusOr<std::vector<int>> ProjectedReluGreedyContinuation(
    const ProjectedReluMemory& model, absl::Span<const int> prefix,
    size_t max_new_tokens) {
  RETURN_IF_ERROR(ValidateProjectedReluMemory(model));
  RETURN_IF_ERROR(Prefix(model, prefix));
  if (max_new_tokens == 0)
    return absl::InvalidArgumentError("generation limit must be positive");
  std::vector<int> context(prefix.begin(), prefix.end()), generated;
  for (size_t step = 0; step < max_new_tokens; ++step) {
    ASSIGN_OR_RETURN(const int token, Evaluate(model, context));
    generated.push_back(token);
    if (token == model.eos_token_id)
      break;
    context.push_back(token);
  }
  return generated;
}

absl::StatusOr<std::string> SerializeProjectedReluMemory(
    const ProjectedReluMemory& model) {
  RETURN_IF_ERROR(ValidateProjectedReluMemory(model));
  std::string bytes(kMagic);
  Append(model.vocabulary_size, 4, bytes);
  Append(model.eos_token_id, 4, bytes);
  Append(kProjectedReluPrompt, 4, bytes);
  Append(kProjectedReluWindow, 4, bytes);
  Append(0, 8, bytes);  // Fixed construction seed.
  Append(model.projection_attempts, 4, bytes);
  Append(model.units.size(), 8, bytes);
  for (double value : model.projection)
    Append(std::bit_cast<uint64_t>(value), 8, bytes);
  for (const auto& unit : model.units)
    for (double value :
         {unit.negative_bias, unit.positive_bias, unit.output_weight})
      Append(std::bit_cast<uint64_t>(value), 8, bytes);
  return bytes;
}

absl::StatusOr<ProjectedReluMemory> DeserializeProjectedReluMemory(
    absl::string_view bytes) {
  if (!bytes.starts_with(kMagic))
    return absl::DataLossError("invalid projected ReLU format");
  bytes.remove_prefix(kMagic.size());
  ASSIGN_OR_RETURN(const auto vocabulary, Read(4, bytes));
  ASSIGN_OR_RETURN(const auto eos, Read(4, bytes));
  ASSIGN_OR_RETURN(const auto prompt, Read(4, bytes));
  ASSIGN_OR_RETURN(const auto window, Read(4, bytes));
  ASSIGN_OR_RETURN(const auto seed, Read(8, bytes));
  ASSIGN_OR_RETURN(const auto attempts, Read(4, bytes));
  ASSIGN_OR_RETURN(const auto units, Read(8, bytes));
  if (vocabulary == 0 || vocabulary > 65536 || eos >= vocabulary ||
      prompt != kProjectedReluPrompt || window != kProjectedReluWindow ||
      seed != 0 || bytes.size() < kProjectedReluWindow * 8)
    return absl::DataLossError("invalid projected ReLU metadata");
  ProjectedReluMemory model{
      .vocabulary_size = static_cast<int>(vocabulary),
      .eos_token_id = static_cast<int>(eos),
      .projection_attempts = static_cast<uint32_t>(attempts)};
  for (double& weight : model.projection) {
    ASSIGN_OR_RETURN(const auto bits, Read(8, bytes));
    weight = std::bit_cast<double>(bits);
  }
  if (units > bytes.size() / 24 || units * 24 != bytes.size())
    return absl::DataLossError(
        "projected unit count disagrees with byte extent");
  model.units.reserve(static_cast<size_t>(units));
  for (uint64_t i = 0; i < units; ++i) {
    ProjectedReluUnit unit;
    for (double* weight :
         {&unit.negative_bias, &unit.positive_bias, &unit.output_weight}) {
      ASSIGN_OR_RETURN(const auto bits, Read(8, bytes));
      *weight = std::bit_cast<double>(bits);
    }
    model.units.push_back(unit);
  }
  RETURN_IF_ERROR(ValidateProjectedReluMemory(model));
  return model;
}

}  // namespace pluto::llm::one_shot_memorizer
