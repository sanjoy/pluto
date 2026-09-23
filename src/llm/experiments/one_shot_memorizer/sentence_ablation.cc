#include "src/llm/experiments/one_shot_memorizer/sentence_ablation.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

absl::Status ValidateTensor(const TensorSpec& tensor) {
  if (tensor.name.empty() || tensor.shape.empty() || tensor.shape.size() > 2)
    return absl::InvalidArgumentError(
        "parameter tensor needs a name and rank one or two");
  size_t elements = 1;
  for (size_t dimension : tensor.shape) {
    if (dimension == 0)
      return absl::InvalidArgumentError(
          "parameter dimensions must be positive");
    if (elements > std::numeric_limits<size_t>::max() / dimension)
      return absl::OutOfRangeError("parameter tensor shape overflows");
    elements *= dimension;
  }
  if (tensor.element_count != elements)
    return absl::InvalidArgumentError(
        "parameter element count disagrees with shape");
  if (tensor.flat_offset > std::numeric_limits<size_t>::max() - elements)
    return absl::OutOfRangeError("parameter flat offset overflows");
  if (tensor.token_embedding &&
      (tensor.shape.size() != 2 || tensor.qkv_width != 0 ||
       tensor.shape[0] > static_cast<size_t>(std::numeric_limits<int>::max())))
    return absl::InvalidArgumentError("invalid token embedding metadata");
  if (tensor.qkv_width != 0 &&
      (tensor.qkv_width > std::numeric_limits<size_t>::max() / 3 ||
       tensor.shape.back() != 3 * tensor.qkv_width ||
       (tensor.shape.size() == 2 && tensor.shape[0] != tensor.qkv_width)))
    return absl::InvalidArgumentError("invalid combined QKV metadata");
  return absl::OkStatus();
}

absl::Status ValidateLayout(absl::Span<const TensorSpec> layout) {
  if (layout.empty())
    return absl::InvalidArgumentError("parameter layout must not be empty");
  size_t offset = 0;
  absl::flat_hash_set<std::string> names;
  for (size_t index = 0; index < layout.size(); ++index) {
    const auto& tensor = layout[index];
    RETURN_IF_ERROR(ValidateTensor(tensor));
    if (tensor.checkpoint_index != index || tensor.flat_offset != offset)
      return absl::InvalidArgumentError(
          "parameter layout indices or offsets are not contiguous");
    if (!names.insert(tensor.name).second)
      return absl::InvalidArgumentError(
          "parameter layout contains duplicate names");
    offset += tensor.element_count;
  }
  return absl::OkStatus();
}

// Keep squared energies until every coordinate has been accumulated. This is
// safe in double for finite FP32 values and size_t-sized arrays, and avoids
// accumulating rounded per-tensor norms when constructing the global summary.
struct SummaryAccumulator {
  DeltaSummary summary;
  double baseline_squared = 0;
  double ablated_squared = 0;
  double delta_squared = 0;

  void Add(float baseline, float ablated, double delta, bool bits_changed) {
    ++summary.element_count;
    summary.bitwise_changed_count += bits_changed;
    summary.numerically_changed_count += baseline != ablated;
    baseline_squared += static_cast<double>(baseline) * baseline;
    ablated_squared += static_cast<double>(ablated) * ablated;
    delta_squared += delta * delta;
    summary.delta_l1 += std::abs(delta);
    summary.maximum_absolute_delta =
        std::max(summary.maximum_absolute_delta, std::abs(delta));
  }

  DeltaSummary Finish() {
    summary.baseline_l2 = std::sqrt(baseline_squared);
    summary.ablated_l2 = std::sqrt(ablated_squared);
    summary.delta_l2 = std::sqrt(delta_squared);
    if (summary.baseline_l2 != 0)
      summary.relative_l2 = summary.delta_l2 / summary.baseline_l2;
    return summary;
  }
};

}  // namespace

absl::StatusOr<std::vector<TensorSpec>> BuildGpt2ParameterLayout(
    const Gpt2ParameterDimensions& dimensions) {
  if (dimensions.vocabulary_size <= 0 || dimensions.model_width <= 0 ||
      dimensions.feed_forward_width <= 0 || dimensions.context_length <= 0 ||
      dimensions.transformer_block_count < 0)
    return absl::InvalidArgumentError("invalid GPT-2 parameter dimensions");
  const size_t width = dimensions.model_width;
  const size_t expansion = dimensions.feed_forward_width;
  const size_t blocks = dimensions.transformer_block_count;
  // The GPU recipe uses int element counts and a 3*width projection. Reject
  // impossible tensor/count arithmetic before allocating layout metadata.
  const size_t int_max = std::numeric_limits<int>::max();
  if (width > int_max / 3 || blocks > (int_max - 4) / 12)
    return absl::OutOfRangeError(
        "GPT-2 parameter dimensions exceed index range");
  std::vector<TensorSpec> layout;
  const size_t tensor_count = 4 + 12 * blocks;
  if (tensor_count > layout.max_size())
    return absl::OutOfRangeError("GPT-2 tensor count exceeds storage range");
  size_t offset = 0;
  auto append = [&](std::string name, std::vector<size_t> shape,
                    bool token_embedding = false,
                    size_t qkv_width = 0) -> absl::Status {
    size_t elements = 1;
    for (size_t dimension : shape) {
      if (dimension > int_max / elements)
        return absl::OutOfRangeError(
            "GPT-2 tensor exceeds GPU element-count range");
      elements *= dimension;
    }
    if (offset > std::numeric_limits<size_t>::max() - elements)
      return absl::OutOfRangeError("GPT-2 parameter offset overflows");
    layout.push_back(TensorSpec{.checkpoint_index = layout.size(),
                                .name = std::move(name),
                                .shape = std::move(shape),
                                .flat_offset = offset,
                                .element_count = elements,
                                .token_embedding = token_embedding,
                                .qkv_width = qkv_width});
    offset += elements;
    return absl::OkStatus();
  };
  RETURN_IF_ERROR(
      append("token_embedding.weight",
             {static_cast<size_t>(dimensions.vocabulary_size), width}, true));
  RETURN_IF_ERROR(
      append("position_embedding.weight",
             {static_cast<size_t>(dimensions.context_length), width}));
  for (size_t block = 0; block < blocks; ++block) {
    const std::string prefix = absl::StrCat("transformer_block_", block, ".");
    RETURN_IF_ERROR(append(prefix + "attention.layer_norm.gamma", {width}));
    RETURN_IF_ERROR(append(prefix + "attention.layer_norm.beta", {width}));
    RETURN_IF_ERROR(append(prefix + "attention.qkv.weight", {width, 3 * width},
                           false, width));
    RETURN_IF_ERROR(
        append(prefix + "attention.qkv.bias", {3 * width}, false, width));
    RETURN_IF_ERROR(append(prefix + "attention.output.weight", {width, width}));
    RETURN_IF_ERROR(append(prefix + "attention.output.bias", {width}));
    RETURN_IF_ERROR(append(prefix + "mlp.layer_norm.gamma", {width}));
    RETURN_IF_ERROR(append(prefix + "mlp.layer_norm.beta", {width}));
    RETURN_IF_ERROR(
        append(prefix + "mlp.expansion.weight", {width, expansion}));
    RETURN_IF_ERROR(append(prefix + "mlp.expansion.bias", {expansion}));
    RETURN_IF_ERROR(
        append(prefix + "mlp.contraction.weight", {expansion, width}));
    RETURN_IF_ERROR(append(prefix + "mlp.contraction.bias", {width}));
  }
  RETURN_IF_ERROR(append("final_layer_norm.gamma", {width}));
  RETURN_IF_ERROR(append("final_layer_norm.beta", {width}));
  RETURN_IF_ERROR(ValidateLayout(layout));
  return layout;
}

const char* QkvComponentName(QkvComponent component) {
  switch (component) {
    case QkvComponent::kNone:
      return "none";
    case QkvComponent::kQuery:
      return "query";
    case QkvComponent::kKey:
      return "key";
    case QkvComponent::kValue:
      return "value";
  }
  return "unknown";
}

absl::StatusOr<ParameterCoordinate> LocateParameter(const TensorSpec& tensor,
                                                    size_t element_index) {
  RETURN_IF_ERROR(ValidateTensor(tensor));
  if (element_index >= tensor.element_count)
    return absl::OutOfRangeError("parameter index is outside its tensor");
  ParameterCoordinate coordinate{
      .checkpoint_index = tensor.checkpoint_index,
      .element_index = element_index,
      .flat_index = tensor.flat_offset + element_index,
      .row = element_index};
  if (tensor.shape.size() == 2) {
    coordinate.row = element_index / tensor.shape[1];
    coordinate.column = element_index % tensor.shape[1];
  }
  if (tensor.token_embedding)
    coordinate.compact_token_id = static_cast<int>(coordinate.row);
  if (tensor.qkv_width != 0) {
    const size_t output = coordinate.column.value_or(coordinate.row);
    const size_t component = output / tensor.qkv_width;
    constexpr QkvComponent components[]{
        QkvComponent::kQuery, QkvComponent::kKey, QkvComponent::kValue};
    coordinate.qkv_component = components[component];
    coordinate.qkv_channel = output % tensor.qkv_width;
  }
  return coordinate;
}

absl::Status ValidateParameterValues(
    absl::Span<const TensorSpec> layout,
    absl::Span<const absl::Span<const float>> tensors) {
  RETURN_IF_ERROR(ValidateLayout(layout));
  if (layout.size() != tensors.size())
    return absl::InvalidArgumentError(
        "parameter tensor count differs from layout");
  for (size_t tensor = 0; tensor < layout.size(); ++tensor) {
    if (tensors[tensor].size() != layout[tensor].element_count)
      return absl::InvalidArgumentError(absl::StrCat(
          "parameter size differs from layout at weight_", tensor));
    for (float value : tensors[tensor])
      if (!std::isfinite(value))
        return absl::InvalidArgumentError(
            absl::StrCat("nonfinite parameter at weight_", tensor));
  }
  return absl::OkStatus();
}

absl::StatusOr<ParameterDeltaReport> CompareParameterValues(
    absl::Span<const TensorSpec> layout,
    absl::Span<const absl::Span<const float>> baseline,
    absl::Span<const absl::Span<const float>> ablated, size_t top_count) {
  RETURN_IF_ERROR(ValidateParameterValues(layout, baseline));
  RETURN_IF_ERROR(ValidateParameterValues(layout, ablated));
  ParameterDeltaReport result;
  result.tensors.reserve(layout.size());
  SummaryAccumulator total;
  for (size_t tensor = 0; tensor < layout.size(); ++tensor) {
    TensorDelta output{.tensor = layout[tensor]};
    output.deltas.resize(layout[tensor].element_count);
    output.bitwise_changed.resize(layout[tensor].element_count);
    SummaryAccumulator summary;
    std::vector<size_t> changed;
    for (size_t index = 0; index < output.deltas.size(); ++index) {
      const float before = baseline[tensor][index],
                  after = ablated[tensor][index];
      const double delta = static_cast<double>(before) - after;
      const bool bits_changed =
          std::bit_cast<uint32_t>(before) != std::bit_cast<uint32_t>(after);
      output.deltas[index] = delta;
      output.bitwise_changed[index] = bits_changed;
      summary.Add(before, after, delta, bits_changed);
      total.Add(before, after, delta, bits_changed);
      if (bits_changed && top_count != 0)
        changed.push_back(index);
    }
    output.summary = summary.Finish();
    const size_t selected = std::min(top_count, changed.size());
    std::partial_sort(changed.begin(), changed.begin() + selected,
                      changed.end(), [&](size_t left, size_t right) {
                        const double a = std::abs(output.deltas[left]);
                        const double b = std::abs(output.deltas[right]);
                        return a == b ? left < right : a > b;
                      });
    output.top_coordinates.reserve(selected);
    for (size_t index : absl::MakeConstSpan(changed).subspan(0, selected)) {
      ASSIGN_OR_RETURN(auto coordinate, LocateParameter(output.tensor, index));
      output.top_coordinates.push_back(
          CoordinateDelta{.coordinate = std::move(coordinate),
                          .baseline = baseline[tensor][index],
                          .ablated = ablated[tensor][index],
                          .delta = output.deltas[index]});
    }
    result.tensors.push_back(std::move(output));
  }
  result.total = total.Finish();
  return result;
}

}  // namespace pluto::llm::one_shot_memorizer
