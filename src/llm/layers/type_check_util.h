#pragma once

#include <cstddef>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "src/llm/layer.h"
#include "src/util/status_macros.h"

namespace pluto::llm::internal {

// Shared by device/reference combinators. Exact value equality deliberately
// keeps the batch sentinel distinct from fixed extents and other axes.
inline absl::Status ValidateTypes(absl::Span<const ActivationType> types) {
  for (const auto& type : types)
    RETURN_IF_ERROR(type.Validate());
  return absl::OkStatus();
}

inline absl::Status ValidateTypeConnection(
    absl::Span<const ActivationType> producer,
    absl::Span<const ActivationType> consumer) {
  if (producer.size() != consumer.size())
    return absl::InvalidArgumentError(absl::StrCat(
        "producer has ", producer.size(), " outputs but consumer expects ",
        consumer.size(), " inputs"));
  for (size_t i = 0; i < producer.size(); ++i)
    if (producer[i] != consumer[i])
      return absl::InvalidArgumentError(
          absl::StrCat("producer/consumer activation types differ at buffer ",
                       i, " (dtype and every dimension must match exactly)"));
  return absl::OkStatus();
}

inline absl::Status ValidateResidualTypes(
    absl::Span<const ActivationType> inputs,
    absl::Span<const ActivationType> outputs) {
  RETURN_IF_ERROR(ValidateTypes(inputs));
  RETURN_IF_ERROR(ValidateTypes(outputs));
  if (inputs.size() != 1 || outputs.size() != 1)
    return absl::InvalidArgumentError(
        "residual branch must have exactly one input and one output");
  RETURN_IF_ERROR(ValidateTypeConnection(outputs, inputs));
  // The residual addition kernels support these physical storage types, not
  // integers or native FP16/FP8 buffers. Legacy FP16 compute stores FP32 here.
  if (inputs[0].data_type() != DataType::BF16 &&
      inputs[0].data_type() != DataType::FP32)
    return absl::UnimplementedError(
        "residual addition requires BF16 or FP32 activations");
  return absl::OkStatus();
}

}  // namespace pluto::llm::internal
