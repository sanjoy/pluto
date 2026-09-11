#include "ai-slop/weight_analysis/head_context/selection.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>

namespace pluto::weight_analysis::head_context {
namespace {

bool ValidScale(float scale) {
  return scale == 0.0f || scale == 0.5f || scale == 1.0f;
}

bool Finite(uint16_t bits) { return (bits & 0x7f80) != 0x7f80; }

// Call only after validation. For normal exponents >= 2, halving decrements
// the exponent. Smaller values shift into/subdivide the subnormal range;
// two low bits determine whether the halfway result needs an even rounding.
uint16_t ScaleValidated(uint16_t bits, float scale) {
  if (scale == 0.0f)
    return 0;
  if (scale == 1.0f)
    return bits;
  const uint16_t sign = bits & 0x8000;
  const uint16_t magnitude = bits & 0x7fff;
  if (magnitude >= 0x100)
    return sign | (magnitude - 0x80);
  return sign | ((magnitude >> 1) + ((magnitude & 3) == 3 ? 1 : 0));
}

}  // namespace

absl::Status ValidateSelection(const Geometry& geometry,
                               const Selection& selection, int total_rows) {
  if (geometry.blocks <= 0 || geometry.context_length <= 0 ||
      geometry.heads <= 0 || geometry.head_dimension <= 0 || total_rows <= 0 ||
      total_rows % geometry.context_length != 0 ||
      geometry.heads >
          std::numeric_limits<int>::max() / geometry.head_dimension) {
    return absl::InvalidArgumentError("invalid head-context geometry");
  }
  const int width = geometry.heads * geometry.head_dimension;
  if (total_rows > std::numeric_limits<int>::max() / width ||
      selection.block < 0 || selection.block >= geometry.blocks ||
      selection.head < 0 || selection.head >= geometry.heads ||
      selection.sequence < 0 ||
      selection.sequence >= total_rows / geometry.context_length ||
      selection.query_position < 0 ||
      selection.query_position >= geometry.context_length ||
      (selection.scope != QueryScope::kSelectedQuery &&
       selection.scope != QueryScope::kOtherQueries &&
       selection.scope != QueryScope::kAllQueries) ||
      !ValidScale(selection.scale)) {
    return absl::InvalidArgumentError("invalid head-context intervention");
  }
  return absl::OkStatus();
}

absl::StatusOr<uint16_t> ScaleBf16(uint16_t bits, float scale) {
  if (!ValidScale(scale) || !Finite(bits)) {
    return absl::InvalidArgumentError(
        "expected finite BF16 and dose 0, 0.5, or 1");
  }
  return ScaleValidated(bits, scale);
}

absl::Status ScaleContext(absl::Span<const uint16_t> original,
                          absl::Span<uint16_t> destination, int total_rows,
                          const Geometry& geometry,
                          const Selection& selection) {
  const auto status = ValidateSelection(geometry, selection, total_rows);
  if (!status.ok())
    return status;
  const size_t width =
      static_cast<size_t>(geometry.heads) * geometry.head_dimension;
  const size_t elements = static_cast<size_t>(total_rows) * width;
  if (original.size() != elements || destination.size() != elements)
    return absl::InvalidArgumentError("head-context array shape differs");
  // std::less gives a total pointer order even for unrelated allocations;
  // ordinary relational pointer comparisons would not provide that guarantee.
  const std::less<const uint16_t*> before;
  if (before(original.data(), destination.data() + destination.size()) &&
      before(destination.data(), original.data() + original.size())) {
    return absl::InvalidArgumentError("head-context arrays must not overlap");
  }
  if (!std::all_of(original.begin(), original.end(), Finite))
    return absl::InvalidArgumentError("nonfinite original BF16 context");
  std::copy(original.begin(), original.end(), destination.begin());
  for (int query = 0; query < geometry.context_length; ++query) {
    const bool selected_query = query == selection.query_position;
    if ((selection.scope == QueryScope::kSelectedQuery && !selected_query) ||
        (selection.scope == QueryScope::kOtherQueries && selected_query)) {
      continue;
    }
    const size_t row =
        static_cast<size_t>(selection.sequence) * geometry.context_length +
        query;
    const size_t offset = row * width + static_cast<size_t>(selection.head) *
                                            geometry.head_dimension;
    for (int lane = 0; lane < geometry.head_dimension; ++lane) {
      destination[offset + lane] =
          ScaleValidated(original[offset + lane], selection.scale);
    }
  }
  return absl::OkStatus();
}

}  // namespace pluto::weight_analysis::head_context
