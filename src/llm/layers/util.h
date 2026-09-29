#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/buffer.h"
#include "src/cuda/executor.h"
#include "src/llm/layer.h"

namespace pluto::llm::internal {

// Shared activation conversions used by imported projections, normalization,
// and cached attention. Each result owns fresh storage on executor; conversion
// is stream ordered and does not synchronize. Layers enforce their own shape
// limits; these helpers only require a positive, exactly sized vector.
absl::StatusOr<Buffer> AllocateFloatVector(cuda::Executor& executor,
                                           int elements);
absl::StatusOr<Buffer> ToFloat(cuda::Executor& executor, const Buffer& input,
                               int elements);
absl::StatusOr<Buffer> ToBFloat16(cuda::Executor& executor, const Buffer& input,
                                  int elements);

// Checks input count, positive element counts, BF16 byte sizes, and executor
// ownership before any layer kernel is launched.
absl::Status ValidateBFloat16Inputs(cuda::Executor& executor,
                                    absl::Span<const Buffer> inputs,
                                    absl::Span<const int> element_counts);

// Matrix math uses 16-wide cuTile MMA operands. Tensor extents remain runtime
// values; this is a compute-tile constraint, not a model-shape specialization.
inline constexpr int kDenseTile = 16;

inline int TileCount(int extent) { return extent / kDenseTile; }

// Use only for kernels whose loads and stores mask the final partial tile.
// The quotient/remainder form avoids overflow for a positive INT_MAX extent.
inline int MaskedTileCount(int extent) {
  return extent / kDenseTile + (extent % kDenseTile != 0);
}

inline int RoundUpToTile(int extent) {
  return ((extent + kDenseTile - 1) / kDenseTile) * kDenseTile;
}

inline size_t ActivationElementBytes(DataType data_type) {
  // The legacy FP16 path retains FP32 activation storage for compatibility.
  // BF16 models use compact BF16 activation storage; gradients and master
  // parameters remain FP32 in both modes.
  return data_type == DataType::BF16 ? 2 : sizeof(float);
}

inline absl::Status ValidateTiledExtent(int extent, const char* name) {
  if (extent <= 0 || extent % kDenseTile != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, " must be a positive multiple of ", kDenseTile,
                     "; got ", extent));
  }
  return absl::OkStatus();
}

inline absl::Status ValidatePositiveExtent(int extent, const char* name) {
  if (extent <= 0)
    return absl::InvalidArgumentError(
        absl::StrCat(name, " must be positive; got ", extent));
  return absl::OkStatus();
}

inline absl::Status ValidateExecutor(cuda::Executor& expected,
                                     cuda::Executor& actual, const char* name) {
  if (&actual != &expected) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, " was created for a different CUDA Executor"));
  }
  return absl::OkStatus();
}

inline absl::StatusOr<int> MatrixRows(cuda::Executor& executor,
                                      const Buffer& buffer, int columns,
                                      const char* name) {
  if (&buffer.executor() != &executor) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, " belongs to a different CUDA Executor"));
  }
  const size_t row_bytes = static_cast<size_t>(columns) * sizeof(float);
  if (columns <= 0 || buffer.size_bytes() == 0 ||
      buffer.size_bytes() % row_bytes != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        name, " is not a non-empty float matrix with ", columns, " columns"));
  }
  const size_t rows = buffer.size_bytes() / row_bytes;
  if (rows > static_cast<size_t>(std::numeric_limits<int>::max()))
    return absl::InvalidArgumentError(absl::StrCat(name, " has too many rows"));
  return static_cast<int>(rows);
}

inline absl::StatusOr<int> ActivationRows(cuda::Executor& executor,
                                          const Buffer& buffer, int columns,
                                          DataType data_type,
                                          const char* name) {
  if (&buffer.executor() != &executor) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, " belongs to a different CUDA Executor"));
  }
  const size_t row_bytes =
      static_cast<size_t>(columns) * ActivationElementBytes(data_type);
  if (columns <= 0 || buffer.size_bytes() == 0 ||
      buffer.size_bytes() % row_bytes != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, " is not a non-empty activation matrix with ",
                     columns, " columns"));
  }
  const size_t rows = buffer.size_bytes() / row_bytes;
  if (rows > static_cast<size_t>(std::numeric_limits<int>::max()))
    return absl::InvalidArgumentError(absl::StrCat(name, " has too many rows"));
  return static_cast<int>(rows);
}

inline absl::StatusOr<int> ElementCount(cuda::Executor& executor,
                                        const Buffer& buffer,
                                        size_t element_bytes,
                                        const char* name) {
  if (&buffer.executor() != &executor) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, " belongs to a different CUDA Executor"));
  }
  if (buffer.size_bytes() == 0 || buffer.size_bytes() % element_bytes != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, " has an invalid byte size"));
  }
  const size_t count = buffer.size_bytes() / element_bytes;
  if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, " has too many elements"));
  }
  return static_cast<int>(count);
}

inline absl::Status ValidateComputeType(DataType data_type) {
  if (data_type == DataType::FP16 || data_type == DataType::BF16)
    return absl::OkStatus();
  if (data_type != DataType::FP8)
    return absl::UnimplementedError(
        "this backend supports only FP16 and BF16 compute policies");
  return absl::UnimplementedError(
      "FP8 requires an explicit scaling policy; this cuTile backend currently "
      "implements FP16 and BF16 compute with FP32 master weights");
}

// Compatibility spelling for call sites which have not yet become
// dtype-polymorphic.
inline absl::Status ValidateFp16(DataType data_type) {
  return ValidateComputeType(data_type);
}

inline absl::Status ValidateBuffer(cuda::Executor& executor,
                                   const Buffer& buffer, size_t expected_bytes,
                                   const char* name) {
  if (buffer.size_bytes() != expected_bytes) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, " has ", buffer.size_bytes(), " bytes; expected ",
                     expected_bytes));
  }
  if (&buffer.executor() != &executor) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, " belongs to a different CUDA Executor"));
  }
  return absl::OkStatus();
}

}  // namespace pluto::llm::internal
