#ifndef PLUTO_SRC_LLM_LAYERS_INTERNAL_H_
#define PLUTO_SRC_LLM_LAYERS_INTERNAL_H_

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/buffer.h"
#include "src/llm/layer.h"

namespace pluto::llm::internal {

// Matrix math uses 16-wide cuTile MMA operands. Tensor extents remain runtime
// values; this is a compute-tile constraint, not a model-shape specialization.
inline constexpr int kDenseTile = 16;

inline int TileCount(int extent) { return extent / kDenseTile; }

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
    return absl::InvalidArgumentError(absl::StrCat(
        name, " must be a positive multiple of ", kDenseTile, "; got ",
        extent));
  }
  return absl::OkStatus();
}

inline absl::StatusOr<int> MatrixRows(const Buffer& buffer, int columns,
                                      cudaStream_t stream,
                                      const char* name) {
  if (buffer.stream() != stream) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, " belongs to a different CUDA stream"));
  }
  const size_t row_bytes = static_cast<size_t>(columns) * sizeof(float);
  if (columns <= 0 || buffer.size_bytes() == 0 ||
      buffer.size_bytes() % row_bytes != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        name, " is not a non-empty float matrix with ", columns,
        " columns"));
  }
  const size_t rows = buffer.size_bytes() / row_bytes;
  if (rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return absl::InvalidArgumentError(absl::StrCat(name, " has too many rows"));
  }
  return static_cast<int>(rows);
}

inline absl::StatusOr<int> ActivationRows(const Buffer& buffer, int columns,
                                          DataType data_type,
                                          cudaStream_t stream,
                                          const char* name) {
  if (buffer.stream() != stream) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, " belongs to a different CUDA stream"));
  }
  const size_t row_bytes =
      static_cast<size_t>(columns) * ActivationElementBytes(data_type);
  if (columns <= 0 || buffer.size_bytes() == 0 ||
      buffer.size_bytes() % row_bytes != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        name, " is not a non-empty activation matrix with ", columns,
        " columns"));
  }
  const size_t rows = buffer.size_bytes() / row_bytes;
  if (rows > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return absl::InvalidArgumentError(absl::StrCat(name, " has too many rows"));
  }
  return static_cast<int>(rows);
}

inline absl::StatusOr<int> ElementCount(const Buffer& buffer,
                                        size_t element_bytes,
                                        cudaStream_t stream,
                                        const char* name) {
  if (buffer.stream() != stream) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, " belongs to a different CUDA stream"));
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

inline absl::Status CudaStatus(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) return absl::OkStatus();
  return absl::InternalError(
      absl::StrCat(operation, " failed: ", cudaGetErrorName(error), ": ",
                   cudaGetErrorString(error)));
}

inline absl::Status ValidateComputeType(DataType data_type) {
  if (data_type == DataType::FP16 || data_type == DataType::BF16) {
    return absl::OkStatus();
  }
  return absl::UnimplementedError(
      "FP8 requires an explicit scaling policy; this cuTile backend currently "
      "implements FP16 and BF16 compute with FP32 master weights");
}

// Compatibility spelling for call sites which have not yet become
// dtype-polymorphic.
inline absl::Status ValidateFp16(DataType data_type) {
  return ValidateComputeType(data_type);
}

inline absl::Status ValidateBuffer(const Buffer& buffer,
                                   size_t expected_bytes,
                                   cudaStream_t stream, const char* name) {
  if (buffer.size_bytes() != expected_bytes) {
    return absl::InvalidArgumentError(absl::StrCat(
        name, " has ", buffer.size_bytes(), " bytes; expected ",
        expected_bytes));
  }
  if (buffer.stream() != stream) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, " belongs to a different CUDA stream"));
  }
  return absl::OkStatus();
}

}  // namespace pluto::llm::internal

#endif  // PLUTO_SRC_LLM_LAYERS_INTERNAL_H_
