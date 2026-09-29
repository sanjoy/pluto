#pragma once

#include <cstddef>

#include "absl/status/statusor.h"

namespace pluto::llm {

// Physical storage of an imported checkpoint matrix, independent of the
// activation compute type. FP8 matrices require separate FP32 block scales.
enum class MatrixStorage { kBFloat16, kFloat32, kFp8E4M3 };

// Returns the bytes per physical matrix element; rejects unknown storage.
absl::StatusOr<size_t> MatrixElementBytes(MatrixStorage storage);

}  // namespace pluto::llm
