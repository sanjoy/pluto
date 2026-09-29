#include "src/llm/layers/matrix_common.h"

#include <cstdint>

#include "absl/status/status.h"

namespace pluto::llm {

absl::StatusOr<size_t> MatrixElementBytes(MatrixStorage storage) {
  switch (storage) {
    case MatrixStorage::kFloat32:
      return sizeof(float);
    case MatrixStorage::kBFloat16:
      return sizeof(uint16_t);
    case MatrixStorage::kFp8E4M3:
      return sizeof(uint8_t);
  }
  return absl::InvalidArgumentError("unknown matrix storage");
}

}  // namespace pluto::llm
