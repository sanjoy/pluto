#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "src/common/status_macros.h"
#include "src/llm/layer.h"

namespace pluto::llm::reference_internal {

inline constexpr int kDenseTile = 16;

inline int RoundUpToTile(int extent) {
  return ((extent + kDenseTile - 1) / kDenseTile) * kDenseTile;
}

inline size_t ActivationElementBytes(DataType type) {
  return type == DataType::BF16 ? sizeof(uint16_t) : sizeof(float);
}

inline absl::Status ValidateComputeType(DataType type) {
  if (type == DataType::FP16 || type == DataType::BF16) {
    return absl::OkStatus();
  }
  // FP8 arithmetic is meaningless without the same per-tensor scale used by
  // the device kernel. The production backend rejects FP8 for that reason, so
  // the reference rejects it too instead of inventing a different contract.
  return absl::UnimplementedError(
      "FP8 reference requires the backend's explicit scaling policy");
}

inline absl::Status ValidateTiledExtent(int extent, const char* name) {
  if (extent <= 0 || extent % kDenseTile != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        name, " must be a positive multiple of ", kDenseTile));
  }
  return absl::OkStatus();
}

inline absl::Status ValidateBuffer(const HostBuffer& buffer,
                                   size_t expected_bytes,
                                   const char* name) {
  if (buffer.size_bytes() != expected_bytes) {
    return absl::InvalidArgumentError(absl::StrCat(
        name, " has ", buffer.size_bytes(), " bytes; expected ",
        expected_bytes));
  }
  return absl::OkStatus();
}

inline absl::StatusOr<int> ElementCount(const HostBuffer& buffer,
                                        size_t element_bytes,
                                        const char* name) {
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

inline absl::StatusOr<int> MatrixRows(const HostBuffer& buffer, int columns,
                                      const char* name) {
  if (columns <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, " has an invalid column count"));
  }
  ASSIGN_OR_RETURN(int elements,
                   ElementCount(buffer, sizeof(float), name));
  if (elements % columns != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, " is not a float matrix"));
  }
  return elements / columns;
}

inline absl::StatusOr<int> ActivationRows(const HostBuffer& buffer,
                                          int columns, DataType type,
                                          const char* name) {
  if (columns <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, " has an invalid column count"));
  }
  ASSIGN_OR_RETURN(
      int elements,
      ElementCount(buffer, ActivationElementBytes(type), name));
  if (elements % columns != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(name, " is not an activation matrix"));
  }
  return elements / columns;
}

inline uint16_t FloatToBf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  // Round to nearest, ties to even. CUDA's BF16 conversion uses this rounding
  // mode, so storing this 16-bit payload reproduces device activation bytes.
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

inline float Bf16ToFloat(uint16_t value) {
  return std::bit_cast<float>(static_cast<uint32_t>(value) << 16);
}

inline uint16_t FloatToHalf(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t exponent = (bits >> 23) & 0xffu;
  uint32_t mantissa = bits & 0x7fffffu;

  if (exponent == 0xffu) {
    return static_cast<uint16_t>(
        sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
  }

  int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) return static_cast<uint16_t>(sign);
    mantissa |= 0x800000u;
    const int shift = 14 - half_exponent;
    uint32_t rounded = mantissa >> shift;
    const uint32_t remainder = mantissa & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1);
    if (remainder > halfway ||
        (remainder == halfway && (rounded & 1u) != 0)) {
      ++rounded;
    }
    return static_cast<uint16_t>(sign | rounded);
  }

  uint32_t rounded_mantissa = mantissa >> 13;
  const uint32_t remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u ||
      (remainder == 0x1000u && (rounded_mantissa & 1u) != 0)) {
    ++rounded_mantissa;
    if (rounded_mantissa == 0x400u) {
      rounded_mantissa = 0;
      ++half_exponent;
      if (half_exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
      }
    }
  }
  return static_cast<uint16_t>(
      sign | (static_cast<uint32_t>(half_exponent) << 10) |
      rounded_mantissa);
}

inline float HalfToFloat(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
  uint32_t exponent = (value >> 10) & 0x1fu;
  uint32_t mantissa = value & 0x3ffu;
  uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int unbiased = -14;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        --unbiased;
      }
      mantissa &= 0x3ffu;
      bits = sign | (static_cast<uint32_t>(unbiased + 127) << 23) |
             (mantissa << 13);
    }
  } else if (exponent == 0x1fu) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
  }
  return std::bit_cast<float>(bits);
}

inline float QuantizeMmaOperand(float value, DataType type) {
  return type == DataType::BF16
             ? Bf16ToFloat(FloatToBf16(value))
             : HalfToFloat(FloatToHalf(value));
}

inline float LoadActivation(const HostBuffer& buffer, size_t index,
                            DataType type) {
  if (type == DataType::BF16) {
    return Bf16ToFloat(static_cast<const uint16_t*>(buffer.data())[index]);
  }
  return static_cast<const float*>(buffer.data())[index];
}

inline void StoreActivation(HostBuffer* buffer, size_t index, DataType type,
                            float value) {
  if (type == DataType::BF16) {
    static_cast<uint16_t*>(buffer->data())[index] = FloatToBf16(value);
  } else {
    static_cast<float*>(buffer->data())[index] = value;
  }
}

inline absl::StatusOr<HostBuffer> AllocateActivation(size_t elements,
                                                     DataType type) {
  return HostBuffer::Allocate(elements * ActivationElementBytes(type));
}

inline absl::StatusOr<HostBuffer> AllocateFloats(size_t elements,
                                                 bool clear = false) {
  ASSIGN_OR_RETURN(auto buffer,
                   HostBuffer::Allocate(elements * sizeof(float)));
  if (clear) std::memset(buffer.data(), 0, buffer.size_bytes());
  return buffer;
}

}  // namespace pluto::llm::reference_internal
