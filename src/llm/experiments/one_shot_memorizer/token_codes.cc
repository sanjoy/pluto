#include "src/llm/experiments/one_shot_memorizer/token_codes.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/status/status.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

// All arithmetic deliberately wraps modulo 2^64. Specifying the generator
// and rejection rule keeps code assignments reproducible across platforms.
class SplitMix64 {
 public:
  explicit SplitMix64(uint64_t seed) : state_(seed) {}

  uint64_t Bounded(uint64_t bound) {
    // Removing the first 2^64 % bound values leaves an exact multiple of bound.
    const uint64_t threshold = (uint64_t{0} - bound) % bound;
    uint64_t value;
    do {
      value = Next();
    } while (value < threshold);
    return value % bound;
  }

 private:
  uint64_t Next() {
    uint64_t value = (state_ += UINT64_C(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
  }

  uint64_t state_;
};

size_t BalancedCapacity(int width) {
  size_t combinations = 1;
  for (int chosen = 1; chosen <= width / 2; ++chosen)
    combinations = combinations * (width - chosen + 1) / chosen;
  return combinations;
}

absl::Status ValidateCodes(const TokenCodes& codes) {
  if (codes.width <= 0 || codes.vocab_size <= 0)
    return absl::InvalidArgumentError("Token code dimensions must be positive");
  const size_t width = codes.width, vocabulary = codes.vocab_size;
  if (vocabulary > std::numeric_limits<size_t>::max() / width ||
      codes.values.size() != vocabulary * width)
    return absl::InvalidArgumentError(
        "Token code storage does not match shape");
  for (float value : codes.values)
    if (!std::isfinite(value))
      return absl::InvalidArgumentError("Token codes must be finite");
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<TokenCodes> MakeBalancedTokenCodes(int vocab_size, int width,
                                                  uint64_t seed) {
  if (width < 2 || width > 20 || width % 2 != 0)
    return absl::InvalidArgumentError(
        "Balanced code width must be even and between 2 and 20");
  const size_t capacity = BalancedCapacity(width);
  if (vocab_size <= 0 || static_cast<size_t>(vocab_size) > capacity)
    return absl::InvalidArgumentError(
        "Vocabulary exceeds balanced code capacity or is nonpositive");

  std::vector<uint32_t> masks;
  masks.reserve(capacity);
  const uint32_t end = uint32_t{1} << width;
  for (uint32_t mask = 0; mask < end; ++mask)
    if (std::popcount(mask) == width / 2)
      masks.push_back(mask);
  SplitMix64 random(seed);
  for (size_t remaining = masks.size(); remaining > 1; --remaining)
    std::swap(masks[remaining - 1], masks[random.Bounded(remaining)]);

  TokenCodes codes{.width = width, .vocab_size = vocab_size};
  codes.values.reserve(static_cast<size_t>(vocab_size) * width);
  for (int token = 0; token < vocab_size; ++token)
    for (int coordinate = 0; coordinate < width; ++coordinate)
      codes.values.push_back(
          masks[token] & (uint32_t{1} << coordinate) ? 1.0f : -1.0f);
  return codes;
}

absl::StatusOr<TokenCodes> NormalizeTokenCodes(absl::Span<const float> input,
                                               int width) {
  if (width <= 0 || input.empty() || input.size() % width != 0)
    return absl::InvalidArgumentError(
        "Token code input must be a nonempty complete matrix with positive "
        "width");
  const size_t vocabulary = input.size() / width;
  if (vocabulary > static_cast<size_t>(std::numeric_limits<int>::max()))
    return absl::InvalidArgumentError(
        "Token code vocabulary exceeds int range");
  TokenCodes codes{.width = width, .vocab_size = static_cast<int>(vocabulary)};
  codes.values.resize(input.size());
  for (size_t token = 0; token < vocabulary; ++token) {
    const size_t base = token * width;
    double mean = 0;
    bool nonconstant = false;
    for (int coordinate = 0; coordinate < width; ++coordinate) {
      const float value = input[base + coordinate];
      if (!std::isfinite(value))
        return absl::InvalidArgumentError("Token codes must be finite");
      nonconstant |= value != input[base];
      mean += value;
    }
    if (!nonconstant)
      return absl::InvalidArgumentError("Token code rows must not be constant");
    mean /= width;
    double variance = 0;
    for (int coordinate = 0; coordinate < width; ++coordinate) {
      const double centered =
          static_cast<double>(input[base + coordinate]) - mean;
      variance += centered * centered;
    }
    const double rms = std::sqrt(variance / width);
    if (rms == 0 || !std::isfinite(rms))
      return absl::InvalidArgumentError(
          "Every token code row must have finite nonzero centered RMS");
    for (int coordinate = 0; coordinate < width; ++coordinate)
      codes.values[base + coordinate] = static_cast<float>(
          (static_cast<double>(input[base + coordinate]) - mean) / rms);
  }
  return codes;
}

absl::StatusOr<TokenCodes> PermuteTokenCodes(const TokenCodes& codes,
                                             uint64_t seed) {
  const auto status = ValidateCodes(codes);
  if (!status.ok())
    return status;
  TokenCodes permuted = codes;
  SplitMix64 random(seed);
  for (size_t remaining = codes.vocab_size; remaining > 1; --remaining) {
    const size_t other = random.Bounded(remaining);
    for (int coordinate = 0; coordinate < codes.width; ++coordinate)
      std::swap(permuted.values[(remaining - 1) * codes.width + coordinate],
                permuted.values[other * codes.width + coordinate]);
  }
  return permuted;
}

}  // namespace pluto::llm::one_shot_memorizer
