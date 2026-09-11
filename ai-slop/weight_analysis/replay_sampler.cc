// Conditional replay of the CURRENT random-window sampler in dataset.cc.
//
// This program reads no corpus, checkpoint, model, or historical training log.
// Each row is a seed followed by sequence_count successive random starts. For
// ten steps with one or ten sequences per batch, request 10 or 100 starts.
// Both inputs and one-token-shifted targets fit at each reported start.
//
// Although mt19937_64 has a standardized stream, uniform_int_distribution's
// mapping is implementation-dependent. Record --implementation and the binary
// hash with any results. Matching today's sampler is NOT authentication of a
// historical seed, corpus, batch size, restart, or training example.

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string_view>
#include <system_error>

namespace {

constexpr std::string_view kUsage =
    "Usage: replay_sampler total_tokens context sequence_count seed_start "
    "seed_count\n"
    "All arguments are unsigned decimal integers. Counts are positive.\n"
    "Each output row: seed start_0 ... start_(sequence_count-1).\n"
    "Limits: sequence_count <= 10000, seed_count <= 10000, and their product "
    "<= 1000000.\n"
    "Use --implementation to identify the current C++ runtime. This is "
    "conditional replay, not authenticated historical sampling.\n";

bool ParseUnsigned(std::string_view text, uint64_t& result) {
  // Reject signs, spaces, suffixes, and overflow, rather than accepting a
  // partially parsed or wrapped bound and emitting plausible-looking starts.
  if (text.empty())
    return false;
  for (char character : text)
    if (character < '0' || character > '9')
      return false;
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), result);
  return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

int Error(std::string_view message) {
  std::cerr << "replay_sampler: " << message << '\n' << kUsage;
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    std::cout << kUsage;
    return 0;
  }
  if (argc == 2 && std::string_view(argv[1]) == "--implementation") {
    std::cout << "cxx=" << __cplusplus
              << " size_t_bits=" << std::numeric_limits<size_t>::digits;
#if defined(__GLIBCXX__)
    std::cout << " stdlib=libstdc++ stdlib_date=" << __GLIBCXX__;
#if defined(_GLIBCXX_RELEASE)
    std::cout << " stdlib_release=" << _GLIBCXX_RELEASE;
#endif
#elif defined(_LIBCPP_VERSION)
    std::cout << " stdlib=libc++ stdlib_version=" << _LIBCPP_VERSION;
#else
    std::cout << " stdlib=unidentified";
#endif
    std::cout << '\n';
    return 0;
  }
  if (argc != 6)
    return Error("expected exactly five positional arguments");
  uint64_t values[5] = {};
  for (int index = 0; index < 5; ++index)
    if (!ParseUnsigned(argv[index + 1], values[index]))
      return Error("argument is not an unsigned decimal uint64");
  const auto [total_tokens, context, sequence_count, seed_start, seed_count] =
      values;
  if (context == 0 || context > std::numeric_limits<int>::max())
    return Error("context must be positive and fit dataset's int context");
  if (total_tokens <= context ||
      total_tokens > std::numeric_limits<size_t>::max() / sizeof(int)) {
    return Error("token count must exceed context and fit dataset storage");
  }
  if (sequence_count == 0 || sequence_count > 10000 || seed_count == 0 ||
      seed_count > 10000 || sequence_count > 1000000 / seed_count) {
    return Error("requested counts exceed the positive bounded-output limits");
  }
  if (seed_start > std::numeric_limits<uint64_t>::max() - (seed_count - 1))
    return Error("last seed would overflow uint64");

  std::ios::sync_with_stdio(false);
  // The bounds and engine/distribution types intentionally match dataset.cc.
  // Sequence draws are consecutive across batches; a new seed starts a new
  // independent engine and distribution, just as a reset/new iterator would.
  const size_t maximum_start = static_cast<size_t>(total_tokens - context - 1);
  for (uint64_t index = 0; index < seed_count; ++index) {
    const uint64_t seed = seed_start + index;
    std::mt19937_64 random(seed);
    std::uniform_int_distribution<size_t> random_start(0, maximum_start);
    std::cout << seed;
    for (uint64_t sequence = 0; sequence < sequence_count; ++sequence)
      std::cout << ' ' << random_start(random);
    std::cout << '\n';
  }
  std::cout.flush();
  return std::cout ? 0 : 1;
}
