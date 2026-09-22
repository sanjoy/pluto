#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_io.h"

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <bit>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <system_error>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "src/util/status_macros.h"

extern char** environ;

namespace pluto::llm::discretized::generator {
namespace {

absl::Status IoError(absl::string_view operation,
                     const std::filesystem::path& path, int error) {
  return absl::UnknownError(
      absl::StrCat(operation, " ", path.string(), ": ", std::strerror(error)));
}

// SHA-256 processes 512-bit blocks with big-endian words. Keeping the small
// digest implementation here avoids a crypto-library dependency in this tool;
// it identifies artifacts, and is not used for authentication or encryption.
class Sha256Digest {
 public:
  void Add(absl::string_view input) {
    bytes_ += input.size();
    for (unsigned char byte : input) {
      block_[used_++] = byte;
      if (used_ == block_.size()) {
        Compress();
        used_ = 0;
      }
    }
  }

  std::string Finish() {
    uint64_t bits = bytes_ * 8;
    block_[used_++] = 0x80;
    if (used_ > 56) {
      while (used_ != 64)
        block_[used_++] = 0;
      Compress();
      used_ = 0;
    }
    while (used_ < 56)
      block_[used_++] = 0;
    for (int i = 7; i >= 0; --i)
      block_[used_++] = static_cast<unsigned char>(bits >> (i * 8));
    Compress();
    constexpr char kHex[] = "0123456789abcdef";
    std::string result;
    for (uint32_t word : state_)
      for (int i = 7; i >= 0; --i)
        result.push_back(kHex[(word >> (4 * i)) & 15]);
    return result;
  }

 private:
  void Compress() {
    static constexpr uint32_t kRound[] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t words[64];
    for (int i = 0; i < 16; ++i)
      words[i] = (uint32_t{block_[i * 4]} << 24) |
                 (uint32_t{block_[i * 4 + 1]} << 16) |
                 (uint32_t{block_[i * 4 + 2]} << 8) | block_[i * 4 + 3];
    for (int i = 16; i < 64; ++i) {
      uint32_t a = words[i - 15], b = words[i - 2];
      words[i] =
          words[i - 16] + (std::rotr(a, 7) ^ std::rotr(a, 18) ^ (a >> 3)) +
          words[i - 7] + (std::rotr(b, 17) ^ std::rotr(b, 19) ^ (b >> 10));
    }
    auto [a, b, c, d, e, f, g, h] = state_;
    for (int i = 0; i < 64; ++i) {
      uint32_t first = h +
                       (std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25)) +
                       ((e & f) ^ (~e & g)) + kRound[i] + words[i];
      uint32_t second =
          (std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22)) +
          ((a & b) ^ (a & c) ^ (b & c));
      h = g;
      g = f;
      f = e;
      e = d + first;
      d = c;
      c = b;
      b = a;
      a = first + second;
    }
    std::array<uint32_t, 8> updated = {a, b, c, d, e, f, g, h};
    for (int i = 0; i < 8; ++i)
      state_[i] += updated[i];
  }

  std::array<uint32_t, 8> state_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                    0xa54ff53a, 0x510e527f, 0x9b05688c,
                                    0x1f83d9ab, 0x5be0cd19};
  std::array<unsigned char, 64> block_{};
  size_t used_ = 0;
  uint64_t bytes_ = 0;
};

}  // namespace

absl::StatusOr<std::string> ReadFile(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    return IoError("cannot read", path, errno);
  // Read through the stream so IO failures set badbit. Reading directly through
  // istreambuf_iterator can throw inside filebuf on errors such as EISDIR.
  std::string result;
  std::array<char, 65536> bytes;
  while (stream) {
    stream.read(bytes.data(), bytes.size());
    result.append(bytes.data(), stream.gcount());
  }
  if (stream.bad())
    return IoError("read failed", path, EIO);
  return result;
}

absl::Status WriteFile(const std::filesystem::path& path,
                       absl::string_view data) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream)
    return IoError("cannot write", path, errno);
  stream.write(data.data(), data.size());
  stream.close();
  if (!stream)
    return IoError("write failed", path, EIO);
  return absl::OkStatus();
}

std::string TextReport(const Json& value) {
  std::string result;
  auto append = [&](auto& self, const Json& item, int indent) -> void {
    if (item.is_object()) {
      for (const auto& [key, child] : item.items()) {
        absl::StrAppend(&result, std::string(indent, ' '), key, ":");
        if (child.is_structured()) {
          result += '\n';
          self(self, child, indent + 2);
        } else {
          absl::StrAppend(&result, " ", child.dump(), "\n");
        }
      }
    } else if (item.is_array()) {
      for (const auto& child : item) {
        if (child.is_structured()) {
          absl::StrAppend(&result, std::string(indent, ' '), "-\n");
          self(self, child, indent + 2);
        } else {
          absl::StrAppend(&result, std::string(indent, ' '), "- ", child.dump(),
                          "\n");
        }
      }
    } else {
      absl::StrAppend(&result, std::string(indent, ' '), item.dump(), "\n");
    }
  };
  append(append, value, 0);
  return result;
}

std::string Sha256(absl::string_view data) {
  Sha256Digest digest;
  digest.Add(data);
  return digest.Finish();
}

absl::StatusOr<std::string> Sha256File(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    return IoError("cannot hash", path, errno);
  Sha256Digest digest;
  std::array<char, 65536> bytes;
  while (stream) {
    stream.read(bytes.data(), bytes.size());
    digest.Add(absl::string_view(bytes.data(), stream.gcount()));
  }
  if (stream.bad())
    return IoError("hash read failed", path, EIO);
  return digest.Finish();
}

absl::StatusOr<std::filesystem::path> FindExecutable(absl::string_view name) {
  const char* path = std::getenv("PATH");
  if (path != nullptr)
    for (absl::string_view directory : absl::StrSplit(path, ':')) {
      auto candidate =
          std::filesystem::path(std::string(directory)) / std::string(name);
      std::error_code error;
      if (std::filesystem::is_regular_file(candidate, error) &&
          access(candidate.c_str(), X_OK) == 0) {
        candidate = std::filesystem::absolute(candidate, error);
        if (!error)
          return candidate;
      }
    }
  return absl::NotFoundError(
      absl::StrCat(name, " must be installed and executable on PATH"));
}

absl::Status RunProcess(const std::vector<std::string>& arguments) {
  if (arguments.empty())
    return absl::InvalidArgumentError("cannot launch an empty command");
  std::vector<char*> argv;
  for (const auto& argument : arguments)
    argv.push_back(const_cast<char*>(argument.c_str()));
  argv.push_back(nullptr);
  pid_t pid;
  int error =
      posix_spawnp(&pid, argv[0], nullptr, nullptr, argv.data(), environ);
  if (error)
    return IoError("cannot launch", arguments.front(), error);
  int result;
  while (waitpid(pid, &result, 0) == -1)
    if (errno != EINTR)
      return IoError("cannot wait for", arguments.front(), errno);
  if (!WIFEXITED(result) || WEXITSTATUS(result) != 0)
    return absl::InternalError(
        absl::StrCat(arguments.front(), " failed (wait status ", result, ")"));
  return absl::OkStatus();
}
}  // namespace pluto::llm::discretized::generator
