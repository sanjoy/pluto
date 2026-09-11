// Export the exact token stream that Pluto training sees, not a substitute
// implementation from another GPT-2 tokenizer library. In particular, the
// existing native whitespace pre-tokenizer can differ from Hugging Face's.
// This analysis tool deliberately does not change that training behavior.
//
// Usage: tokenize_corpus TOKENIZER_DIR CORPUS OUTPUT.bin
// Outputs have no header and are explicitly little-endian on every host:
//   OUTPUT.bin:             uint32 token_ids[N]
//   OUTPUT.bin.offsets.bin: uint64 byte_offsets[N + 1]
// Token i decodes exactly to corpus[byte_offsets[i]:byte_offsets[i + 1]].
// Offsets count raw bytes, not Unicode characters: a token may split UTF-8.
// Both files must be absent. The tool validates the whole-corpus round trip
// and every individual token's bytes before creating either output file.
//
// --self-test TOKENIZER_DIR checks whitespace, UTF-8, EOS, and empty-input
// round trips using the same native code, without creating output files.
// No transformer, GPU kernel, checkpoint, or model inference is used. The
// production tokenizer does allocate CUDA page-locked host memory for IDs.

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/gpt2_detokenizer.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/util/status_macros.h"

namespace pluto::weight_analysis {
namespace {

struct EncodedCorpus {
  cuda::PageLockedHostArray<int> tokens;
  std::vector<uint64_t> byte_offsets;
};

absl::StatusOr<EncodedCorpus> EncodeAndValidate(
    cuda::Executor& executor, const tokenizer::Gpt2Tokenizer& encoder,
    const tokenizer::Gpt2Detokenizer& decoder, const std::string& corpus) {
  ASSIGN_OR_RETURN(auto tokens, encoder.Encode(executor, corpus));
  ASSIGN_OR_RETURN(auto decoded, decoder.Decode(tokens.span()));
  if (decoded != corpus) {
    return absl::DataLossError(
        "native tokenizer round trip changed corpus bytes");
  }

  // Cache decoded vocabulary entries, so frequent tokens are not repeatedly
  // decoded. An empty cache entry is safe because GPT-2 tokens have >=1 byte.
  std::vector<std::string> token_bytes(encoder.vocab_size());
  std::vector<uint64_t> offsets;
  offsets.reserve(tokens.size() + 1);
  offsets.push_back(0);
  for (const int token : tokens) {
    if (token < 0 || token >= encoder.vocab_size())
      return absl::DataLossError("native encoder returned an invalid token ID");
    if (token_bytes[token].empty()) {
      ASSIGN_OR_RETURN(token_bytes[token],
                       decoder.Decode(absl::MakeConstSpan(&token, 1)));
    }
    const std::string& bytes = token_bytes[token];
    const uint64_t begin = offsets.back();
    if (bytes.empty() || begin > corpus.size() ||
        bytes.size() > corpus.size() - begin ||
        corpus.compare(begin, bytes.size(), bytes) != 0) {
      return absl::DataLossError(
          "individual token bytes do not match the corpus at their offset");
    }
    offsets.push_back(begin + bytes.size());
  }
  if (offsets.back() != corpus.size()) {
    return absl::DataLossError(
        "token byte offsets do not span the whole corpus");
  }
  return EncodedCorpus{std::move(tokens), std::move(offsets)};
}

// Each new file is deleted on failure, but a preexisting path is never opened
// for writing or removed. O_EXCL enforces this even in the presence of races.
class NewOutput final {
 public:
  explicit NewOutput(std::filesystem::path path) : path_(std::move(path)) {}
  ~NewOutput() {
    if (descriptor_ >= 0)
      close(descriptor_);
    if (created_ && !committed_)
      unlink(path_.c_str());
  }
  NewOutput(const NewOutput&) = delete;
  NewOutput& operator=(const NewOutput&) = delete;

  absl::Status Open() {
    descriptor_ =
        open(path_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (descriptor_ < 0) {
      return absl::FailedPreconditionError(
          absl::StrCat("cannot exclusively create ", path_.string(), ": ",
                       std::strerror(errno)));
    }
    created_ = true;
    return absl::OkStatus();
  }

  absl::Status Write(const std::vector<unsigned char>& bytes) {
    size_t position = 0;
    while (position < bytes.size()) {
      const ssize_t written =
          write(descriptor_, bytes.data() + position, bytes.size() - position);
      if (written < 0 && errno == EINTR)
        continue;
      if (written <= 0) {
        return absl::InternalError(absl::StrCat("cannot write ", path_.string(),
                                                ": ", std::strerror(errno)));
      }
      position += static_cast<size_t>(written);
    }
    return absl::OkStatus();
  }

  absl::Status Close() {
    const int descriptor = std::exchange(descriptor_, -1);
    if (close(descriptor) != 0) {
      return absl::InternalError(absl::StrCat("cannot close ", path_.string(),
                                              ": ", std::strerror(errno)));
    }
    return absl::OkStatus();
  }

  void Commit() { committed_ = true; }

 private:
  std::filesystem::path path_;
  int descriptor_ = -1;
  bool created_ = false;
  bool committed_ = false;
};

// Serialize integer values instead of dumping native-endian arrays. The small
// reusable buffer bounds temporary memory even for a large training corpus.
template <class Range>
absl::Status WriteLittleEndian(NewOutput& output, const Range& values,
                               int width) {
  std::vector<unsigned char> bytes;
  constexpr size_t kChunkBytes = 65536;
  bytes.reserve(kChunkBytes);
  for (const auto value : values) {
    const uint64_t bits = static_cast<uint64_t>(value);
    for (int index = 0; index < width; ++index)
      bytes.push_back(static_cast<unsigned char>((bits >> (8 * index)) & 255));
    if (bytes.size() >= kChunkBytes) {
      RETURN_IF_ERROR(output.Write(bytes));
      bytes.clear();
    }
  }
  return output.Write(bytes);
}

absl::Status Export(const std::filesystem::path& tokenizer_directory,
                    const std::filesystem::path& corpus_path,
                    const std::filesystem::path& output_path) {
  // Own the host-memory pool before creating any token arrays.
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto encoder,
                   tokenizer::Gpt2Tokenizer::Load(tokenizer_directory));
  ASSIGN_OR_RETURN(auto decoder,
                   tokenizer::Gpt2Detokenizer::Load(tokenizer_directory));
  std::ifstream input(corpus_path, std::ios::binary);
  if (!input) {
    return absl::NotFoundError(
        absl::StrCat("cannot open corpus ", corpus_path.string()));
  }
  const std::string corpus((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
  if (input.bad())
    return absl::DataLossError("error reading corpus bytes");
  ASSIGN_OR_RETURN(auto encoded,
                   EncodeAndValidate(*executor, *encoder, *decoder, corpus));

  NewOutput ids(output_path);
  NewOutput offsets(absl::StrCat(output_path.string(), ".offsets.bin"));
  RETURN_IF_ERROR(ids.Open());
  RETURN_IF_ERROR(offsets.Open());
  RETURN_IF_ERROR(WriteLittleEndian(ids, encoded.tokens, 4));
  RETURN_IF_ERROR(WriteLittleEndian(offsets, encoded.byte_offsets, 8));
  RETURN_IF_ERROR(ids.Close());
  RETURN_IF_ERROR(offsets.Close());
  ids.Commit();
  offsets.Commit();
  std::cout
      << "{\"format\":\"pluto-native-token-ids-v1\",\"token_count\":"
      << encoded.tokens.size() << ",\"corpus_bytes\":" << corpus.size()
      << ",\"token_dtype\":\"<u4\",\"offset_dtype\":\"<u8\",\"offset_count\":"
      << encoded.byte_offsets.size() << ",\"roundtrip_verified\":true}\n";
  return absl::OkStatus();
}

absl::Status SelfTest(const std::filesystem::path& tokenizer_directory) {
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto encoder,
                   tokenizer::Gpt2Tokenizer::Load(tokenizer_directory));
  ASSIGN_OR_RETURN(auto decoder,
                   tokenizer::Gpt2Detokenizer::Load(tokenizer_directory));
  const std::vector<std::string> examples = {"",
                                             "a\n\nb",
                                             " \t\n word\r\n\r\nnext",
                                             "Hello, 🌍!\n",
                                             "a<|endoftext|>b",
                                             "trailing spaces   ",
                                             "  leading spaces"};
  for (const auto& example : examples) {
    ASSIGN_OR_RETURN(auto encoded,
                     EncodeAndValidate(*executor, *encoder, *decoder, example));
    if (example == "a\n\nb") {
      // Regression anchor for the native whitespace behavior that motivated
      // this exporter: HF currently emits two 198s here; native emits 628.
      const std::vector<int> expected = {64, 628, 65};
      if (std::vector<int>(encoded.tokens.begin(), encoded.tokens.end()) !=
          expected) {
        return absl::DataLossError("native a\\n\\nb token sequence changed");
      }
    }
  }
  std::cout << "Native tokenizer self-test: " << examples.size()
            << " exact token/byte round trips passed\n";
  return absl::OkStatus();
}

}  // namespace
}  // namespace pluto::weight_analysis

int main(int argc, char** argv) {
  absl::Status result;
  if (argc == 3 && std::string(argv[1]) == "--self-test") {
    result = pluto::weight_analysis::SelfTest(argv[2]);
  } else if (argc == 4) {
    result = pluto::weight_analysis::Export(argv[1], argv[2], argv[3]);
  } else {
    std::cerr << "Usage: tokenize_corpus TOKENIZER_DIR CORPUS OUTPUT.bin\n"
              << "       tokenize_corpus --self-test TOKENIZER_DIR\n";
    return 2;
  }
  if (!result.ok()) {
    std::cerr << result << '\n';
    return 1;
  }
  return 0;
}
