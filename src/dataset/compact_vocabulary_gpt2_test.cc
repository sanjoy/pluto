#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "src/cuda/executor.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/gpt2_detokenizer.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/tokenizer.h"

namespace pluto::tokenizer {
namespace {

std::filesystem::path CorpusPath() {
  const char* runfiles = std::getenv("TEST_SRCDIR");
  const char* workspace = std::getenv("TEST_WORKSPACE");
  if (runfiles != nullptr) {
    return std::filesystem::path(runfiles) /
           (workspace == nullptr ? "_main" : workspace) /
           "testdata/general_facts_dataset.txt";
  }
  return "testdata/general_facts_dataset.txt";
}

class CompactVocabularyGpt2Test : public testing::Test {
 protected:
  void SetUp() override {
    const char* directory = std::getenv("PLUTO_GPT2_TOKENIZER_DIR");
    ASSERT_NE(directory, nullptr)
        << "set PLUTO_GPT2_TOKENIZER_DIR to the saved GPT-2 tokenizer";
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
    auto tokenizer = Gpt2Tokenizer::Load(directory);
    ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();
    tokenizer_ = std::move(*tokenizer);
    auto detokenizer = Gpt2Detokenizer::Load(directory);
    ASSERT_TRUE(detokenizer.ok()) << detokenizer.status();
    detokenizer_ = std::move(*detokenizer);
  }

  void TearDown() override {
    if (executor_ != nullptr) {
      const auto status = executor_->Synchronize();
      EXPECT_TRUE(status.ok()) << status;
    }
    if (!temporary_directory_.empty()) {
      // Only the unique directory created by the serialization test.
      std::error_code error;
      std::filesystem::remove_all(temporary_directory_, error);
      EXPECT_FALSE(error) << error.message();
    }
  }

  int TokenId(absl::string_view text) {
    auto encoded = tokenizer_->Encode(*executor_, text);
    if (!encoded.ok() || encoded->size() != 1) {
      ADD_FAILURE() << "expected one GPT-2 token for " << text << ": "
                    << encoded.status();
      return -1;
    }
    return (*encoded)[0];
  }

  CompactVocabularyMapping Mapping(std::vector<int> retained) {
    retained.push_back(tokenizer_->eos_token_id());
    std::sort(retained.begin(), retained.end());
    retained.erase(std::unique(retained.begin(), retained.end()),
                   retained.end());
    std::vector<int> reverse(tokenizer_->vocab_size(), -1);
    for (int compact = 0; compact < static_cast<int>(retained.size());
         ++compact) {
      if (retained[compact] >= 0 &&
          retained[compact] < tokenizer_->vocab_size())
        reverse[retained[compact]] = compact;
    }
    return {std::move(retained), std::move(reverse),
            tokenizer_->eos_token_id()};
  }

  absl::StatusOr<std::unique_ptr<CompactVocabularyTokenizer>> Create(
      std::vector<int> retained) {
    return CompactVocabularyTokenizer::Create(*tokenizer_,
                                              Mapping(std::move(retained)));
  }

  std::vector<int> ByteTokens() {
    std::vector<int> ids(256);
    std::iota(ids.begin(), ids.end(), 0);
    std::array<bool, 256> seen{};
    for (int id : ids) {
      const auto decoded = detokenizer_->Decode(absl::Span<const int>(&id, 1));
      if (!decoded.ok() || decoded->size() != 1) {
        ADD_FAILURE() << "expected a single-byte token at " << id;
        return {};
      }
      const auto byte = static_cast<unsigned char>((*decoded)[0]);
      EXPECT_FALSE(seen[byte]);
      seen[byte] = true;
    }
    EXPECT_TRUE(
        std::all_of(seen.begin(), seen.end(), [](bool b) { return b; }));
    return ids;
  }

  // Every result must belong to the retained vocabulary, and detokenization
  // through original IDs must reproduce the input's exact bytes.
  absl::StatusOr<std::vector<int>> EncodeOriginal(
      const CompactVocabularyTokenizer& compact, absl::string_view text) {
    auto encoded = compact.Encode(*executor_, text);
    if (!encoded.ok())
      return encoded.status();
    EXPECT_EQ(&encoded->executor(), executor_.get());
    std::vector<int> originals;
    for (int id : encoded->span()) {
      auto original = compact.OriginalId(id);
      if (!original.ok())
        return original.status();
      EXPECT_EQ(*compact.CompactId(*original), id);
      originals.push_back(*original);
    }
    auto decoded = detokenizer_->Decode(originals);
    if (!decoded.ok())
      return decoded.status();
    EXPECT_EQ(*decoded, text);
    return originals;
  }

  std::unique_ptr<cuda::Executor> executor_;
  std::unique_ptr<Gpt2Tokenizer> tokenizer_;
  std::unique_ptr<Gpt2Detokenizer> detokenizer_;
  std::filesystem::path temporary_directory_;
};

TEST_F(CompactVocabularyGpt2Test, BelgiumPromptUsesRetainedSmallerTokens) {
  const std::string prompt = "The capital of Belgium is";
  auto ordinary = tokenizer_->Encode(*executor_, prompt);
  ASSERT_TRUE(ordinary.ok()) << ordinary.status();
  ASSERT_EQ(TokenId(" Belgium"), 15664);
  ASSERT_NE(std::find(ordinary->begin(), ordinary->end(), 15664),
            ordinary->end());

  auto retained = ByteTokens();
  for (int id : ordinary->span())
    if (id != 15664)
      retained.push_back(id);
  auto compact = Create(std::move(retained));
  ASSERT_TRUE(compact.ok()) << compact.status();
  ASSERT_FALSE((*compact)->CompactId(15664).ok());
  auto encoded = EncodeOriginal(**compact, prompt);
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  EXPECT_EQ(std::find(encoded->begin(), encoded->end(), 15664), encoded->end());
  EXPECT_GT(encoded->size(), ordinary->size());
}

TEST_F(CompactVocabularyGpt2Test, FindsCompletePathPastGreedyDeadEnd) {
  const int a = TokenId("a");
  const int ab = TokenId("ab");
  const int bc = TokenId("bc");
  auto compact = Create({a, ab, bc});
  ASSERT_TRUE(compact.ok()) << compact.status();
  ASSERT_FALSE((*compact)->CompactId(TokenId("abc")).ok());
  auto encoded = EncodeOriginal(**compact, "abc");
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  EXPECT_EQ(*encoded, (std::vector<int>{a, bc}));
}

TEST_F(CompactVocabularyGpt2Test, FallbackCanCrossOrdinaryBpeTokenBoundaries) {
  const int a = TokenId("a");
  const int bc = TokenId("bc");
  const int d = TokenId("d");
  auto ordinary = tokenizer_->Encode(*executor_, "abcd");
  ASSERT_TRUE(ordinary.ok()) << ordinary.status();
  ASSERT_EQ(std::vector<int>(ordinary->begin(), ordinary->end()),
            (std::vector<int>{TokenId("ab"), TokenId("cd")}));
  auto compact = Create({a, bc, d});
  ASSERT_TRUE(compact.ok()) << compact.status();
  auto encoded = EncodeOriginal(**compact, "abcd");
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  EXPECT_EQ(*encoded, (std::vector<int>{a, bc, d}));
}

TEST_F(CompactVocabularyGpt2Test, RetainedWholeTokenNeedsNoIntermediateTokens) {
  const int abc = TokenId("abc");
  auto compact = Create({abc});
  ASSERT_TRUE(compact.ok()) << compact.status();
  for (const char* missing : {"a", "b", "c", "ab", "bc"})
    ASSERT_FALSE((*compact)->CompactId(TokenId(missing)).ok());
  auto encoded = EncodeOriginal(**compact, "abc");
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  EXPECT_EQ(*encoded, (std::vector<int>{abc}));
}

TEST_F(CompactVocabularyGpt2Test, RetainsEveryOriginalCorpusTokenId) {
  std::ifstream input(CorpusPath(), std::ios::binary);
  ASSERT_TRUE(input) << CorpusPath();
  const std::string corpus{std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>()};
  auto mapping = BuildCompactVocabularyMapping(*executor_, *tokenizer_, corpus,
                                               tokenizer_->eos_token_id());
  ASSERT_TRUE(mapping.ok()) << mapping.status();
  auto compact = CompactVocabularyTokenizer::Create(*tokenizer_, *mapping);
  ASSERT_TRUE(compact.ok()) << compact.status();

  std::istringstream lines(corpus);
  std::string line;
  int line_count = 0;
  size_t token_count = 0;
  while (std::getline(lines, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    SCOPED_TRACE(line);
    auto ordinary = tokenizer_->Encode(*executor_, line);
    ASSERT_TRUE(ordinary.ok()) << ordinary.status();
    auto encoded = EncodeOriginal(**compact, line);
    ASSERT_TRUE(encoded.ok()) << encoded.status();
    EXPECT_EQ(*encoded, std::vector<int>(ordinary->begin(), ordinary->end()));
    token_count += encoded->size();
    ++line_count;
  }
  EXPECT_EQ(line_count, 1024);
  EXPECT_EQ(token_count, 14098u);

  ASSERT_FALSE((*compact)->CompactId(15664).ok());
  auto belgium = EncodeOriginal(**compact, "The capital of Belgium is");
  ASSERT_TRUE(belgium.ok()) << belgium.status();
  EXPECT_EQ(*belgium,
            (std::vector<int>{464, 3139, 286, 347, 417, 70, 1505, 318}));
}

TEST_F(CompactVocabularyGpt2Test, ReportsReadableFailingUnicodeSubstring) {
  auto compact = Create({TokenId("ok")});
  ASSERT_TRUE(compact.ok()) << compact.status();
  auto encoded = (*compact)->Encode(*executor_, "ok café");
  ASSERT_FALSE(encoded.ok());
  EXPECT_EQ(encoded.status().code(), absl::StatusCode::kInvalidArgument);
  const std::string message(encoded.status().message());
  EXPECT_NE(message.find(" café"), std::string::npos) << message;
  EXPECT_NE(message.find("byte 2"), std::string::npos) << message;
  EXPECT_NE(message.find("could not be encoded"), std::string::npos) << message;
  EXPECT_EQ(message.find("Ġ"), std::string::npos) << message;
  EXPECT_EQ(message.find("token id"), std::string::npos) << message;

  auto after_eos = (*compact)->Encode(*executor_, "ok<|endoftext|> café");
  ASSERT_FALSE(after_eos.ok());
  EXPECT_EQ(after_eos.status().code(), absl::StatusCode::kInvalidArgument);
  const std::string after_eos_message(after_eos.status().message());
  EXPECT_NE(after_eos_message.find(" café"), std::string::npos)
      << after_eos_message;
  EXPECT_NE(after_eos_message.find("byte 15"), std::string::npos)
      << after_eos_message;
}

TEST_F(CompactVocabularyGpt2Test, SharedTokenizerCachesDoNotMixActiveMappings) {
  const int a = TokenId("a");
  const int ab = TokenId("ab");
  const int bc = TokenId("bc");
  const int c = TokenId("c");
  const int abc = TokenId("abc");
  auto first = Create({a, bc});
  auto second = Create({ab, c});
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(second.ok()) << second.status();
  for (int repeat = 0; repeat < 5; ++repeat) {
    auto first_ids = EncodeOriginal(**first, "abc");
    auto second_ids = EncodeOriginal(**second, "abc");
    ASSERT_TRUE(first_ids.ok()) << first_ids.status();
    ASSERT_TRUE(second_ids.ok()) << second_ids.status();
    EXPECT_EQ(*first_ids, (std::vector<int>{a, bc}));
    EXPECT_EQ(*second_ids, (std::vector<int>{ab, c}));
    EXPECT_EQ(TokenId("abc"), abc);
  }
}

TEST_F(CompactVocabularyGpt2Test, EqualLengthFallbackPathsAreDeterministic) {
  const int ab = TokenId("ab");
  const int c = TokenId("c");
  auto compact = Create({TokenId("a"), ab, TokenId("bc"), c});
  ASSERT_TRUE(compact.ok()) << compact.status();
  for (int repeat = 0; repeat < 5; ++repeat) {
    auto encoded = EncodeOriginal(**compact, "abc");
    ASSERT_TRUE(encoded.ok()) << encoded.status();
    // Both a+bc and ab+c use two tokens; prefer the longer first token.
    EXPECT_EQ(*encoded, (std::vector<int>{ab, c}));
  }
}

TEST_F(CompactVocabularyGpt2Test,
       ByteFallbackRoundTripsUnicodeAndControlBytes) {
  auto compact = Create(ByteTokens());
  ASSERT_TRUE(compact.ok()) << compact.status();
  const std::vector<std::string> samples = {
      "naïve café 🌍",
      "  leading\tspaces\r\nnext line\n",
      "!?... ()—‘quoted’",
      std::string{"before\0after", 12},
      std::string{"\x01\x1f\x7f", 3},
      std::string{"\xED\x9F\xBF\xEE\x80\x80\xF4\x8F\xBF\xBF", 10}};
  for (const auto& sample : samples) {
    SCOPED_TRACE(testing::PrintToString(sample));
    auto encoded = EncodeOriginal(**compact, sample);
    ASSERT_TRUE(encoded.ok()) << encoded.status();
    EXPECT_EQ(encoded->size(), sample.size());
    for (int id : *encoded)
      EXPECT_LT(id, 256);
  }
}

TEST_F(CompactVocabularyGpt2Test, EmptyInputAndEosRemainAtomic) {
  const int a = TokenId("a");
  const int b = TokenId("b");
  const int eos = tokenizer_->eos_token_id();
  auto compact = Create({a, b});
  ASSERT_TRUE(compact.ok()) << compact.status();
  auto empty = EncodeOriginal(**compact, "");
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_TRUE(empty->empty());
  auto encoded = EncodeOriginal(**compact, "a<|endoftext|><|endoftext|>b");
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  EXPECT_EQ(*encoded, (std::vector<int>{a, eos, eos, b}));
}

TEST_F(CompactVocabularyGpt2Test,
       InvalidUtf8StillFailsWithCompleteByteAlphabet) {
  auto compact = Create(ByteTokens());
  ASSERT_TRUE(compact.ok()) << compact.status();
  for (const std::string& invalid :
       {std::string{"\x80", 1}, std::string{"\xC3", 1},
        std::string{"\xC0\xAF", 2}, std::string{"\xED\xA0\x80", 3},
        std::string{"\xF4\x90\x80\x80", 4}}) {
    auto encoded = (*compact)->Encode(*executor_, "valid prefix " + invalid);
    EXPECT_EQ(encoded.status().code(), absl::StatusCode::kInvalidArgument);
  }
  auto valid = EncodeOriginal(**compact, "valid again 🌍");
  ASSERT_TRUE(valid.ok()) << valid.status();
}

TEST_F(CompactVocabularyGpt2Test,
       RestrictedInterfaceRejectsBadMasksAndAbsentEos) {
  const Tokenizer& encoder = *tokenizer_;
  for (int delta : {-1, 1}) {
    const std::vector<uint8_t> invalid(tokenizer_->vocab_size() + delta, 1);
    EXPECT_EQ(
        encoder.EncodeWithVocabulary(*executor_, "", invalid).status().code(),
        absl::StatusCode::kInvalidArgument);
  }
  std::vector<uint8_t> mask(tokenizer_->vocab_size(), 1);
  mask[tokenizer_->eos_token_id()] = 0;
  auto encoded =
      encoder.EncodeWithVocabulary(*executor_, "<|endoftext|>", mask);
  ASSERT_FALSE(encoded.ok());
  EXPECT_EQ(encoded.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(encoded.status().message().find("<|endoftext|>"),
            absl::string_view::npos);

  std::fill(mask.begin(), mask.end(), 0);
  auto empty = encoder.EncodeWithVocabulary(*executor_, "", mask);
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_TRUE(empty->empty());

  const int a = TokenId("a");
  mask[a] = 2;
  auto nonzero = encoder.EncodeWithVocabulary(*executor_, "a", mask);
  ASSERT_TRUE(nonzero.ok()) << nonzero.status();
  EXPECT_EQ(std::vector<int>(nonzero->begin(), nonzero->end()),
            (std::vector<int>{a}));
}

TEST_F(CompactVocabularyGpt2Test, SavedMappingRestoresFallbackEncoding) {
  auto compact = Create({TokenId("a"), TokenId("ab"), TokenId("bc")});
  ASSERT_TRUE(compact.ok()) << compact.status();
  const std::string pattern =
      (std::filesystem::path(testing::TempDir()) / "compact_gpt2.XXXXXX")
          .string();
  std::vector<char> writable(pattern.begin(), pattern.end());
  writable.push_back('\0');
  const char* directory = mkdtemp(writable.data());
  ASSERT_NE(directory, nullptr);
  temporary_directory_ = directory;
  const auto path = temporary_directory_ / "mapping.tsv";
  ASSERT_TRUE((*compact)->SaveToFile(path).ok());
  auto loaded = CompactVocabularyTokenizer::LoadFromFile(*tokenizer_, path);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  auto before = EncodeOriginal(**compact, "abc<|endoftext|>abc");
  auto after = EncodeOriginal(**loaded, "abc<|endoftext|>abc");
  ASSERT_TRUE(before.ok()) << before.status();
  ASSERT_TRUE(after.ok()) << after.status();
  EXPECT_EQ(*before, *after);
  EXPECT_EQ((*compact)->original_token_ids(), (*loaded)->original_token_ids());
}

}  // namespace
}  // namespace pluto::tokenizer
