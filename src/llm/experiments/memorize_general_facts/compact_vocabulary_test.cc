#include "src/llm/experiments/memorize_general_facts/compact_vocabulary.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "src/dataset/plain_text_tokenizer.h"

namespace pluto::llm::memorize_general_facts {
namespace {

class RecordingTokenizer final : public tokenizer::Tokenizer {
 public:
  absl::StatusOr<cuda::PageLockedHostArray<int>> Encode(
      cuda::Executor& executor, absl::string_view text) const override {
    encoded_text.emplace_back(text);
    return bytes_.Encode(executor, text);
  }
  int vocab_size() const override { return bytes_.vocab_size(); }
  mutable std::vector<std::string> encoded_text;

 private:
  tokenizer::PlainTextTokenizer bytes_;
};

// Deliberately returns the same shared array, as permitted by Tokenizer.
class FixedTokenizer final : public tokenizer::Tokenizer {
 public:
  FixedTokenizer(cuda::PageLockedHostArray<int> tokens, int vocabulary_size)
      : tokens_(std::move(tokens)), vocabulary_size_(vocabulary_size) {}
  absl::StatusOr<cuda::PageLockedHostArray<int>> Encode(
      cuda::Executor&, absl::string_view) const override {
    return tokens_;
  }
  int vocab_size() const override { return vocabulary_size_; }

 private:
  cuda::PageLockedHostArray<int> tokens_;
  int vocabulary_size_;
};

class CompactVocabularyTest : public testing::Test {
 protected:
  void SetUp() override {
    auto executor = cuda::Executor::Create();
    ASSERT_TRUE(executor.ok()) << executor.status();
    executor_ = std::move(*executor);
    const std::string pattern =
        (std::filesystem::path(testing::TempDir()) / "compact_vocab.XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const char* created = mkdtemp(writable.data());
    ASSERT_NE(created, nullptr);
    directory_ = created;
  }

  void TearDown() override {
    if (executor_ != nullptr) {
      const auto status = executor_->Synchronize();
      EXPECT_TRUE(status.ok()) << status;
    }
    if (!directory_.empty()) {
      // Only the unique directory successfully created by this fixture.
      std::error_code error;
      std::filesystem::remove_all(directory_, error);
      EXPECT_FALSE(error) << error.message();
    }
  }

  void Write(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary);
    output << text;
    output.close();
    ASSERT_TRUE(output);
  }

  std::string Read(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    EXPECT_TRUE(input);
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
  }

  std::unique_ptr<cuda::Executor> executor_;
  tokenizer::PlainTextTokenizer tokenizer_;
  std::filesystem::path directory_;
};

TEST_F(CompactVocabularyTest,
       SortedMappingSurvivesLineReorderingAndDuplication) {
  auto first = CompactVocabularyTokenizer::Create(*executor_, tokenizer_,
                                                  "za\n b\n", 255);
  auto reordered = CompactVocabularyTokenizer::Create(*executor_, tokenizer_,
                                                      " b\nza\nza", 255);
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(reordered.ok()) << reordered.status();
  const std::vector<int> expected{' ', 'a', 'b', 'z', 255};
  const auto original_ids = (*first)->original_token_ids();
  EXPECT_EQ(std::vector<int>(original_ids.begin(), original_ids.end()),
            expected);
  EXPECT_EQ((*first)->vocab_size(), 5);
  EXPECT_EQ((*first)->original_vocab_size(), 256);
  EXPECT_EQ((*first)->original_eos_token_id(), 255);
  EXPECT_EQ((*first)->eos_token_id(), 4);
  const auto path = directory_ / "mapping.tsv";
  ASSERT_TRUE((*first)->SaveToFile(path).ok());
  EXPECT_TRUE((*reordered)->ValidateFile(path).ok());
  for (size_t compact = 0; compact < expected.size(); ++compact) {
    auto original = (*first)->OriginalId(static_cast<int>(compact));
    auto roundtrip = (*first)->CompactId(expected[compact]);
    ASSERT_TRUE(original.ok()) << original.status();
    ASSERT_TRUE(roundtrip.ok()) << roundtrip.status();
    EXPECT_EQ(*original, expected[compact]);
    EXPECT_EQ(*roundtrip, static_cast<int>(compact));
  }
}

TEST_F(CompactVocabularyTest, UsesExactPerLineTokenizationAndCrLfConventions) {
  RecordingTokenizer recording;
  auto compact = CompactVocabularyTokenizer::Create(*executor_, recording,
                                                    "ab\r\n cd\r\n", 255);
  ASSERT_TRUE(compact.ok()) << compact.status();
  EXPECT_EQ(recording.encoded_text, (std::vector<std::string>{"ab", " cd"}));
  const auto path = directory_ / "mapping.tsv";
  ASSERT_TRUE((*compact)->SaveToFile(path).ok());
  auto lf = CompactVocabularyTokenizer::Create(*executor_, tokenizer_,
                                               "ab\n cd\n", 255);
  ASSERT_TRUE(lf.ok()) << lf.status();
  EXPECT_TRUE((*lf)->ValidateFile(path).ok());
  EXPECT_TRUE((*compact)->CompactId(' ').ok());
  EXPECT_FALSE((*compact)->CompactId('\r').ok());
  EXPECT_FALSE((*compact)->CompactId('\n').ok());
}

TEST_F(CompactVocabularyTest,
       EncodeKeepsEveryInputTokenAndAddsNoSpecialTokens) {
  auto compact = CompactVocabularyTokenizer::Create(*executor_, tokenizer_,
                                                    "qabcdef", 255);
  ASSERT_TRUE(compact.ok()) << compact.status();
  EXPECT_EQ((*compact)->vocab_size(), 8);  // Seven input IDs, plus EOS.
  auto encoded = (*compact)->Encode(*executor_, "qabcdef");
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  ASSERT_EQ(encoded->size(), 7u);
  const std::string original = "qabcdef";
  for (size_t index = 0; index < encoded->size(); ++index) {
    auto restored = (*compact)->OriginalId((*encoded)[index]);
    ASSERT_TRUE(restored.ok()) << restored.status();
    EXPECT_EQ(*restored, original[index]);
  }
  auto empty = (*compact)->Encode(*executor_, "");
  ASSERT_TRUE(empty.ok()) << empty.status();
  EXPECT_TRUE(empty->empty());
}

TEST_F(CompactVocabularyTest, EosIsIncludedOnceAndNeedNotBeLargestOriginalId) {
  auto compact =
      CompactVocabularyTokenizer::Create(*executor_, tokenizer_, "ba", 'a');
  ASSERT_TRUE(compact.ok()) << compact.status();
  EXPECT_EQ((*compact)->vocab_size(), 2);
  EXPECT_EQ((*compact)->eos_token_id(), 0);
  auto encoded = (*compact)->Encode(*executor_, "ba");
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  EXPECT_EQ(std::vector<int>(encoded->begin(), encoded->end()),
            (std::vector<int>{1, 0}));
}

TEST_F(CompactVocabularyTest, RejectsInactiveAndOutOfRangeTokenIds) {
  auto compact =
      CompactVocabularyTokenizer::Create(*executor_, tokenizer_, "ab", 255);
  ASSERT_TRUE(compact.ok()) << compact.status();
  EXPECT_EQ((*compact)->Encode(*executor_, "ac").status().code(),
            absl::StatusCode::kInvalidArgument);
  for (int original :
       {static_cast<int>('c'), -1, 256, std::numeric_limits<int>::max()}) {
    EXPECT_EQ((*compact)->CompactId(original).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
  for (int compact_id : {-1, 3, std::numeric_limits<int>::min(),
                         std::numeric_limits<int>::max()}) {
    EXPECT_EQ((*compact)->OriginalId(compact_id).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST_F(CompactVocabularyTest, RejectsInvalidCorporaAndOriginalEos) {
  for (const char* corpus : {"", "\n", "a\n\n", "a\n \t\nb", "a\r\n\r\nb"}) {
    SCOPED_TRACE(corpus);
    EXPECT_EQ(
        CompactVocabularyTokenizer::Create(*executor_, tokenizer_, corpus, 255)
            .status()
            .code(),
        absl::StatusCode::kInvalidArgument);
  }
  for (int eos : {-1, 256, std::numeric_limits<int>::max()}) {
    EXPECT_EQ(
        CompactVocabularyTokenizer::Create(*executor_, tokenizer_, "ab", eos)
            .status()
            .code(),
        absl::StatusCode::kInvalidArgument);
  }
}

TEST_F(CompactVocabularyTest, RejectsInvalidIdsReturnedByOriginalTokenizer) {
  for (int invalid : {-1, 256, std::numeric_limits<int>::max()}) {
    auto tokens = cuda::PageLockedHostArray<int>::CopyFrom(
        *executor_, std::vector<int>{invalid});
    ASSERT_TRUE(tokens.ok()) << tokens.status();
    FixedTokenizer source(*tokens, 256);
    EXPECT_EQ(
        CompactVocabularyTokenizer::Create(*executor_, source, "text", 255)
            .status()
            .code(),
        absl::StatusCode::kInvalidArgument);
  }
  auto empty = cuda::PageLockedHostArray<int>::Allocate(*executor_, 0);
  ASSERT_TRUE(empty.ok()) << empty.status();
  FixedTokenizer empty_source(*empty, 256);
  EXPECT_EQ(
      CompactVocabularyTokenizer::Create(*executor_, empty_source, "text", 255)
          .status()
          .code(),
      absl::StatusCode::kInvalidArgument);
  FixedTokenizer zero_vocabulary(*empty, 0);
  EXPECT_EQ(
      CompactVocabularyTokenizer::Create(*executor_, zero_vocabulary, "text", 0)
          .status()
          .code(),
      absl::StatusCode::kInvalidArgument);
}

TEST_F(CompactVocabularyTest, DoesNotMutateSharedOriginalStorage) {
  auto tokens = cuda::PageLockedHostArray<int>::CopyFrom(
      *executor_, std::vector<int>{7, 20});
  ASSERT_TRUE(tokens.ok()) << tokens.status();
  FixedTokenizer source(*tokens, 32);
  auto compact =
      CompactVocabularyTokenizer::Create(*executor_, source, "text", 31);
  ASSERT_TRUE(compact.ok()) << compact.status();
  auto encoded = (*compact)->Encode(*executor_, "text");
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  EXPECT_NE(encoded->data(), tokens->data());
  EXPECT_EQ(std::vector<int>(encoded->begin(), encoded->end()),
            (std::vector<int>{0, 1}));
  EXPECT_EQ(std::vector<int>(tokens->begin(), tokens->end()),
            (std::vector<int>{7, 20}));
  for (int invalid_or_inactive : {-1, 8, 32}) {
    (*tokens)[0] = invalid_or_inactive;
    EXPECT_EQ((*compact)->Encode(*executor_, "text").status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST_F(CompactVocabularyTest,
       SavesCanonicalMappingAndRefusesDifferentExistingMap) {
  auto compact =
      CompactVocabularyTokenizer::Create(*executor_, tokenizer_, "ba", 255);
  ASSERT_TRUE(compact.ok()) << compact.status();
  const auto path = directory_ / "mapping.tsv";
  ASSERT_TRUE((*compact)->SaveToFile(path).ok());
  const std::string expected =
      "compact_vocabulary_v1\noriginal_vocab_size\t256\n"
      "original_eos_token\t255\ncompact_vocab_size\t3\n"
      "compact_id\toriginal_id\n0\t97\n1\t98\n2\t255\n";
  EXPECT_EQ(Read(path), expected);
  EXPECT_TRUE((*compact)->ValidateFile(path).ok());
  EXPECT_TRUE((*compact)->SaveToFile(path).ok());
  auto different =
      CompactVocabularyTokenizer::Create(*executor_, tokenizer_, "bc", 255);
  ASSERT_TRUE(different.ok()) << different.status();
  EXPECT_EQ((*different)->ValidateFile(path).code(),
            absl::StatusCode::kDataLoss);
  EXPECT_EQ((*different)->SaveToFile(path).code(), absl::StatusCode::kDataLoss);
  EXPECT_EQ(Read(path), expected);
}

TEST_F(CompactVocabularyTest, RejectsMalformedOrMismatchedMappingFiles) {
  auto compact =
      CompactVocabularyTokenizer::Create(*executor_, tokenizer_, "ba", 255);
  ASSERT_TRUE(compact.ok()) << compact.status();
  const auto path = directory_ / "mapping.tsv";
  ASSERT_TRUE((*compact)->SaveToFile(path).ok());
  const std::string valid = Read(path);
  const std::vector<std::pair<std::string, std::string>> changes{
      {"v1", "v2"},
      {"original_vocab_size\t256", "original_vocab_size\t257"},
      {"original_eos_token\t255", "original_eos_token\t254"},
      {"compact_vocab_size\t3", "compact_vocab_size\t4"},
      {"0\t97", "0\t98"},  // Duplicate original ID.
      {"1\t98", "0\t98"},  // Duplicate compact ID.
      {"0\t97", "0\t-1"},
      {"2\t255", "2\t999"},
      {"1\t98", "1\txx"},
      {"0\t97\n1\t98", "0\t98\n1\t97"},  // Unsorted original IDs.
      {"0\t97\n1\t98", "1\t98\n0\t97"},  // Reordered rows.
      {"2\t255\n", ""},                  // Missing EOS.
  };
  for (const auto& [old_text, new_text] : changes) {
    SCOPED_TRACE(new_text);
    std::string invalid = valid;
    const size_t position = invalid.find(old_text);
    ASSERT_NE(position, std::string::npos);
    invalid.replace(position, old_text.size(), new_text);
    Write(path, invalid);
    EXPECT_EQ((*compact)->ValidateFile(path).code(),
              absl::StatusCode::kDataLoss);
  }
  for (const std::string& invalid :
       {std::string(), valid.substr(0, valid.size() - 1), valid + "extra",
        valid + "\n"}) {
    Write(path, invalid);
    EXPECT_EQ((*compact)->ValidateFile(path).code(),
              absl::StatusCode::kDataLoss);
  }
}

TEST_F(CompactVocabularyTest, RejectsMissingEmptyAndNonFileMappingPaths) {
  auto compact =
      CompactVocabularyTokenizer::Create(*executor_, tokenizer_, "ab", 255);
  ASSERT_TRUE(compact.ok()) << compact.status();
  EXPECT_EQ((*compact)->ValidateFile(directory_ / "missing.tsv").code(),
            absl::StatusCode::kNotFound);
  EXPECT_EQ((*compact)->ValidateFile({}).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ((*compact)->SaveToFile({}).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ((*compact)->ValidateFile(directory_).code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ((*compact)->SaveToFile(directory_).code(),
            absl::StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace pluto::llm::memorize_general_facts
