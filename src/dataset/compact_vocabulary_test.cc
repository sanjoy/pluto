#include "src/dataset/compact_vocabulary.h"

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

namespace pluto::tokenizer {
namespace {

// A convenience for the existing end-to-end corpus/serialization checks. The
// standalone helper and corpus-independent constructor are tested separately.
absl::StatusOr<std::unique_ptr<CompactVocabularyTokenizer>> CreateFromCorpus(
    cuda::Executor& executor, const Tokenizer& original,
    absl::string_view corpus, int eos) {
  auto mapping = BuildCompactVocabularyMapping(executor, original, corpus, eos);
  if (!mapping.ok())
    return mapping.status();
  return CompactVocabularyTokenizer::Create(original, std::move(*mapping));
}

class RecordingTokenizer final : public Tokenizer {
 public:
  absl::StatusOr<cuda::PageLockedHostArray<int>> Encode(
      cuda::Executor& executor, absl::string_view text) const override {
    encoded_text.emplace_back(text);
    return bytes_.Encode(executor, text);
  }
  int vocab_size() const override { return bytes_.vocab_size(); }
  mutable std::vector<std::string> encoded_text;

 private:
  PlainTextTokenizer bytes_;
};

// Deliberately returns the same shared array, as permitted by Tokenizer.
class FixedTokenizer final : public Tokenizer {
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

// These constructor tests intentionally create no executor. Constructing a
// wrapper only needs a completed ID mapping, not access to a corpus or CUDA.
class NoEncodeTokenizer final : public Tokenizer {
 public:
  explicit NoEncodeTokenizer(int vocabulary_size = 8)
      : vocabulary_size_(vocabulary_size) {}
  absl::StatusOr<cuda::PageLockedHostArray<int>> Encode(
      cuda::Executor&, absl::string_view) const override {
    ++encode_calls;
    return absl::FailedPreconditionError("Encode must not be called");
  }
  int vocab_size() const override { return vocabulary_size_; }
  mutable int encode_calls = 0;

 private:
  int vocabulary_size_;
};

CompactVocabularyMapping ExampleMapping() {
  return {.compact_to_original = {1, 3, 7},
          .original_to_compact = {-1, 0, -1, 1, -1, -1, -1, 2},
          .original_eos_id = 3};
}

TEST(CompactVocabularyConstructionTest, NeedsOnlyOriginalTokenizerAndMapping) {
  NoEncodeTokenizer original;
  const auto mapping = ExampleMapping();
  auto compact = CompactVocabularyTokenizer::Create(original, mapping);
  ASSERT_TRUE(compact.ok()) << compact.status();
  EXPECT_EQ(original.encode_calls, 0);
  EXPECT_EQ((*compact)->vocab_size(), 3);
  EXPECT_EQ((*compact)->original_vocab_size(), 8);
  EXPECT_EQ((*compact)->eos_token_id(), 1);
  EXPECT_EQ((*compact)->original_eos_token_id(), 3);
  for (int compact_id = 0; compact_id < (*compact)->vocab_size();
       ++compact_id) {
    const int original_id = mapping.compact_to_original[compact_id];
    ASSERT_TRUE((*compact)->OriginalId(compact_id).ok());
    ASSERT_TRUE((*compact)->CompactId(original_id).ok());
    EXPECT_EQ(*(*compact)->OriginalId(compact_id), original_id);
    EXPECT_EQ(*(*compact)->CompactId(original_id), compact_id);
  }
  EXPECT_EQ(mapping.compact_to_original, ExampleMapping().compact_to_original);
  EXPECT_EQ(mapping.original_to_compact, ExampleMapping().original_to_compact);
  EXPECT_EQ(mapping.original_eos_id, 3);
}

TEST(CompactVocabularyConstructionTest, OwnsCopiedAndMovedMappingStorage) {
  NoEncodeTokenizer original;
  auto mapping = ExampleMapping();
  auto copied = CompactVocabularyTokenizer::Create(original, mapping);
  ASSERT_TRUE(copied.ok()) << copied.status();
  mapping.compact_to_original.assign(1, 0);
  mapping.original_to_compact.assign(8, -1);
  mapping.original_eos_id = 0;
  auto moved = [&] {
    auto local = ExampleMapping();
    return CompactVocabularyTokenizer::Create(original, std::move(local));
  }();
  ASSERT_TRUE(moved.ok()) << moved.status();
  for (const auto* compact : {copied->get(), moved->get()}) {
    EXPECT_EQ(std::vector<int>(compact->original_token_ids().begin(),
                               compact->original_token_ids().end()),
              (std::vector<int>{1, 3, 7}));
    ASSERT_TRUE(compact->CompactId(7).ok());
    EXPECT_EQ(*compact->CompactId(7), 2);
    EXPECT_EQ(compact->original_eos_token_id(), 3);
    EXPECT_EQ(compact->eos_token_id(), 1);
  }
  EXPECT_EQ(original.encode_calls, 0);
}

TEST(CompactVocabularyConstructionTest, AcceptsMappingContainingOnlyEos) {
  NoEncodeTokenizer original;
  CompactVocabularyMapping mapping{
      .compact_to_original = {7},
      .original_to_compact = {-1, -1, -1, -1, -1, -1, -1, 0},
      .original_eos_id = 7};
  auto compact =
      CompactVocabularyTokenizer::Create(original, std::move(mapping));
  ASSERT_TRUE(compact.ok()) << compact.status();
  EXPECT_EQ((*compact)->vocab_size(), 1);
  EXPECT_EQ((*compact)->eos_token_id(), 0);
  EXPECT_EQ(original.encode_calls, 0);
}

TEST(CompactVocabularyConstructionTest,
     RejectsMalformedMappingsWithoutEncoding) {
  NoEncodeTokenizer original;
  std::vector<std::pair<std::string, CompactVocabularyMapping>> invalid;
  auto add_case = [&](const char* name, auto change) {
    auto mapping = ExampleMapping();
    change(mapping);
    invalid.emplace_back(name, std::move(mapping));
  };
  add_case("empty active set", [](auto& m) {
    m.compact_to_original.clear();
    m.original_to_compact.assign(8, -1);
  });
  add_case("unsorted but otherwise inverse", [](auto& m) {
    m.compact_to_original = {3, 1, 7};
    m.original_to_compact[1] = 1;
    m.original_to_compact[3] = 0;
  });
  add_case("duplicate original", [](auto& m) { m.compact_to_original[0] = 3; });
  add_case("negative original", [](auto& m) { m.compact_to_original[0] = -1; });
  add_case("out-of-range original",
           [](auto& m) { m.compact_to_original[2] = 8; });
  add_case("huge original", [](auto& m) {
    m.compact_to_original[2] = std::numeric_limits<int>::max();
  });
  add_case("negative EOS", [](auto& m) { m.original_eos_id = -1; });
  add_case("out-of-range EOS", [](auto& m) { m.original_eos_id = 8; });
  add_case("absent EOS", [](auto& m) { m.original_eos_id = 2; });
  add_case("short reverse map",
           [](auto& m) { m.original_to_compact.pop_back(); });
  add_case("long reverse map",
           [](auto& m) { m.original_to_compact.push_back(-1); });
  add_case("empty reverse map", [](auto& m) { m.original_to_compact.clear(); });
  add_case("wrong active inverse",
           [](auto& m) { m.original_to_compact[1] = 2; });
  add_case("missing active inverse",
           [](auto& m) { m.original_to_compact[7] = -1; });
  add_case("missing EOS inverse",
           [](auto& m) { m.original_to_compact[3] = -1; });
  add_case("inactive aliases active",
           [](auto& m) { m.original_to_compact[0] = 0; });
  add_case("invalid inactive sentinel",
           [](auto& m) { m.original_to_compact[0] = -2; });
  add_case("out-of-range compact ID",
           [](auto& m) { m.original_to_compact[7] = 3; });
  add_case("huge compact ID", [](auto& m) {
    m.original_to_compact[7] = std::numeric_limits<int>::max();
  });
  for (const auto& [name, mapping] : invalid) {
    SCOPED_TRACE(name);
    EXPECT_EQ(
        CompactVocabularyTokenizer::Create(original, mapping).status().code(),
        absl::StatusCode::kInvalidArgument);
  }
  EXPECT_EQ(original.encode_calls, 0);
  for (int size : {0, -1}) {
    NoEncodeTokenizer invalid_original(size);
    EXPECT_EQ(
        CompactVocabularyTokenizer::Create(invalid_original, ExampleMapping())
            .status()
            .code(),
        absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(invalid_original.encode_calls, 0);
  }
}

class CompactVocabularyLoadingTest : public testing::Test {
 protected:
  void SetUp() override {
    const std::string pattern =
        (std::filesystem::path(testing::TempDir()) / "compact_load.XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const char* created = mkdtemp(writable.data());
    ASSERT_NE(created, nullptr);
    directory_ = created;
    path_ = directory_ / "mapping.tsv";
  }

  void TearDown() override {
    EXPECT_EQ(original_.encode_calls, 0);
    if (!directory_.empty()) {
      // Only the unique directory successfully created by this fixture.
      std::error_code error;
      std::filesystem::remove_all(directory_, error);
      EXPECT_FALSE(error) << error.message();
    }
  }

  void Write(const std::string& text) {
    std::ofstream output(path_, std::ios::binary);
    output << text;
    output.close();
    ASSERT_TRUE(output);
  }

  const std::string canonical_ =
      "compact_vocabulary_v1\noriginal_vocab_size\t8\n"
      "original_eos_token\t3\ncompact_vocab_size\t3\n"
      "compact_id\toriginal_id\n0\t1\n1\t3\n2\t7\n";
  NoEncodeTokenizer original_;
  std::filesystem::path directory_;
  std::filesystem::path path_;
};

TEST_F(CompactVocabularyLoadingTest, RoundTripsWithoutCorpusOrExecutor) {
  auto saved = CompactVocabularyTokenizer::Create(original_, ExampleMapping());
  ASSERT_TRUE(saved.ok()) << saved.status();
  ASSERT_TRUE((*saved)->SaveToFile(path_).ok());
  auto loaded = CompactVocabularyTokenizer::LoadFromFile(original_, path_);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  EXPECT_EQ((*loaded)->vocab_size(), 3);
  EXPECT_EQ((*loaded)->original_vocab_size(), 8);
  EXPECT_EQ((*loaded)->eos_token_id(), 1);
  EXPECT_EQ((*loaded)->original_eos_token_id(), 3);
  const auto ids = (*loaded)->original_token_ids();
  EXPECT_EQ(std::vector<int>(ids.begin(), ids.end()),
            (std::vector<int>{1, 3, 7}));
  for (int compact_id = 0; compact_id < (*loaded)->vocab_size(); ++compact_id) {
    auto original_id = (*loaded)->OriginalId(compact_id);
    ASSERT_TRUE(original_id.ok()) << original_id.status();
    auto roundtrip = (*loaded)->CompactId(*original_id);
    ASSERT_TRUE(roundtrip.ok()) << roundtrip.status();
    EXPECT_EQ(*roundtrip, compact_id);
  }
  for (int original_id : {-1, 0, 2, 4, 5, 6, 8})
    EXPECT_EQ((*loaded)->CompactId(original_id).status().code(),
              absl::StatusCode::kInvalidArgument);
  for (int compact_id : {-1, 3})
    EXPECT_EQ((*loaded)->OriginalId(compact_id).status().code(),
              absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE((*loaded)->ValidateFile(path_).ok());
  const auto roundtrip_path = directory_ / "roundtrip.tsv";
  ASSERT_TRUE((*loaded)->SaveToFile(roundtrip_path).ok());
  EXPECT_TRUE((*saved)->ValidateFile(roundtrip_path).ok());
}

TEST_F(CompactVocabularyLoadingTest, LoadsEosOnlyAndZeroOriginalId) {
  Write(
      "compact_vocabulary_v1\noriginal_vocab_size\t8\n"
      "original_eos_token\t0\ncompact_vocab_size\t1\n"
      "compact_id\toriginal_id\n0\t0\n");
  auto loaded = CompactVocabularyTokenizer::LoadFromFile(original_, path_);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  EXPECT_EQ((*loaded)->vocab_size(), 1);
  EXPECT_EQ((*loaded)->eos_token_id(), 0);
  EXPECT_EQ((*loaded)->original_eos_token_id(), 0);
  ASSERT_TRUE((*loaded)->OriginalId(0).ok());
  EXPECT_EQ(*(*loaded)->OriginalId(0), 0);
  ASSERT_TRUE((*loaded)->CompactId(0).ok());
  EXPECT_EQ(*(*loaded)->CompactId(0), 0);
  EXPECT_FALSE((*loaded)->CompactId(1).ok());
  EXPECT_TRUE((*loaded)->ValidateFile(path_).ok());
}

TEST_F(CompactVocabularyLoadingTest, RejectsMalformedMappingContents) {
  const std::vector<std::pair<std::string, std::string>> changes{
      {"v1", "v2"},
      {"original_vocab_size", "vocabulary_size"},
      {"original_vocab_size\t8", "original_vocab_size\t0"},
      {"original_vocab_size\t8", "original_vocab_size\t-1"},
      {"original_vocab_size\t8", "original_vocab_size\t2147483648"},
      {"original_vocab_size\t8", "original_vocab_size\t08"},
      {"original_vocab_size\t8", "original_vocab_size\t8 "},
      {"original_eos_token\t3", "original_eos_token\t-1"},
      {"original_eos_token\t3", "original_eos_token\t8"},
      {"original_eos_token\t3", "original_eos_token\t2"},
      {"original_eos_token\t3\n", ""},
      {"compact_vocab_size\t3", "compact_vocab_size\t0"},
      {"compact_vocab_size\t3", "compact_vocab_size\t9"},
      {"compact_vocab_size\t3", "compact_vocab_size\t2"},
      {"compact_vocab_size\t3", "compact_vocab_size\t4"},
      {"compact_vocab_size\t3", "compact_vocab_size\t2147483648"},
      {"compact_id\toriginal_id", "original_id\tcompact_id"},
      {"0\t1", "1\t1"},
      {"1\t3", "0\t3"},
      {"1\t3", "2\t3"},
      {"0\t1", "-1\t1"},
      {"0\t1", "+0\t1"},
      {"0\t1", "00\t1"},
      {"0\t1", "2147483648\t1"},
      {"0\t1", "0\t3"},
      {"0\t1\n1\t3", "0\t3\n1\t1"},
      {"0\t1", "0\t-1"},
      {"0\t1", "0\t-0"},
      {"0\t1", "0\t8"},
      {"0\t1", "0\t2147483648"},
      {"0\t1", "0\t"},
      {"0\t1", "0\t01"},
      {"0\t1", "0\tone"},
      {"0\t1", "0\t1x"},
      {"0\t1", "0 1"},
      {"0\t1", "0\t\t1"},
      {"0\t1", "0\t1\t1"},
      {"0\t1", "0\t" + std::string(4096, '9')},
  };
  for (const auto& [old_text, new_text] : changes) {
    SCOPED_TRACE(new_text);
    std::string invalid = canonical_;
    const size_t position = invalid.find(old_text);
    ASSERT_NE(position, std::string::npos);
    invalid.replace(position, old_text.size(), new_text);
    Write(invalid);
    EXPECT_EQ(CompactVocabularyTokenizer::LoadFromFile(original_, path_)
                  .status()
                  .code(),
              absl::StatusCode::kDataLoss);
  }
}

TEST_F(CompactVocabularyLoadingTest, RejectsTruncationAndTrailingData) {
  for (size_t length = 0; length < canonical_.size(); ++length) {
    SCOPED_TRACE(length);
    Write(canonical_.substr(0, length));
    EXPECT_EQ(CompactVocabularyTokenizer::LoadFromFile(original_, path_)
                  .status()
                  .code(),
              absl::StatusCode::kDataLoss);
  }
  for (const std::string& suffix :
       {std::string("\n"), std::string("extra"), std::string("3\t0\n"),
        std::string(1, '\0')}) {
    SCOPED_TRACE(suffix);
    Write(canonical_ + suffix);
    EXPECT_EQ(CompactVocabularyTokenizer::LoadFromFile(original_, path_)
                  .status()
                  .code(),
              absl::StatusCode::kDataLoss);
  }
  std::string crlf;
  for (char character : canonical_) {
    if (character == '\n')
      crlf.push_back('\r');
    crlf.push_back(character);
  }
  Write(crlf);
  EXPECT_EQ(CompactVocabularyTokenizer::LoadFromFile(original_, path_)
                .status()
                .code(),
            absl::StatusCode::kDataLoss);
}

TEST_F(CompactVocabularyLoadingTest, RejectsIncompatibleOriginalVocabulary) {
  Write(canonical_);
  for (int original_size : {7, 9}) {
    NoEncodeTokenizer incompatible(original_size);
    EXPECT_EQ(CompactVocabularyTokenizer::LoadFromFile(incompatible, path_)
                  .status()
                  .code(),
              absl::StatusCode::kFailedPrecondition);
    EXPECT_EQ(incompatible.encode_calls, 0);
  }
  for (int original_size : {0, -1}) {
    NoEncodeTokenizer invalid(original_size);
    EXPECT_EQ(CompactVocabularyTokenizer::LoadFromFile(invalid, path_)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(invalid.encode_calls, 0);
  }
}

TEST_F(CompactVocabularyLoadingTest, RejectsMissingEmptyAndNonFilePaths) {
  EXPECT_EQ(CompactVocabularyTokenizer::LoadFromFile(original_, path_)
                .status()
                .code(),
            absl::StatusCode::kNotFound);
  EXPECT_EQ(
      CompactVocabularyTokenizer::LoadFromFile(original_, {}).status().code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(CompactVocabularyTokenizer::LoadFromFile(original_, directory_)
                .status()
                .code(),
            absl::StatusCode::kFailedPrecondition);
}

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
  PlainTextTokenizer tokenizer_;
  std::filesystem::path directory_;
};

TEST_F(CompactVocabularyTest, DiscoveryReturnsCanonicalBidirectionalMapping) {
  RecordingTokenizer original;
  auto mapping =
      BuildCompactVocabularyMapping(*executor_, original, "za\r\n b\nza", 255);
  ASSERT_TRUE(mapping.ok()) << mapping.status();
  const std::vector<int> expected{' ', 'a', 'b', 'z', 255};
  EXPECT_EQ(mapping->compact_to_original, expected);
  EXPECT_EQ(mapping->original_eos_id, 255);
  ASSERT_EQ(mapping->original_to_compact.size(), 256u);
  std::vector<int> expected_reverse(256, -1);
  for (size_t compact = 0; compact < expected.size(); ++compact)
    expected_reverse[expected[compact]] = static_cast<int>(compact);
  EXPECT_EQ(mapping->original_to_compact, expected_reverse);
  EXPECT_EQ(original.encoded_text,
            (std::vector<std::string>{"za", " b", "za"}));
  auto wrapper = CompactVocabularyTokenizer::Create(original, *mapping);
  ASSERT_TRUE(wrapper.ok()) << wrapper.status();
  // Construction reuses the discovered map and must not tokenize a second time.
  EXPECT_EQ(original.encoded_text.size(), 3u);
  auto reordered =
      BuildCompactVocabularyMapping(*executor_, original, " b\nza", 255);
  ASSERT_TRUE(reordered.ok()) << reordered.status();
  EXPECT_EQ(reordered->compact_to_original, mapping->compact_to_original);
  EXPECT_EQ(reordered->original_to_compact, mapping->original_to_compact);
}

TEST_F(CompactVocabularyTest,
       SortedMappingSurvivesLineReorderingAndDuplication) {
  auto first = CreateFromCorpus(*executor_, tokenizer_, "za\n b\n", 255);
  auto reordered = CreateFromCorpus(*executor_, tokenizer_, " b\nza\nza", 255);
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
  auto compact = CreateFromCorpus(*executor_, recording, "ab\r\n cd\r\n", 255);
  ASSERT_TRUE(compact.ok()) << compact.status();
  EXPECT_EQ(recording.encoded_text, (std::vector<std::string>{"ab", " cd"}));
  const auto path = directory_ / "mapping.tsv";
  ASSERT_TRUE((*compact)->SaveToFile(path).ok());
  auto lf = CreateFromCorpus(*executor_, tokenizer_, "ab\n cd\n", 255);
  ASSERT_TRUE(lf.ok()) << lf.status();
  EXPECT_TRUE((*lf)->ValidateFile(path).ok());
  EXPECT_TRUE((*compact)->CompactId(' ').ok());
  EXPECT_FALSE((*compact)->CompactId('\r').ok());
  EXPECT_FALSE((*compact)->CompactId('\n').ok());
}

TEST_F(CompactVocabularyTest,
       EncodeKeepsEveryInputTokenAndAddsNoSpecialTokens) {
  auto compact = CreateFromCorpus(*executor_, tokenizer_, "qabcdef", 255);
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
  auto compact = CreateFromCorpus(*executor_, tokenizer_, "ba", 'a');
  ASSERT_TRUE(compact.ok()) << compact.status();
  EXPECT_EQ((*compact)->vocab_size(), 2);
  EXPECT_EQ((*compact)->eos_token_id(), 0);
  auto encoded = (*compact)->Encode(*executor_, "ba");
  ASSERT_TRUE(encoded.ok()) << encoded.status();
  EXPECT_EQ(std::vector<int>(encoded->begin(), encoded->end()),
            (std::vector<int>{1, 0}));
}

TEST_F(CompactVocabularyTest, RejectsInactiveAndOutOfRangeTokenIds) {
  auto compact = CreateFromCorpus(*executor_, tokenizer_, "ab", 255);
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

TEST_F(CompactVocabularyTest, IdentifiesTheUnencodableByteAndItsOffset) {
  auto compact = CreateFromCorpus(*executor_, tokenizer_, "ab", 255);
  ASSERT_TRUE(compact.ok()) << compact.status();
  const auto missing = (*compact)->Encode(*executor_, "aabcba");
  EXPECT_EQ(missing.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(missing.status().message(),
            "substring \"c\" could not be encoded using the compact vocabulary "
            "at byte 3");
  const auto control = (*compact)->Encode(*executor_, std::string{"a\0b", 3});
  EXPECT_EQ(control.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(control.status().message().find("substring \"\\000\""),
            absl::string_view::npos);
  EXPECT_NE(control.status().message().find("at byte 1"),
            absl::string_view::npos);
}

TEST_F(CompactVocabularyTest, DefaultVocabularyHookReportsTextWithoutMutation) {
  RecordingTokenizer original;
  auto compact = CreateFromCorpus(*executor_, original, "ab", 255);
  ASSERT_TRUE(compact.ok()) << compact.status();
  const auto missing = (*compact)->Encode(*executor_, "ac");
  EXPECT_EQ(missing.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      missing.status().message(),
      "substring \"ac\" could not be encoded using the compact vocabulary");
  auto ordinary = original.Encode(*executor_, "ac");
  ASSERT_TRUE(ordinary.ok()) << ordinary.status();
  EXPECT_EQ(ordinary->span(), absl::Span<const int>({'a', 'c'}));
}

TEST_F(CompactVocabularyTest, VocabularyHooksRejectInvalidMasksBeforeEncoding) {
  RecordingTokenizer original;
  for (size_t size : {0u, 255u, 257u}) {
    const std::vector<uint8_t> invalid(size, 1);
    EXPECT_EQ(original.EncodeWithVocabulary(*executor_, "ab", invalid)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(tokenizer_.EncodeWithVocabulary(*executor_, "ab", invalid)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
  EXPECT_TRUE(original.encoded_text.empty());
  const std::vector<uint8_t> all_allowed(256, 2);
  auto bytes = tokenizer_.EncodeWithVocabulary(*executor_, "ab", all_allowed);
  ASSERT_TRUE(bytes.ok()) << bytes.status();
  EXPECT_EQ(bytes->span(), absl::Span<const int>({'a', 'b'}));
}

TEST_F(CompactVocabularyTest, RejectsInvalidCorporaAndOriginalEos) {
  for (const char* corpus : {"", "\n", "a\n\n", "a\n \t\nb", "a\r\n\r\nb"}) {
    SCOPED_TRACE(corpus);
    EXPECT_EQ(BuildCompactVocabularyMapping(*executor_, tokenizer_, corpus, 255)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
  for (int eos : {-1, 256, std::numeric_limits<int>::max()}) {
    EXPECT_EQ(BuildCompactVocabularyMapping(*executor_, tokenizer_, "ab", eos)
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
    EXPECT_EQ(BuildCompactVocabularyMapping(*executor_, source, "text", 255)
                  .status()
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
  auto empty = cuda::PageLockedHostArray<int>::Allocate(*executor_, 0);
  ASSERT_TRUE(empty.ok()) << empty.status();
  FixedTokenizer empty_source(*empty, 256);
  EXPECT_EQ(BuildCompactVocabularyMapping(*executor_, empty_source, "text", 255)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  FixedTokenizer zero_vocabulary(*empty, 0);
  EXPECT_EQ(
      BuildCompactVocabularyMapping(*executor_, zero_vocabulary, "text", 0)
          .status()
          .code(),
      absl::StatusCode::kInvalidArgument);
}

TEST_F(CompactVocabularyTest, DoesNotMutateSharedOriginalStorage) {
  auto tokens = cuda::PageLockedHostArray<int>::CopyFrom(
      *executor_, std::vector<int>{7, 20});
  ASSERT_TRUE(tokens.ok()) << tokens.status();
  FixedTokenizer source(*tokens, 32);
  auto mapping = BuildCompactVocabularyMapping(*executor_, source, "text", 31);
  ASSERT_TRUE(mapping.ok()) << mapping.status();
  EXPECT_EQ(std::vector<int>(tokens->begin(), tokens->end()),
            (std::vector<int>{7, 20}));
  auto compact =
      CompactVocabularyTokenizer::Create(source, std::move(*mapping));
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
  auto compact = CreateFromCorpus(*executor_, tokenizer_, "ba", 255);
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
  auto different = CreateFromCorpus(*executor_, tokenizer_, "bc", 255);
  ASSERT_TRUE(different.ok()) << different.status();
  EXPECT_EQ((*different)->ValidateFile(path).code(),
            absl::StatusCode::kDataLoss);
  EXPECT_EQ((*different)->SaveToFile(path).code(), absl::StatusCode::kDataLoss);
  EXPECT_EQ(Read(path), expected);
}

TEST_F(CompactVocabularyTest, RejectsMalformedOrMismatchedMappingFiles) {
  auto compact = CreateFromCorpus(*executor_, tokenizer_, "ba", 255);
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
  auto compact = CreateFromCorpus(*executor_, tokenizer_, "ab", 255);
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
}  // namespace pluto::tokenizer
