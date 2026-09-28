#include "src/dataset/qwen_tokenizer.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"

namespace pluto::tokenizer {
namespace {

class QwenTokenizerTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    const char* directory = std::getenv("PLUTO_QWEN_TOKENIZER_DIR");
    if (directory == nullptr)
      return;
    auto loaded = QwenTokenizer::Load(directory);
    ASSERT_TRUE(loaded.ok()) << loaded.status();
    tokenizer_ = std::move(*loaded);
  }
  void SetUp() override {
    if (!tokenizer_)
      GTEST_SKIP()
          << "set PLUTO_QWEN_TOKENIZER_DIR for official checkpoint parity";
  }
  static void TearDownTestSuite() { tokenizer_.reset(); }
  static std::unique_ptr<QwenTokenizer> tokenizer_;
};

std::unique_ptr<QwenTokenizer> QwenTokenizerTest::tokenizer_;

TEST_F(QwenTokenizerTest, MatchesOfficialHuggingFaceTokenIds) {
  // Golden IDs from tokenizers 0.22.0, Tokenizer.from_file(tokenizer.json),
  // using the official Qwen/Qwen3.8-27B-FP8 tokenizer, without postprocessing.
  // These cover Qwen3.8-specific single digits, combining marks, NFC, case
  // insensitive contractions, Unicode whitespace, and added non-special tokens.
  const std::vector<std::pair<std::string, std::vector<int>>> cases = {
      {"", {}},
      {"Hello, world!", {9419, 11, 1814, 0}},
      {"Explain gravity in one sentence.",
       {814, 20139, 22525, 303, 799, 11316, 13}},
      {"1234567890 １２３ ١٢٣",
       {16,    17,    18,    19,  20,  21, 22,  23, 24,  15, 220,
        19496, 24128, 32405, 220, 149, 94, 149, 95, 149, 96}},
      {"WE'RE I'LL He's can't",
       {12100, 90902, 353, 6, 3950, 1216, 579, 628, 914}},
      {"  leading spaces", {220, 6187, 12258}},
      {"foo\t\tbar\u00a0\u00a0baz", {7724, 197, 87586, 3966, 3966, 41159}},
      {"Hello!\r\n\n  Next\n", {9419, 0, 78035, 220, 9019, 198}},
      {"cafe\u0301 naïve A\u030a", {895, 56868, 91603, 571, 76533}},
      {"你好，世界！日本語 한글", {109266, 3709, 96748, 6115, 247359, 209758}},
      {"नमस्ते বাংলা العربية", {58069, 84237, 150104, 153348, 166781, 171405}},
      {"emoji 👩🏽‍💻 🌍!",
       {36280, 59720, 102, 9008, 237, 121, 373, 235, 88995, 119, 10838, 234,
        235, 0}},
      {std::string("before\0after", 12), {14372, 188, 10378}},
      {"a<think>b</think><|im_end|>", {64, 248068, 65, 248069, 248046}},
      {"<|fim_prefix|>x<tool_call>{}</tool_call>",
       {248060, 87, 248058, 6061, 248059}},
      {"\u0301\u0302 foo", {52033, 136, 224, 14785}},
      {"\u2000\u2000word\u2028\u2028next",
       {373, 224, 373, 224, 1119, 373, 101, 373, 101, 3480}},
      {"a  \r\n \tb", {64, 10109, 220, 2161}},
      {"1.23456789 -5e10",
       {16, 13, 17, 18, 19, 20, 21, 22, 23, 24, 471, 20, 68, 16, 15}},
      {"\v\fA\x1c"
       "B",
       {199, 200, 32, 216, 33}},
  };
  EXPECT_EQ(tokenizer_->vocab_size(), 248077);
  EXPECT_EQ(tokenizer_->eos_token_id(), 248046);
  for (const auto& [text, reference] : cases) {
    SCOPED_TRACE(testing::PrintToString(text));
    auto ids = tokenizer_->Encode(text);
    ASSERT_TRUE(ids.ok()) << ids.status();
    EXPECT_EQ(*ids, reference);
  }
}

TEST_F(QwenTokenizerTest, RecognizesEveryAddedToken) {
  const std::vector<std::string> tokens = {
      "<|endoftext|>",
      "<|im_start|>",
      "<|im_end|>",
      "<|object_ref_start|>",
      "<|object_ref_end|>",
      "<|box_start|>",
      "<|box_end|>",
      "<|quad_start|>",
      "<|quad_end|>",
      "<|vision_start|>",
      "<|vision_end|>",
      "<|vision_pad|>",
      "<|image_pad|>",
      "<|video_pad|>",
      "<tool_call>",
      "</tool_call>",
      "<|fim_prefix|>",
      "<|fim_middle|>",
      "<|fim_suffix|>",
      "<|fim_pad|>",
      "<|repo_name|>",
      "<|file_sep|>",
      "<tool_response>",
      "</tool_response>",
      "<think>",
      "</think>",
      "<|audio_start|>",
      "<|audio_end|>",
      "<tts_pad>",
      "<tts_text_bos>",
      "<tts_text_eod>",
      "<tts_text_bos_single>",
      "<|audio_pad|>",
  };
  std::string text;
  std::vector<int> expected;
  for (size_t i = 0; i < tokens.size(); ++i) {
    text.append(tokens[i]);
    expected.push_back(248044 + i);
  }
  auto ids = tokenizer_->Encode(text);
  ASSERT_TRUE(ids.ok()) << ids.status();
  EXPECT_EQ(*ids, expected);
  auto decoded = tokenizer_->Decode(expected);
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(*decoded, text);
}

TEST_F(QwenTokenizerTest, MatchesReferenceForRandomizedUnicodeBoundaries) {
  // Each checksum is FNV-1a over little-endian uint32 IDs from the same
  // official Hugging Face tokenizer as the explicit golden cases above.
  // A fixed xorshift32 stream exercises 6,045 reference tokens, including
  // whitespace/mark transitions and added tokens beside combining characters.
  const std::vector<std::string> pieces = {
      "a",
      "B",
      "hello",
      " world",
      "12345",
      "１２",
      "١٢",
      "can't",
      "WE'RE",
      "\u0301",
      "\u0302",
      "é",
      "e\u0301",
      "A\u030a",
      "नमस्ते",
      "বাংলা",
      "العربية",
      "你好",
      "日本語",
      "한글",
      "\u1112\u1161\u11ab",
      "👩🏽‍💻",
      "🌍",
      "✈️",
      "...",
      "!?\r\n",
      "--",
      "_",
      "/",
      " ",
      "  ",
      "\t",
      "\t\t",
      "\n",
      "\r\n",
      "\v",
      "\f",
      "\u0085",
      "\u00a0",
      "\u2000",
      "\u200b",
      "\u2028",
      "\u2029",
      "\u3000",
      std::string(1, '\0'),
      "\x1c",
      "<think>",
      "</think>",
      "<|im_end|>",
      "<tool_call>",
      "𐐀",
      "𝟘",
      "\u093f",
      "\u05b0",
  };
  const uint64_t expected[] = {
      0xcb74056f93db6c1aULL, 0xb2f595e2ccb1c1acULL, 0xcad9c6281f779d2dULL,
      0x01777a45c8563185ULL, 0xa1c0c40befc02d2bULL, 0xd34acf5e402808b5ULL,
      0x2efbac61116adfc6ULL, 0xfa140791bda2040eULL, 0x8e0d289809154c7aULL,
      0x003009878dbf0094ULL, 0x6b6ae568932a7a6cULL, 0xb04fb3a1a4a36811ULL,
      0xc69b5316de786c12ULL, 0xb601009720fc1f16ULL, 0x24819c943b0af7cfULL,
      0x59512456d97a0739ULL, 0xae3001281c1b1a43ULL, 0x7b3e27907fac043cULL,
      0xd58717059fcff575ULL, 0xdc372161ac15ff78ULL, 0x96898715467e0e60ULL,
      0x8c1d2db6f0fdceaaULL, 0x25a6a92e520a039bULL, 0xd603c597413cd0fbULL,
      0xdf54be34f87bf9f2ULL, 0xce123d2c51d4d331ULL, 0xf1da65c8476f9fa3ULL,
      0x87d0ec90ae02d5bcULL, 0xcc1db207baea51a5ULL, 0x7a464031bbf18ae5ULL,
      0x5a669b6c5a099f03ULL, 0x50a0447d1af94b5cULL, 0xc8e8289cfc12db9fULL,
      0xa21ac44474b69eb8ULL, 0x3ac83df285a9d5ceULL, 0x09de26527392ad37ULL,
      0x836e253d2e2bd1c3ULL, 0x8960a33e26f7f4d6ULL, 0xa96e168bbe7304c2ULL,
      0x1011accd2576d08eULL, 0xf11c2d456840fef7ULL, 0xf1fa07b31240be0eULL,
      0x87773cd35a8a4b6aULL, 0x3aa92f388fa470a8ULL, 0xf6a3751ae165982aULL,
      0xae18463dea73d2caULL, 0x4b709e01d498397aULL, 0xf5a8755a4ab24d59ULL,
      0x45d85da984d225c4ULL, 0xe5a4fd9bbb2ab7a7ULL, 0x529b4ff6e3846135ULL,
      0xcb84be9650c3f9dbULL, 0xd27aa4b27fe93c4bULL, 0xa9209534d29ab247ULL,
      0xccd7341203fa567dULL, 0xb27b0265e455c1a5ULL, 0x9d8ce53627f596d9ULL,
      0xef04635fd0e8a562ULL, 0x90cf8629335a7bdaULL, 0xadd56113bf24128aULL,
      0x9a18504d828fa7a2ULL, 0x9a614e43ea1742e0ULL, 0xc9a6a337ba7c2335ULL,
      0x1f1577d772ab5000ULL,
  };
  uint32_t state = 0x51a8c37d;
  const auto random = [&] {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
  };
  size_t total_tokens = 0;
  for (uint64_t checksum : expected) {
    std::string text;
    const int length = 32 + random() % 48;
    for (int i = 0; i < length; ++i)
      text.append(pieces[random() % pieces.size()]);
    SCOPED_TRACE(testing::PrintToString(text));
    auto ids = tokenizer_->Encode(text);
    ASSERT_TRUE(ids.ok()) << ids.status();
    total_tokens += ids->size();
    uint64_t actual = 14695981039346656037ULL;
    for (int id : *ids) {
      for (int shift = 0; shift < 32; shift += 8) {
        actual ^= (static_cast<uint32_t>(id) >> shift) & 0xff;
        actual *= 1099511628211ULL;
      }
    }
    EXPECT_EQ(actual, checksum);
  }
  EXPECT_EQ(total_tokens, size_t{6045});
}

TEST_F(QwenTokenizerTest, NormalizesNfcAndPreservesCompleteDecodedBytes) {
  auto composed = tokenizer_->Encode("café Å 한글");
  auto decomposed =
      tokenizer_->Encode("cafe\u0301 A\u030a \u1112\u1161\u11ab글");
  ASSERT_TRUE(composed.ok()) << composed.status();
  ASSERT_TRUE(decomposed.ok()) << decomposed.status();
  EXPECT_EQ(*composed, *decomposed);
  auto decoded = tokenizer_->Decode(*decomposed);
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(*decoded, "café Å 한글");

  for (const std::string& text :
       {std::string(), std::string("café 世界 👩🏽‍💻"),
        std::string("before\0after", 12),
        std::string("a<|im_start|>\n<think>hello</think>")}) {
    SCOPED_TRACE(testing::PrintToString(text));
    auto ids = tokenizer_->Encode(text);
    ASSERT_TRUE(ids.ok()) << ids.status();
    auto result = tokenizer_->Decode(*ids);
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(*result, text);
    std::string incremental;
    for (int id : *ids) {
      auto part = tokenizer_->Decode(absl::Span<const int>(&id, 1));
      ASSERT_TRUE(part.ok()) << part.status();
      incremental.append(*part);
    }
    EXPECT_EQ(incremental, text);
  }
}

TEST_F(QwenTokenizerTest, ChatPromptMatchesOfficialTemplate) {
  const std::string prompt =
      "<|im_start|>user\nHello!<|im_end|>\n"
      "<|im_start|>assistant\n<think>\n\n</think>\n\n";
  EXPECT_EQ(QwenTokenizer::ChatPrompt("\u2000 \tHello! \r\n\u00a0"), prompt);
  auto ids = tokenizer_->Encode(prompt);
  ASSERT_TRUE(ids.ok()) << ids.status();
  EXPECT_EQ(*ids,
            (std::vector<int>{248045, 846, 198, 9419, 0, 248046, 198, 248045,
                              74455, 198, 248068, 271, 248069, 271}));
  EXPECT_EQ(
      QwenTokenizer::ChatPrompt("Hello!", true),
      "<|im_start|>system\nReasoning effort is set to xhigh. Please think "
      "carefully through the task, validate key assumptions, consider "
      "plausible "
      "alternatives, and prioritize correctness, consistency, and clarity in "
      "the final answer.<|im_end|>\n"
      "<|im_start|>user\nHello!<|im_end|>\n"
      "<|im_start|>assistant\n<think>\n");
}

TEST_F(QwenTokenizerTest, RejectsMalformedUtf8AndUnknownIds) {
  for (const std::string& malformed :
       {std::string("\xff", 1), std::string("\xc0\xaf", 2),
        std::string("\xed\xa0\x80", 3), std::string("\xf4\x90\x80\x80", 4),
        std::string("\xc3", 1), std::string("\xc3\x28", 2)}) {
    auto ids = tokenizer_->Encode(malformed);
    EXPECT_EQ(ids.status().code(), absl::StatusCode::kInvalidArgument);
  }
  for (int id : {-1, 248077, 248319, std::numeric_limits<int>::max()}) {
    auto text = tokenizer_->Decode(absl::Span<const int>(&id, 1));
    EXPECT_EQ(text.status().code(), absl::StatusCode::kInvalidArgument);
  }
}

TEST(QwenTokenizerLoadTest, ReportsMissingTokenizerFile) {
  auto tokenizer = QwenTokenizer::Load(
      "/tmp/pluto-qwen-tokenizer-deliberately-missing-directory");
  EXPECT_EQ(tokenizer.status().code(), absl::StatusCode::kNotFound);
}

// A complete, tiny tokenizer exercises loading without downloading a model.
// IDs 0..255 represent their exact bytes; 256 is the single merge "a" + "b".
std::string SyntheticTokenizerJson() {
  std::string json = R"({
"truncation":null,"padding":null,
"normalizer":{"type":"NFC"},
"pre_tokenizer":{"type":"Sequence","pretokenizers":[
  {"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},
  {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}
]},
"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
"added_tokens":[)";
  const std::vector<std::string> added = {"<|im_end|>", "<|im_start|>",
                                          "<think>", "</think>"};
  for (size_t i = 0; i < added.size(); ++i) {
    if (i != 0)
      json.push_back(',');
    json.append("{\"id\":" + std::to_string(257 + i) + ",\"content\":\"" +
                added[i] +
                "\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,"
                "\"normalized\":false,\"special\":true}");
  }
  json.append(R"(],
"model":{"type":"BPE","dropout":null,"unk_token":null,
"continuing_subword_prefix":"","end_of_word_suffix":"",
"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
"vocab":{)");
  int replacement = 256;
  for (int byte = 0; byte < 256; ++byte) {
    const bool direct = (byte >= 33 && byte <= 126) ||
                        (byte >= 161 && byte <= 172) || byte >= 174;
    char entry[32];
    std::snprintf(entry, sizeof(entry), "%s\"\\u%04x\":%d",
                  byte == 0 ? "" : ",", direct ? byte : replacement++, byte);
    json.append(entry);
  }
  json.append(R"(,"ab":256},"merges":[["a","b"]]}})");
  return json;
}

class QwenTokenizerFixtureTest : public testing::Test {
 protected:
  void SetUp() override {
    std::string pattern =
        (std::filesystem::path(testing::TempDir()) / "qwen-tokenizer-XXXXXX")
            .string();
    ASSERT_NE(::mkdtemp(pattern.data()), nullptr);
    directory_ = pattern;
  }
  void TearDown() override {
    if (!directory_.empty()) {
      std::error_code error;
      std::filesystem::remove_all(directory_, error);
    }
  }
  bool Write(absl::string_view contents) {
    std::ofstream stream(directory_ / "tokenizer.json", std::ios::binary);
    stream.write(contents.data(), contents.size());
    return stream.good();
  }
  std::filesystem::path directory_;
};

TEST_F(QwenTokenizerFixtureTest, LoadsAndEncodesWithoutAnExternalCheckpoint) {
  ASSERT_TRUE(Write(SyntheticTokenizerJson()));
  auto tokenizer = QwenTokenizer::Load(directory_);
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();
  EXPECT_EQ((*tokenizer)->vocab_size(), 261);
  EXPECT_EQ((*tokenizer)->eos_token_id(), 257);
  auto ids = (*tokenizer)->Encode("abab<|im_end|>e\u0301");
  ASSERT_TRUE(ids.ok()) << ids.status();
  EXPECT_EQ(*ids, (std::vector<int>{256, 256, 257, 195, 169}));
  auto text = (*tokenizer)->Decode(*ids);
  ASSERT_TRUE(text.ok()) << text.status();
  EXPECT_EQ(*text, "abab<|im_end|>é");

  std::vector<int> byte_ids;
  std::string bytes;
  for (int byte = 0; byte < 256; ++byte) {
    byte_ids.push_back(byte);
    bytes.push_back(static_cast<char>(byte));
  }
  auto decoded = (*tokenizer)->Decode(byte_ids);
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_EQ(*decoded, bytes);
  EXPECT_EQ((*tokenizer)->Encode("\xff").status().code(),
            absl::StatusCode::kInvalidArgument);
  const std::vector<int> invalid_ids = {261};
  EXPECT_EQ((*tokenizer)->Decode(invalid_ids).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(QwenTokenizerFixtureTest, SupportsAValidByteOnlyBpeVocabulary) {
  std::string json = SyntheticTokenizerJson();
  const std::string merge = R"("merges":[["a","b"]])";
  json.replace(json.find(merge), merge.size(), R"("merges":[])");
  ASSERT_TRUE(Write(json));
  auto tokenizer = QwenTokenizer::Load(directory_);
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();
  auto ids = (*tokenizer)->Encode("ab");
  ASSERT_TRUE(ids.ok()) << ids.status();
  EXPECT_EQ(*ids, (std::vector<int>{97, 98}));
}

TEST_F(QwenTokenizerFixtureTest, RejectsUnsupportedOrIncompleteDescriptors) {
  const std::vector<std::pair<std::string, std::string>> edits = {
      {R"("type":"NFC")", R"("type":"NFKC")"},
      {R"("type":"BPE")", R"("type":"WordPiece")"},
      {R"("type":"Split")", R"("type":"Whitespace")"},
      {R"("type":"ByteLevel")", R"("type":"ByteFallback")"},
      {R"("behavior":"Isolated")", R"("behavior":"Removed")"},
      {R"("invert":false)", R"("invert":true)"},
      {R"("use_regex":false)", R"("use_regex":true)"},
      {R"("add_prefix_space":false)", R"("add_prefix_space":true)"},
      {R"("normalized":false)", R"("normalized":true)"},
      {R"("lstrip":false)", R"("lstrip":true)"},
      {R"("normalizer":)", R"("unused_normalizer":)"},
      {R"("truncation":null)", R"("truncation":{"max_length":4})"},
      {R"("padding":null)", R"("padding":{"length":8})"},
      {R"("Regex":)", R"("String":)"},
      {"(?i:'s", "(?i:'z"},
      {R"("byte_fallback":false)", R"("byte_fallback":true)"},
      {R"("ignore_merges":false)", R"("ignore_merges":true)"},
      {R"("dropout":null)", R"("dropout":0.1)"},
      {R"("single_word":false,)", ""},
      {R"("behavior":"Isolated",)", ""},
      {R"("trim_offsets":false,)", ""},
      {R"("merges":[["a","b"]])", R"("merges":[["a","b"],["a","b"]])"},
      {R"("\u0021":33)", R"("\u0021":34)"},
      {R"("\u0021":33)", R"("\u0021":9999999)"},
      {R"("\u0021":33)", R"("\u0021":99999999999999999999999)"},
      {R"("<|im_end|>")", R"("<|other_end|>")"},
      {R"("content":"<think>")", R"("content":"")"},
      {R"("content":"<think>")", R"("content":"\ud800")"},
      {R"("content":"<think>")", R"("content":"\ud800 \udc00")"},
  };
  for (const auto& [before, after] : edits) {
    SCOPED_TRACE(before + " -> " + after);
    std::string json = SyntheticTokenizerJson();
    const size_t position = json.find(before);
    ASSERT_NE(position, std::string::npos);
    json.replace(position, before.size(), after);
    ASSERT_TRUE(Write(json));
    auto tokenizer = QwenTokenizer::Load(directory_);
    EXPECT_EQ(tokenizer.status().code(), absl::StatusCode::kInvalidArgument);
  }
}

TEST_F(QwenTokenizerFixtureTest, RejectsMalformedJsonAndTrailingData) {
  for (const std::string& json :
       {std::string("{"), std::string("[]"), std::string("{\"model\":{}}"),
        SyntheticTokenizerJson() + "garbage",
        SyntheticTokenizerJson() + std::string(1, '\0')}) {
    ASSERT_TRUE(Write(json));
    auto tokenizer = QwenTokenizer::Load(directory_);
    EXPECT_EQ(tokenizer.status().code(), absl::StatusCode::kInvalidArgument);
  }
  for (const std::string number : {"01", "-", "1e", "1.", "++1"}) {
    std::string json = SyntheticTokenizerJson();
    json.insert(1, "\"unused\":" + number + ",");
    ASSERT_TRUE(Write(json));
    auto tokenizer = QwenTokenizer::Load(directory_);
    EXPECT_EQ(tokenizer.status().code(), absl::StatusCode::kInvalidArgument);
  }
}

}  // namespace
}  // namespace pluto::tokenizer
