#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/state_vector_codegen.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/utils.h"

namespace pluto::llm::discretized::generator {
namespace {

constexpr size_t kVectorsPerShard = 8192;
constexpr char kRuntime[] =
    "src/llm/experiments/memorize_general_facts/discretized_model/runtime.h";

// A reference into the immutable capture archive; no representative or
// averaged vector can accidentally replace the original BF16 words.
struct OriginalVector {
  int id;
  int boundary;
  const std::vector<uint16_t>* bits;
};

std::string BoundaryName(int stage) {
  if (stage == 0)
    return "token_plus_position_embedding";
  return absl::StrCat(
      "block_", (stage - 1) / 2,
      stage % 2 ? ".after_attention_residual" : ".after_mlp_residual");
}

std::string HexWord(uint16_t word) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result = "0x0000";
  for (int digit = 0; digit < 4; ++digit)
    result[5 - digit] = digits[(word >> (4 * digit)) & 15];
  return result;
}

std::string RenderShard(size_t shard, size_t width,
                        const std::vector<OriginalVector>& vectors) {
  const size_t begin = shard * kVectorsPerShard;
  const size_t end = std::min(begin + kVectorsPerShard, vectors.size());
  std::string body = "namespace {\nconst int kOriginalIds[] = {\n";
  for (size_t row = begin; row < end; ++row)
    absl::StrAppend(&body, vectors[row].id, ",\n");
  body += "};\nconst int kOriginalBoundaries[] = {\n";
  for (size_t row = begin; row < end; ++row)
    absl::StrAppend(&body, vectors[row].boundary, ",\n");
  body += "};\nconst uint16_t kWords[] = {\n";
  for (size_t row = begin; row < end; ++row) {
    absl::StrAppend(&body, "// Original state ", vectors[row].id, ".\n");
    for (uint16_t word : *vectors[row].bits)
      absl::StrAppend(&body, HexWord(word), ", ");
    body += "\n";
  }
  absl::StrAppend(&body,
                  "};\n}  // namespace\n\nStateVectorShard "
                  "GeneratedStateVectorShard",
                  shard, "() {\nreturn {{kOriginalIds, ", end - begin,
                  "}, {kOriginalBoundaries, ", end - begin, "}, {kWords, ",
                  (end - begin) * width, "}};\n}\n");
  return Source(body, "\"state_vectors.h\"",
                "Exact original BF16 activations, for inspection only.\n"
                "Each row preserves its original capture-state ID.");
}

std::string RenderPrinter(const std::vector<std::string>& state_rows,
                          size_t shard_count, int width) {
  std::string body = R"cpp(namespace {
                           std::string BoundaryName(int boundary) {
                             if (boundary == 0)
                               return "token_plus_position_embedding";
                             return "block_" +
                                    std::to_string((boundary - 1) / 2) +
                                    (boundary % 2 ? ".after_attention_residual"
                                                  : ".after_mlp_residual");
                           }
                           // A contiguous slice of the original vectors that
                           // form one compacted state.
                           struct StateVectorGroup {
                             int id;  // Final, possibly relabeled hidden-state
                                      // ID.
                             // One residual boundary, or both sides of a
                             // bijective MLP that now share this symbol.
                             const char* boundary;
                             size_t first;  // Global row offset into the
                                            // sharded archive.
                             size_t count;  // Number of distinct original
                                            // vectors.
                           };
  )cpp";
  body += CppArray("StateVectorGroup", "kStates", state_rows);
  absl::StrAppend(
      &body, "constexpr size_t kStateCount = ", state_rows.size(),
      ";\nconstexpr size_t kWidth = ", width,
      ";\nconstexpr size_t kVectorsPerShard = ", kVectorsPerShard,
      ";\nStateVectorShard Shard(size_t index) {\nswitch (index) {\n");
  for (size_t shard = 0; shard < shard_count; ++shard)
    absl::StrAppend(&body, "case ", shard, ": return GeneratedStateVectorShard",
                    shard, "();\n");
  body += R"cpp(default:
                  return {};
                  }
                  }
                  }  // namespace

                  absl::Status PrintState(DiscreteHiddenState state, std::ostream& output) {
                    const auto* group = std::lower_bound(
                        kStates, kStates + kStateCount, state.value,
                        [](const StateVectorGroup& row, int id) {
                          return row.id < id;
                        });
                    if (group == kStates + kStateCount || group->id != state.value)
                      return absl::NotFoundError(
                          "unknown discrete hidden "
                          "state");

                    // Format independently of the caller's locale, precision,
                    // and flags. Only an unformatted write touches the caller's
                    // stream, so even width survives.
                    std::ostringstream text;
                    text.imbue(std::locale::classic());
                    text << std::setprecision(std::numeric_limits<float>::max_digits10);
                    text << "State " << group->id << " (" << group->boundary
                         << "): " << group->count << " vectors, width "
                         << kWidth << '\n';
                    for (size_t index = group->first;
                         index < group->first + group->count; ++index) {
                      const auto shard = Shard(index / kVectorsPerShard);
                      const size_t row = index % kVectorsPerShard;
                      const auto bits = shard.words.subspan(row * kWidth, kWidth);
                      text << "  original " << shard.original_ids[row] << " ("
                           << BoundaryName(shard.original_boundaries[row])
                           << "): [";
                      for (size_t channel = 0; channel < kWidth; ++channel) {
                        if (channel != 0)
                          text << ", ";
                        // BF16 is exactly the high half of IEEE float,
                        // including signed zero.
                        const auto fp32_bits = static_cast<uint32_t>(bits[channel]) << 16;
                        text << std::bit_cast<float>(fp32_bits);
                      }
                      text << "] bf16=[";
                      for (size_t channel = 0; channel < kWidth; ++channel) {
                        if (channel != 0)
                          text << ", ";
                        text << "0x" << std::hex << std::setfill('0')
                             << std::setw(4)
                             << static_cast<unsigned int>(bits[channel])
                             << std::dec;
                      }
                      text << "]\n";
                    }
                    const std::string rendered = text.str();
                    output.write(rendered.data(),
                                 static_cast<std::streamsize>(rendered.size()));
                    if (!output)
                      return absl::InternalError(
                          "failed writing discrete "
                          "state vectors");
                    return absl::OkStatus();
                  }
  )cpp";
  return "#include <algorithm>\n#include <bit>\n#include <iomanip>\n"
         "#include <limits>\n#include <locale>\n#include <ostream>\n"
         "#include <sstream>\n#include <string>\n" +
         Source(body, "\"state_vectors.h\"",
                "Printing is the only floating-point operation here.\n"
                "The transition network never reads these inspection vectors.");
}

std::string RenderTest(const std::vector<std::string>& state_rows, int width) {
  std::string body = R"cpp(namespace {
                           struct ExpectedGroup {
                             int id;
                             const char* boundary;
                             size_t first;
                             size_t count;
                             uint64_t fingerprint;
                           };
  )cpp";
  body += CppArray("ExpectedGroup", "kExpected", state_rows);
  absl::StrAppend(&body,
                  "constexpr size_t kExpectedCount = ", state_rows.size(),
                  ";\nconstexpr size_t kExpectedWidth = ", width, ";\n");
  body += R"cpp(
    // Fingerprints commit to IDs, original boundary labels, and exact words:
    // four little-endian ID bytes, label bytes plus NUL, then two per BF16
    // word.
    void HashBytes(uint64_t& hash, uint32_t value, int count) {
      for (int byte = 0; byte < count; ++byte) {
        hash ^= (value >> (8 * byte)) & 255;
        hash *= uint64_t{1099511628211};
      }
    }

    std::vector<std::string_view> Components(std::string_view text) {
      std::vector<std::string_view> result;
      while (!text.empty()) {
        const size_t comma = text.find(',');
        auto component = text.substr(0, comma);
        while (component.starts_with(' '))
          component.remove_prefix(1);
        result.push_back(component);
        if (comma == std::string_view::npos)
          break;
        text.remove_prefix(comma + 1);
      }
      return result;
    }

    void CheckOriginalVector(std::string_view line, uint64_t& fingerprint) {
      ASSERT_TRUE(line.starts_with("  original "));
      line.remove_prefix(11);
      const size_t open = line.find(" (");
      ASSERT_NE(open, std::string_view::npos);
      uint32_t original = 0;
      const auto id_result =
          std::from_chars(line.data(), line.data() + open, original);
      ASSERT_EQ(id_result.ec, std::errc{});
      ASSERT_EQ(id_result.ptr, line.data() + open);
      HashBytes(fingerprint, original, 4);
      line.remove_prefix(open + 2);
      const size_t close = line.find("): [");
      ASSERT_NE(close, std::string_view::npos);
      ASSERT_GT(close, 0u);
      for (unsigned char byte : line.substr(0, close))
        HashBytes(fingerprint, byte, 1);
      HashBytes(fingerprint, 0, 1);
      line.remove_prefix(close + 4);
      const size_t marker = line.find("] bf16=[");
      ASSERT_NE(marker, std::string_view::npos);
      const auto decimals = Components(line.substr(0, marker));
      line.remove_prefix(marker + 8);
      ASSERT_TRUE(line.ends_with(']'));
      line.remove_suffix(1);
      const auto words = Components(line);
      ASSERT_EQ(decimals.size(), kExpectedWidth);
      ASSERT_EQ(words.size(), kExpectedWidth);
      for (size_t channel = 0; channel < kExpectedWidth; ++channel) {
        const auto decimal = decimals[channel];
        float value = 0;
        const auto decimal_result = std::from_chars(
            decimal.data(), decimal.data() + decimal.size(), value);
        ASSERT_EQ(decimal_result.ec, std::errc{});
        ASSERT_EQ(decimal_result.ptr, decimal.data() + decimal.size());
        const auto word_text = words[channel];
        ASSERT_EQ(word_text.size(), 6u);
        ASSERT_TRUE(word_text.starts_with("0x"));
        uint32_t word = 0;
        const auto word_result =
            std::from_chars(word_text.data() + 2,
                            word_text.data() + word_text.size(), word, 16);
        ASSERT_EQ(word_result.ec, std::errc{});
        ASSERT_EQ(word_result.ptr, word_text.data() + word_text.size());
        // Bitwise comparison preserves even negative zero and subnormal values.
        EXPECT_EQ(std::bit_cast<uint32_t>(value), word << 16);
        HashBytes(fingerprint, word, 2);
      }
    }
  )cpp";
  body +=
      R"cpp(TEST(GeneratedStateVectors, PrintsEveryOriginalVectorInEveryState) {
              const auto& print = GeneratedModel().print_state;
              ASSERT_TRUE(static_cast<bool>(print));
              for (size_t index = 0; index < kExpectedCount; ++index) {
                const auto& group = kExpected[index];
                SCOPED_TRACE(group.id);
                std::ostringstream output;
                ASSERT_TRUE(print(DiscreteHiddenState{group.id}, output).ok());
                const std::string rendered = output.str();
                std::ostringstream expected;
                expected << "State " << group.id << " (" << group.boundary
                         << "): " << group.count << " vectors, width "
                         << kExpectedWidth << '\n';
                EXPECT_TRUE(rendered.starts_with(expected.str())) << group.id;
                EXPECT_EQ(static_cast<size_t>(std::count(rendered.begin(),
                                                         rendered.end(), '\n')),
                          group.count + 1)
                    << group.id;
                size_t count = 0;
                size_t position = 0;
                while ((position = rendered.find("  original ", position)) !=
                       std::string::npos) {
                  ++count;
                  position += 11;
                }
                EXPECT_EQ(count, group.count) << group.id;
                std::istringstream input(rendered);
                std::string line;
                ASSERT_TRUE(static_cast<bool>(std::getline(input, line)));
                uint64_t fingerprint = 14695981039346656037ULL;
                for (size_t row = 0; row < group.count; ++row) {
                  ASSERT_TRUE(static_cast<bool>(std::getline(input, line)));
                  ASSERT_NO_FATAL_FAILURE(CheckOriginalVector(line, fingerprint));
                }
                EXPECT_FALSE(static_cast<bool>(std::getline(input, line)));
                EXPECT_EQ(fingerprint, group.fingerprint);
              }
            }

            TEST(GeneratedStateVectors, RejectsUnknownStatesWithoutPrinting) {
              const auto& print = GeneratedModel().print_state;
              ASSERT_TRUE(static_cast<bool>(print));
              std::ostringstream output;
              EXPECT_EQ(print(DiscreteHiddenState{-1}, output).code(),
                        absl::StatusCode::kNotFound);
              EXPECT_TRUE(output.str().empty());
            }

            class CommaPunctuation final : public std::numpunct<char> {
             protected:
              char do_decimal_point() const override { return ','; }
            };

            TEST(GeneratedStateVectors, PreservesCallerFormattingAndReportsWriteFailure) {
              const auto& print = GeneratedModel().print_state;
              ASSERT_TRUE(static_cast<bool>(print));
              if (kExpectedCount == 0)
                return;
              std::ostringstream expected;
              ASSERT_TRUE(print(DiscreteHiddenState{kExpected[0].id}, expected).ok());
              std::ostringstream output;
              output.imbue(std::locale(std::locale::classic(), new CommaPunctuation));
              output << std::hex << std::showbase << std::setprecision(2);
              output.width(17);
              output.fill('*');
              const auto flags = output.flags();
              const auto locale = output.getloc();
              ASSERT_TRUE(print(DiscreteHiddenState{kExpected[0].id}, output).ok());
              EXPECT_EQ(output.str(), expected.str());
              EXPECT_EQ(output.flags(), flags);
              EXPECT_EQ(output.getloc(), locale);
              EXPECT_EQ(output.precision(), 2);
              EXPECT_EQ(output.width(), 17);
              EXPECT_EQ(output.fill(), '*');
              std::ostringstream broken;
              broken.setstate(std::ios_base::badbit);
              EXPECT_EQ(
                  print(DiscreteHiddenState{kExpected[0].id}, broken).code(),
                  absl::StatusCode::kInternal);
            }
            }  // namespace
      )cpp";
  return "#include <algorithm>\n#include <bit>\n#include <charconv>\n"
         "#include <cstdint>\n#include <iomanip>\n#include <locale>\n"
         "#include <sstream>\n#include <string>\n#include <string_view>\n"
         "#include <system_error>\n#include <vector>\n"
         "#include \"gtest/gtest.h\"\n" +
         Source(body, "\"model.h\"",
                "Check all original IDs and exact BF16 words via the public "
                "model callback.\nDecimal vectors must also round-trip to "
                "the original FP32 bits.");
}

void HashBytes(uint64_t& hash, uint32_t value, int count) {
  for (int byte = 0; byte < count; ++byte) {
    hash ^= (value >> (8 * byte)) & 255;
    hash *= uint64_t{1099511628211};
  }
}

}  // namespace

absl::StatusOr<std::map<std::string, std::string>> RenderStateVectors(
    const CapturedModel& model, const CapturedStateVectors& archive) {
  if (model.metadata.width <= 0 || model.metadata.vocab_size < 0 ||
      model.metadata.layers < 0)
    return absl::InvalidArgumentError("invalid state-vector model dimensions");
  const bool has_boundaries = !archive.original_boundaries.empty();
  if (has_boundaries &&
      archive.original_boundaries.size() != archive.original_states.size())
    return absl::InvalidArgumentError(
        "original boundary metadata must cover the vector archive exactly");
  std::map<int, const CapturedState*> states;
  for (const auto& state : model.states) {
    if (state.id < model.metadata.vocab_size || state.boundary < 0 ||
        state.boundary > 2 * static_cast<int64_t>(model.metadata.layers))
      return absl::InvalidArgumentError("invalid state-vector ID or boundary");
    if (state.shared_boundary &&
        (state.boundary % 2 != 1 ||
         *state.shared_boundary != static_cast<int64_t>(state.boundary) + 1 ||
         *state.shared_boundary >
             2 * static_cast<int64_t>(model.metadata.layers)))
      return absl::InvalidArgumentError("invalid shared MLP state boundary");
    if (state.shared_boundary && !has_boundaries)
      return absl::InvalidArgumentError(
          "shared MLP states require original vector boundaries");
    if (!states.emplace(state.id, &state).second)
      return absl::InvalidArgumentError("duplicate state-vector ID");
  }
  absl::flat_hash_set<int> seen;
  std::vector<OriginalVector> originals;
  std::vector<std::string> state_rows;
  std::vector<std::string> expected_rows;
  for (const auto& [id, state] : states) {
    if (!state->members || state->members->empty())
      return absl::InvalidArgumentError(
          "state-vector printing requires complete original membership");
    auto members = *state->members;
    std::sort(members.begin(), members.end());
    std::string boundary_names = BoundaryName(state->boundary);
    if (state->shared_boundary)
      absl::StrAppend(&boundary_names, " + ",
                      BoundaryName(*state->shared_boundary));
    state_rows.push_back(absl::StrCat("{", id, ", ", Literal(boundary_names),
                                      ", ", originals.size(), ", ",
                                      members.size(), "}"));
    uint64_t fingerprint = 14695981039346656037ULL;
    bool saw_primary = false;
    bool saw_shared = false;
    for (int original : members) {
      if (original < model.metadata.vocab_size || !seen.insert(original).second)
        return absl::InvalidArgumentError(
            "invalid or duplicate original state-vector member");
      auto found = archive.original_states.find(original);
      if (found == archive.original_states.end())
        return absl::InvalidArgumentError(
            absl::StrCat("missing original state vector ", original));
      int original_boundary = state->boundary;
      if (has_boundaries) {
        const auto boundary = archive.original_boundaries.find(original);
        if (boundary == archive.original_boundaries.end() ||
            !state->HasBoundary(boundary->second))
          return absl::InvalidArgumentError(
              "original vector boundary disagrees with its compacted state");
        original_boundary = boundary->second;
      }
      saw_primary |= original_boundary == state->boundary;
      saw_shared |= state->shared_boundary == original_boundary;
      if (found->second.size() != static_cast<size_t>(model.metadata.width))
        return absl::InvalidArgumentError(
            "original state-vector width disagrees with model");
      for (uint16_t word : found->second)
        if ((word & 0x7f80) == 0x7f80)
          return absl::InvalidArgumentError(
              "original state vector contains nonfinite BF16 data");
      originals.push_back({original, original_boundary, &found->second});
      HashBytes(fingerprint, static_cast<uint32_t>(original), 4);
      for (unsigned char byte : BoundaryName(original_boundary))
        HashBytes(fingerprint, byte, 1);
      HashBytes(fingerprint, 0, 1);
      for (uint16_t word : found->second)
        HashBytes(fingerprint, word, 2);
    }
    if (state->shared_boundary && (!saw_primary || !saw_shared))
      return absl::InvalidArgumentError(
          "shared state provenance must include both original boundaries");
    expected_rows.push_back(
        absl::StrCat(state_rows.back().substr(0, state_rows.back().size() - 1),
                     ", ", fingerprint, "ULL}"));
  }
  if (seen.size() != archive.original_states.size())
    return absl::InvalidArgumentError(
        "state-vector archive contains unused original states");

  const size_t shard_count = originals.size() / kVectorsPerShard +
                             (originals.size() % kVectorsPerShard != 0);
  std::string header = absl::StrCat(
      "// Private generated vector-inspection API; do not edit.\n#pragma once\n"
      "#include <cstdint>\n#include <iosfwd>\n#include \"",
      kRuntime,
      "\"\nnamespace pluto::llm::discretized::gen::internal {\n"
      "// Immutable original-vector rows, backed by static generated arrays.\n"
      "struct StateVectorShard {\n"
      "  absl::Span<const int> original_ids;  // One capture-state ID per "
      "row.\n"
      "  absl::Span<const int> original_boundaries;  // Original residual "
      "stage per row.\n"
      "  absl::Span<const uint16_t> words;  // Row-major exact BF16 words.\n"
      "};\n"
      "// Prints every original activation in one compacted hidden state.\n"
      "// Unknown IDs fail without output; formatting on output is preserved.\n"
      "absl::Status PrintState(DiscreteHiddenState state, std::ostream& "
      "output);\n");
  std::map<std::string, std::string> files;
  for (size_t shard = 0; shard < shard_count; ++shard) {
    absl::StrAppend(&header, "StateVectorShard GeneratedStateVectorShard",
                    shard, "();\n");
    files[absl::StrCat("state_vectors_", shard, ".cc")] =
        RenderShard(shard, model.metadata.width, originals);
  }
  header += "}  // namespace pluto::llm::discretized::gen::internal\n";
  files["state_vectors.h"] = std::move(header);
  files["state_vectors.cc"] =
      RenderPrinter(state_rows, shard_count, model.metadata.width);
  files["state_vectors_test.cc"] =
      RenderTest(expected_rows, model.metadata.width);
  return files;
}

}  // namespace pluto::llm::discretized::generator
