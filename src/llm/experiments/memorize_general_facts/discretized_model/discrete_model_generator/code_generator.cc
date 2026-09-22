#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/code_generator.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_model_util.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/layer_codegen.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/utils.h"
#include "src/util/status_macros.h"

namespace pluto::llm::discretized::generator {
namespace {

constexpr absl::string_view kRuntime =
    "src/llm/experiments/memorize_general_facts/discretized_model/runtime.h";
constexpr absl::string_view kPackage =
    "//src/llm/experiments/memorize_general_facts/discretized_model";
constexpr absl::string_view kGen = "pluto::llm::discretized::gen";
constexpr absl::string_view kInternal =
    "pluto::llm::discretized::gen::internal";

std::string BoundaryName(int stage) {
  if (stage == 0)
    return "token_plus_position_embedding";
  return absl::StrCat(
      "block_", (stage - 1) / 2,
      stage % 2 ? ".after_attention_residual" : ".after_mlp_residual");
}

absl::StatusOr<std::string> RenderEncoder(
    const CapturedModel& model, const std::vector<std::string>& names) {
  std::map<std::string, std::vector<int>> prefixes;
  for (const auto& sample : model.samples) {
    std::string text;
    std::vector<int> ids;
    for (int token : sample.tokens) {
      text += model.metadata.vocabulary[token].bytes;
      ids.push_back(token);
      auto [it, inserted] = prefixes.emplace(text, ids);
      if (!inserted && it->second != ids)
        return absl::InvalidArgumentError(
            "ambiguous captured text-prefix encoding");
    }
  }
  std::vector<std::string> tokens, rows;
  for (const auto& [text, ids] : prefixes) {
    rows.push_back(absl::StrCat("{{", Literal(text), ", ", text.size(), "}, ",
                                tokens.size(), ", ", ids.size(), "}"));
    for (int token : ids)
      tokens.push_back(absl::StrCat("vocab::", names[token]));
  }
  std::string body =
      "namespace {\nstruct PromptRow { absl::string_view text; size_t offset; "
      "size_t length; };\n";
  body += CppArray("DiscreteToken", "kTokens", tokens);
  body += CppArray("PromptRow", "kPrompts", rows) + "}\n";
  body +=
      R"cpp(absl::StatusOr<std::vector<DiscreteToken>> EncodeGeneratedPrompt(
                absl::string_view text) {
              auto row = std::lower_bound(
                  std::begin(kPrompts), std::end(kPrompts), text,
                  [](const PromptRow& r, absl::string_view key) {
                    return r.text < key;
                  });
              if (row == std::end(kPrompts) || row->text != text)
                return absl::NotFoundError(
                    "unsupported text encoding: use a captured corpus prefix "
                    "at a token boundary, or --token_ids");
              return std::vector<DiscreteToken>(
                  kTokens + row->offset, kTokens + row->offset + row->length);
            })cpp";
  return "#include <algorithm>\n#include <iterator>\n" +
         Source(body, "\"cli_support.h\"", "", true);
}

std::string RenderVerification(const CapturedModel& model,
                               const std::vector<std::string>& names) {
  std::vector<std::string> tokens, rows;
  for (const auto& sample : model.samples) {
    rows.push_back(
        absl::StrCat("{", tokens.size(), ", ", sample.tokens.size(), "}"));
    for (int token : sample.tokens)
      tokens.push_back(absl::StrCat("vocab::", names[token]));
  }
  std::string body =
      "namespace {\nstruct Sample { size_t offset; size_t length; };\n";
  body += CppArray("DiscreteToken", "kExpectedTokens", tokens);
  body += CppArray("Sample", "kSamples", rows) + "}\n";
  body += R"cpp(absl::Status VerifyGeneratedModel(const DiscreteModel& model,
                                                  std::ostream& output) {
                  size_t targets = 0;
                  size_t sentences = 0;
                  for (const auto& sample : kSamples) {
                    const auto original = absl::MakeConstSpan(
                        kExpectedTokens + sample.offset, sample.length);
                    auto generated = Generate(
                        model, original.first(model.prompt_token_count),
                        sample.length - model.prompt_token_count + 1);
                    if (!generated.ok())
                      return generated.status();
                    std::vector<DiscreteToken> expected(
                        original.begin() + model.prompt_token_count,
                        original.end());
                    expected.push_back(model.eos_token);
                    if (*generated != expected) {
                      output << "mismatch at verification sentence " << sentences + 1 << "\n";
                      return absl::DataLossError(
                          "integer autoregressive completion differs from "
                          "independent fixture");
                    }
                    targets += expected.size();
                    ++sentences;
                  }
                  output << "errors=0/" << targets
                         << " exact_sentences=" << sentences << "/" << sentences
                         << " eos=" << sentences
                         << " autoregressive=true integer_only=true\n";
                  return absl::OkStatus();
                })cpp";
  return "#include <ostream>\n" + Source(body, "\"cli_support.h\"", "", true);
}

std::string RenderBuild(int layers, bool compact) {
  std::vector<std::string> sources = {"entry.cc", "model.cc",
                                      "language_modeling_head.cc",
                                      "vocabulary.cc", "tables.h"};
  for (int block = 0; block < layers; ++block) {
    sources.push_back(absl::StrCat("attention_", block, ".cc"));
    sources.push_back(absl::StrCat("mlp_", block, ".cc"));
  }
  std::sort(sources.begin(), sources.end());
  std::string text =
      R"build(# Generated ordinary C++ targets; no build-time generation or GPU dependency.
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

cc_library(
    name = "vocabulary_tokens",
    hdrs = ["vocabulary_tokens.h"],
    visibility = ["//visibility:private"],
)build";
  absl::StrAppend(&text, "    deps = [\"", kPackage, ":runtime\"],\n)\n");
  text += "cc_library(\n    name = \"model\",\n    srcs = [\n";
  for (const auto& source : sources)
    absl::StrAppend(&text, "        \"", source, "\",\n");
  text += R"build(    ],
    hdrs = ["model.h"],
    copts = ["-fvisibility=hidden", "-fvisibility-inlines-hidden"],
    strip_include_prefix = ".",
    include_prefix = "pluto/discretized/gen",
    visibility = ["//visibility:public"],
)build";
  absl::StrAppend(&text, "    deps = [\":vocabulary_tokens\", \"", kPackage,
                  ":runtime\"],\n)\n");
  text += R"build(cc_library(
    name = "cli_support",
    hdrs = ["cli_support.h"],
    strip_include_prefix = ".",
    include_prefix = "pluto/discretized/gen/internal",
    visibility = ["//visibility:private"],
)build";
  absl::StrAppend(&text, "    deps = [\"", kPackage, ":runtime\"],\n)\n");
  for (const std::string name : {"prompt_encoder", "verification"}) {
    absl::StrAppend(&text, "cc_library(\n    name = \"", name,
                    "\",\n    srcs = [\"", name, ".cc\"],\n",
                    "    copts = [\"-fvisibility=hidden\", "
                    "\"-fvisibility-inlines-hidden\"],\n",
                    "    # Private entry points link into the caller, not a "
                    "separate shared object.\n",
                    "    linkstatic = True,\n    visibility = "
                    "[\"//visibility:private\"],\n",
                    "    deps = [\":cli_support\", \":vocabulary_tokens\", \"",
                    kPackage, ":runtime\"],\n)\n");
  }
  absl::StrAppend(
      &text, "cc_binary(\n    name = \"discretized_model\",\n    srcs = [\"",
      kPackage,
      ":main.cc\"],\n    deps = [\":cli_support\", \":model\", "
      "\":prompt_encoder\", \":verification\", \"",
      kPackage, ":runtime\"],\n)\n",
      "cc_test(\n    name = \"generated_model_test\",\n    srcs = "
      "[\"generated_model_test.cc\"],\n",
      "    deps = [\":cli_support\", \":model\", \":verification\", "
      "\":vocabulary_tokens\", \"",
      kPackage, ":runtime\", \"@googletest//:gtest_main\"],\n)\n");
  if (compact)
    absl::StrAppend(&text,
                    "\n# Independent per-boundary fixtures are test-only, "
                    "never linked by the CLI.\n",
                    "cc_test(\n    name = \"generated_transition_test\",\n    "
                    "srcs = [\"generated_transition_test.cc\"],\n",
                    "    deps = [\":model\", \":vocabulary_tokens\", \"",
                    kPackage,
                    ":runtime\", \"@googletest//:gtest_main\"],\n)\n");
  return text;
}

absl::StatusOr<std::string> RenderStateIndex(const CapturedModel& model) {
  std::map<std::pair<int, int>, int> entries;
  for (const auto& row : model.position_embedding.transitions)
    entries[{row.token, row.position}] = row.output;
  const int layers = model.metadata.layers;
  std::vector<std::map<std::vector<int>, int>> attention(layers);
  std::vector<std::map<int, int>> mlp(layers);
  for (int block = 0; block < layers; ++block) {
    for (const auto& row : model.transformers[block].attention.transitions)
      attention[block][row.prefix] = row.output;
    for (const auto& row : model.transformers[block].mlp.transitions)
      mlp[block][row.input] = row.output;
  }
  std::map<int, size_t> counts;
  // Store complete readable examples, not a generic serialization tree.
  // C++ literal escaping keeps embedded tabs/newlines and arbitrary token bytes
  // inside one TSV field without losing their exact byte values.
  std::map<int, std::vector<std::string>> examples;
  auto observe = [&](const std::vector<int>& states,
                     const std::vector<std::string>& prefixes) {
    for (size_t position = 0; position < states.size(); ++position) {
      const int state = states[position];
      ++counts[state];
      auto& selected = examples[state];
      if (selected.size() < 3 &&
          std::find(selected.begin(), selected.end(), prefixes[position]) ==
              selected.end())
        selected.push_back(prefixes[position]);
    }
  };
  for (const auto& sample : model.samples) {
    std::string text;
    std::vector<std::string> prefixes;
    std::vector<int> states;
    std::vector<int> ids;
    for (size_t position = 0; position < sample.tokens.size(); ++position) {
      const int token = sample.tokens[position];
      ids.push_back(token);
      text += model.metadata.vocabulary[token].bytes;
      std::string context = absl::StrCat(
          "tokens=[", absl::StrJoin(ids, ","),
          "]; text=", Literal(absl::string_view(text).substr(0, 160)));
      if (text.size() > 160)
        context += "; truncated_after_bytes=160";
      prefixes.push_back(std::move(context));
      auto entry = entries.find({token, position});
      if (entry == entries.end())
        return absl::InvalidArgumentError(
            "state-index replay encountered an unsupported entry");
      states.push_back(entry->second);
    }
    observe(states, prefixes);
    for (int block = 0; block < layers; ++block) {
      std::vector<int> next, prefix;
      for (int state : states) {
        prefix.push_back(state);
        auto row = attention[block].find(prefix);
        if (row == attention[block].end())
          return absl::InvalidArgumentError(
              "state-index replay encountered an unsupported attention "
              "history");
        next.push_back(row->second);
      }
      states = std::move(next);
      observe(states, prefixes);
      for (int& state : states) {
        auto row = mlp[block].find(state);
        if (row == mlp[block].end())
          return absl::InvalidArgumentError(
              "state-index replay encountered an unsupported MLP state");
        state = row->second;
      }
      observe(states, prefixes);
    }
  }
  std::string output =
      "# Inspection only; never linked into or read by inference.\n"
      "# Examples are empirical captured token-prefix contexts, not semantic "
      "labels.\n"
      "# Occurrences count each real sample position once at its named "
      "boundary, including prompt positions; no padding or repeated "
      "autoregressive passes.\n"
      "# Each example has complete compact token IDs and text preview of at "
      "most 160 original bytes; at most three distinct examples per state.\n"
      "state_id\tboundary\tobserved_"
      "occurrences\tempirical_prefix_examples\toriginal_member_count\n";
  // Sort only references: membership lists can be large and need no copying.
  std::vector<const CapturedState*> states;
  for (const auto& state : model.states)
    states.push_back(&state);
  std::sort(states.begin(), states.end(),
            [](const auto* a, const auto* b) { return a->id < b->id; });
  for (const auto* state_ptr : states) {
    const auto& row = *state_ptr;
    const int state = row.id;
    absl::StrAppend(
        &output, state, "\t", BoundaryName(row.boundary), "\t", counts[state],
        "\t", absl::StrJoin(examples[state], " | "), "\t",
        row.members ? absl::StrCat(row.members->size()) : "unknown", "\n");
  }
  return output;
}

}  // namespace

absl::StatusOr<FileMap> RenderModel(const CapturedModel& model,
                                    bool include_state_index,
                                    bool compact_transitions) {
  RETURN_IF_ERROR(ValidateModel(model));
  const int layers = model.metadata.layers, eos = model.metadata.eos_token;
  const auto& vocab = model.metadata.vocabulary;
  std::vector<std::string> names;
  for (size_t i = 0; i < vocab.size(); ++i)
    names.push_back(TokenName(vocab[i].bytes, i, eos));
  FileMap files;
  std::string declarations =
      "PositionEmbedding& GeneratedPositionEmbedding();\nabsl::Span<const "
      "VocabularyRow> GeneratedVocabulary();\nMap& "
      "GeneratedLanguageModelingHead();\n";
  for (int block = 0; block < layers; ++block)
    absl::StrAppend(&declarations, "CausalAttention& GeneratedAttention", block,
                    "();\nMap& GeneratedMlp", block, "();\n");
  files["tables.h"] = absl::StrCat(
      "// Private generated boundary declarations.\n#pragma once\n#include \"",
      kRuntime, "\"\nnamespace ", kInternal, " {\n", declarations,
      "}  // namespace ", kInternal, "\n");
  files["model.h"] = absl::StrCat(
      "// Generated public model factory; do not edit.\n#pragma once\n#include "
      "\"",
      kRuntime, "\"\n\nnamespace ", kGen,
      " {\n// Returns the generated model and its stateless operations with "
      "static lifetime.\n",
      "[[gnu::visibility(\"default\")]] const DiscreteModel& "
      "GeneratedModel();\n}  // namespace ",
      kGen, "\n");
  files["cli_support.h"] = absl::StrCat(
      "// Private declarations for the generated CLI and its tests; do not "
      "edit.\n#pragma once\n#include <iosfwd>\n#include \"",
      kRuntime, "\"\n\nnamespace ", kInternal, " {\n",
      "// Encodes only captured text prefixes; independent of production "
      "inference.\n",
      "absl::StatusOr<std::vector<DiscreteToken>> "
      "EncodeGeneratedPrompt(absl::string_view text);\n",
      "// Checks complete corpus continuations using independently linked "
      "fixtures.\n",
      "absl::Status VerifyGeneratedModel(const DiscreteModel& model, "
      "std::ostream& output);\n",
      "}  // namespace ", kInternal, "\n");
  std::string tokens_header = absl::StrCat(
      "// Generated by generate_discretized_model; do not edit.\n#pragma "
      "once\n#include \"",
      kRuntime, "\"\n\nnamespace ", kInternal, "::vocab {\n",
      "// One constant per compact vocabulary token; values are not GPT-2 "
      "IDs.\n",
      "// Names retain token case, spell whitespace/punctuation, and end "
      "with\n",
      "// the compact ID to keep every spelling unique. Long stems are "
      "capped.\n",
      "// Comments show token bytes and original GPT-2 IDs; Eos names the "
      "readout\n",
      "// terminator. Internal residual-state IDs intentionally remain "
      "numeric.\n");
  for (size_t i = 0; i < vocab.size(); ++i)
    absl::StrAppend(&tokens_header, "// ", Literal(vocab[i].bytes),
                    "; original GPT-2 ID ", vocab[i].original_id,
                    ".\ninline constexpr DiscreteToken ", names[i], "{", i,
                    "};\n");
  files["vocabulary_tokens.h"] =
      absl::StrCat(tokens_header, "}  // namespace ", kInternal, "::vocab\n");
  std::vector<std::string> rows;
  std::string body;
  rows.clear();
  for (const auto& row : vocab) {
    const std::string& bytes = row.bytes;
    rows.push_back(absl::StrCat("{", row.original_id, ", {", Literal(bytes),
                                ", ", bytes.size(), "}}"));
  }
  body = "namespace {\n" + CppArray("VocabularyRow", "kRows", rows) + "}\n";
  absl::StrAppend(
      &body,
      "absl::Span<const VocabularyRow> GeneratedVocabulary() { return {kRows, ",
      rows.size(), "}; }");
  files["vocabulary.cc"] =
      Source(body, "\"tables.h\"",
             "Vocabulary rows retain exact original GPT-2 bytes and IDs.");
  body = "const DiscreteModel& GeneratedModel() {\n";
  if (layers) {
    rows.clear();
    for (int block = 0; block < layers; ++block)
      rows.push_back(absl::StrCat("{internal::GeneratedAttention", block,
                                  "(), internal::GeneratedMlp", block, "()}"));
    body += "  static " + CppArray("Transformer", "kTransformers", rows);
  }
  absl::StrAppend(
      &body, "  static const DiscreteModel model{1024, ",
      model.metadata.prompt_tokens, ", internal::vocab::", names[eos],
      ", internal::GeneratedVocabulary(), ",
      layers ? absl::StrCat("{kTransformers, ", layers, "}") : "{}",
      ", internal::GeneratedLanguageModelingHead(), "
      "internal::GeneratedPositionEmbedding()};\n  return model;\n}");
  files["model.cc"] =
      Source(body, "\"model.h\"",
             "Original boundary order, with no cross-layer folding.", true,
             kGen, {"tables.h"});
  ASSIGN_OR_RETURN(files["prompt_encoder.cc"], RenderEncoder(model, names));
  files["verification.cc"] = RenderVerification(model, names);
  std::string test = absl::StrCat(
      "#include <sstream>\n#include \"model.h\"\n#include "
      "\"cli_support.h\"\n#include \"vocabulary_tokens.h\"\n#include "
      "\"gtest/gtest.h\"\nnamespace ",
      kGen, " {\n",
      "TEST(GeneratedIntegerModel, NamedVocabularyCoversEveryCompactId) {\n  "
      "constexpr DiscreteToken kTokens[] = {\n");
  for (const auto& name : names)
    absl::StrAppend(&test, "internal::vocab::", name, ",\n");
  test += R"cpp(}
                ;
                ASSERT_EQ(absl::MakeConstSpan(kTokens).size(),
                          GeneratedModel().vocabulary.size());
                for (size_t token = 0; token < absl::MakeConstSpan(kTokens).size(); ++token)
                  EXPECT_EQ(kTokens[token].value, token);
                }
                TEST(GeneratedIntegerModel, IndependentAutoregressiveCorpusVerification) {
                  const auto& model = GeneratedModel();
                  ASSERT_TRUE(ValidateModel(model).ok());
                  std::ostringstream report;
                  auto result = internal::VerifyGeneratedModel(model, report);
                  EXPECT_TRUE(result.ok()) << result << "\n" << report.str();
                }
  )cpp";
  files["generated_model_test.cc"] =
      absl::StrCat(test, "}  // namespace ", kGen, "\n");
  files["BUILD.bazel"] = RenderBuild(layers, compact_transitions);
  if (include_state_index) {
    ASSIGN_OR_RETURN(files["state_index.tsv"], RenderStateIndex(model));
    bool all_members = true;
    for (const auto& row : model.states)
      all_members &= row.members.has_value();
    if (all_members) {
      std::map<int, CapturedState> states;
      for (const auto& row : model.states)
        states[row.id] = row;
      std::string members =
          "# Inspection only: IDs refer to the exact, uncompacted "
          "baseline.\nstate_id\tboundary\toriginal_state_ids\n";
      for (const auto& [id, row] : states) {
        std::vector<int> original = *row.members;
        std::sort(original.begin(), original.end());
        absl::StrAppend(&members, id, "\t", BoundaryName(row.boundary), "\t[",
                        absl::StrJoin(original, ","), "]\n");
      }
      files["state_members.tsv"] = std::move(members);
    }
  }
  RETURN_IF_ERROR(RenderLayerSources(model, names, compact_transitions, files));
  return files;
}

absl::Status PublishFiles(const FileMap& files,
                          const std::filesystem::path& destination) {
  const auto name = destination.filename().string();
  if (name.empty() || name == "." || name == "..")
    return absl::InvalidArgumentError("output must name a fresh directory");
  std::error_code error;
  const auto parent = std::filesystem::canonical(
      destination.has_parent_path() ? destination.parent_path()
                                    : std::filesystem::path("."),
      error);
  if (error)
    return absl::InvalidArgumentError(
        absl::StrCat("invalid output parent: ", error.message()));
  const auto target = parent / name;
  struct stat info;
  if (::lstat(target.c_str(), &info) == 0)
    return absl::AlreadyExistsError(
        absl::StrCat("refusing existing generated output: ", target.string()));
  if (errno != ENOENT)
    return absl::UnknownError(
        absl::StrCat("cannot inspect output: ", std::strerror(errno)));
  std::string staging_name =
      (parent / absl::StrCat(".", name, ".emit-XXXXXX")).string();
  if (!::mkdtemp(staging_name.data()))
    return absl::UnknownError(absl::StrCat("cannot create staging directory: ",
                                           std::strerror(errno)));
  const std::filesystem::path staging(staging_name);
  // Remove only our own fresh staging directory on failure. renameat2 below is
  // atomic and refuses clobbering a destination created by a concurrent writer.
  auto cleanup = [&] {
    std::error_code ignored;
    std::filesystem::remove_all(staging, ignored);
  };
  for (const auto& [filename, contents] : files) {
    if (std::filesystem::path(filename).filename() != filename ||
        filename == "." || filename == "..") {
      cleanup();
      return absl::InvalidArgumentError(
          "generated file must have a plain relative filename");
    }
    const auto path = staging / filename;
    const int fd =
        ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) {
      const std::string message = std::strerror(errno);
      cleanup();
      return absl::UnknownError(
          absl::StrCat("cannot create generated file: ", message));
    }
    size_t written = 0;
    int failure = 0;
    while (written < contents.size()) {
      const auto count =
          ::write(fd, contents.data() + written, contents.size() - written);
      if (count < 0 && errno == EINTR)
        continue;
      if (count <= 0) {
        failure = count < 0 ? errno : EIO;
        break;
      }
      written += count;
    }
    if (!failure && ::fsync(fd) != 0)
      failure = errno;
    if (::close(fd) != 0 && !failure)
      failure = errno;
    if (failure) {
      cleanup();
      return absl::UnknownError(absl::StrCat("cannot write generated file: ",
                                             std::strerror(failure)));
    }
  }
  if (::syscall(SYS_renameat2, AT_FDCWD, staging.c_str(), AT_FDCWD,
                target.c_str(), 1) != 0) {
    const int failure = errno;
    cleanup();
    return failure == EEXIST
               ? absl::AlreadyExistsError("refusing existing generated output")
               : absl::UnknownError(
                     absl::StrCat("cannot publish generated output: ",
                                  std::strerror(failure)));
  }
  return absl::OkStatus();
}

absl::Status EmitModel(const CapturedModel& model,
                       const std::filesystem::path& destination,
                       bool include_state_index, bool compact_transitions) {
  ASSIGN_OR_RETURN(auto files,
                   RenderModel(model, include_state_index, compact_transitions));
  return PublishFiles(files, destination);
}

}  // namespace pluto::llm::discretized::generator
