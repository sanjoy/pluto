#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_logic.h"

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_attention_logic.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_pointwise.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_transition_tests.h"
#include "src/util/status_macros.h"

namespace pluto::llm::discretized::generator {
namespace {

// Reports deliberately use plain text. The in-memory JSON representation is
// an implementation detail, not another generated interchange artifact.
std::string ReportValue(const Json& value) {
  if (value.is_string())
    return value.get<std::string>();
  if (value.is_array()) {
    std::vector<std::string> parts;
    for (const auto& item : value)
      parts.push_back(ReportValue(item));
    return parts.empty() ? "(none)" : absl::StrJoin(parts, ", ");
  }
  return value.dump();
}

}  // namespace

absl::Status RenderCompact(const Json& model,
                           const std::vector<std::string>& token_names,
                           FileMap& files) {
  TokenNames names;
  std::vector<std::string> qualified_names;
  for (size_t index = 0; index < token_names.size(); ++index) {
    qualified_names.push_back(absl::StrCat("vocab::", token_names[index]));
    names[index] = qualified_names.back();
  }
  std::string report =
      "Exact individual transition domains; no cross-layer folding.\n"
      "Source measurements are unformatted bytes, not executable size.\n";
  auto install = [&](const std::string& filename,
                     const RenderedTransition& rendered,
                     absl::string_view interface, absl::string_view factory,
                     absl::string_view description,
                     bool vocabulary = false) -> absl::Status {
    auto found = files.find(filename);
    if (found == files.end())
      return absl::InternalError(
          absl::StrCat("missing plain generated file: ", filename));
    size_t before = found->second.size();
    found->second =
        Source(TransitionObject(rendered.source, interface, factory),
               "\"tables.h\"", description, vocabulary);
    absl::StrAppend(&report, "\n", filename, "\n");
    for (const auto& [key, value] : rendered.stats.items())
      absl::StrAppend(&report, "  ", key, ": ", ReportValue(value), "\n");
    absl::StrAppend(
        &report, "  unformatted_source_bytes_before: ", before,
        "\n  unformatted_source_bytes_after: ", found->second.size(), "\n");
    return absl::OkStatus();
  };
  ASSIGN_OR_RETURN(auto entry, RenderEntry("Lookup", model["entry"], names));
  RETURN_IF_ERROR(install(
      "entry.cc", entry, "PositionEmbedding", "GeneratedPositionEmbedding",
      "Entry: compact token and absolute position -> residual symbol.\n"
      "A default symbol plus exceptional positions describes each token;\n"
      "support masks reject every token/position absent from the source.",
      true));
  for (size_t block = 0; block < model["attention"].size(); ++block) {
    ASSIGN_OR_RETURN(auto attention,
                     RenderAttention("Lookup", model["attention"][block]));
    RETURN_IF_ERROR(
        install(absl::StrCat("attention_", block, ".cc"), attention,
                "CausalAttention", absl::StrCat("GeneratedAttention", block),
                absl::StrCat("Block ", block,
                             ": exact causal-history decision program.\n",
                             "Shared suffixes and sequence checks compress "
                             "this boundary only.\n",
                             "No neighboring layer, sentence identity or "
                             "future token is consulted.")));
    ASSIGN_OR_RETURN(auto mlp, RenderPointwise("Lookup", model["mlp"][block]));
    RETURN_IF_ERROR(install(
        absl::StrCat("mlp_", block, ".cc"), mlp, "Map",
        absl::StrCat("GeneratedMlp", block),
        absl::StrCat(
            "Block ", block, ": pointwise MLP residual transition.\n",
            "Symbol renaming may expose a guarded offset. This function and\n",
            "both boundary alphabets remain separate; it is not a layer "
            "bypass.")));
  }
  ASSIGN_OR_RETURN(
      auto head,
      RenderPointwise("Lookup", model["language_modeling_head"], &names));
  RETURN_IF_ERROR(install(
      "language_modeling_head.cc", head, "Map", "GeneratedLanguageModelingHead",
      "Final residual symbol -> named vocabulary token.\n"
      "No unobserved input is assigned a default prediction.",
      true));
  absl::StrAppend(&report, "\nWithin-boundary relabeling\n");
  if (model.contains("stats") &&
      model["stats"].contains("pointwise_relabeling"))
    for (const auto& [key, value] :
         model["stats"]["pointwise_relabeling"].items())
      absl::StrAppend(&report, "  ", key, ": ", ReportValue(value), "\n");
  files["transition_patterns.txt"] = std::move(report);
  if (model.contains("state_relabeling")) {
    std::string mapping =
        "# Pure within-boundary renaming relative to generator input.\n"
        "old_state_id\tnew_state_id\tboundary_index\n";
    for (const auto& row : model["state_relabeling"])
      absl::StrAppend(&mapping, row[0].get<int>(), "\t", row[1].get<int>(),
                      "\t", row[2].get<int>(), "\n");
    files["state_relabeling.tsv"] = std::move(mapping);
  }
  ASSIGN_OR_RETURN(files["generated_transition_test.cc"],
                   RenderTransitionTest(model, qualified_names));
  return absl::OkStatus();
}

}  // namespace pluto::llm::discretized::generator
