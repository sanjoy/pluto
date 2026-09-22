#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_logic.h"

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_attention_logic.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_pointwise.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_transition_tests.h"
#include "src/util/status_macros.h"

namespace pluto::llm::discretized::generator {

absl::Status RenderCompact(const SymbolicModel& model,
                           const std::vector<std::string>& token_names,
                           FileMap& files) {
  TokenNames names;
  std::vector<std::string> qualified_names;
  for (size_t index = 0; index < token_names.size(); ++index) {
    qualified_names.push_back(absl::StrCat("vocab::", token_names[index]));
    names[index] = qualified_names.back();
  }
  auto install = [&](const std::string& filename, absl::string_view source,
                     absl::string_view interface, absl::string_view factory,
                     absl::string_view description,
                     bool vocabulary = false) -> absl::Status {
    auto found = files.find(filename);
    if (found == files.end())
      return absl::InternalError(
          absl::StrCat("missing plain generated file: ", filename));
    found->second = Source(TransitionObject(source, interface, factory),
                           "\"tables.h\"", description, vocabulary);
    return absl::OkStatus();
  };
  ASSIGN_OR_RETURN(auto entry, RenderEntry("Lookup", model.entry, names));
  RETURN_IF_ERROR(install(
      "entry.cc", entry, "PositionEmbedding", "GeneratedPositionEmbedding",
      "Entry: compact token and absolute position -> residual symbol.\n"
      "A default symbol plus exceptional positions describes each token;\n"
      "support masks reject every token/position absent from the source.",
      true));
  for (size_t block = 0; block < model.transformers.size(); ++block) {
    ASSIGN_OR_RETURN(
        auto attention,
        RenderAttention("Lookup", model.transformers[block].attention));
    RETURN_IF_ERROR(
        install(absl::StrCat("attention_", block, ".cc"), attention,
                "CausalAttention", absl::StrCat("GeneratedAttention", block),
                absl::StrCat("Block ", block,
                             ": exact causal-history decision program.\n",
                             "Shared suffixes and sequence checks compress "
                             "this boundary only.\n",
                             "No neighboring layer, sentence identity or "
                             "future token is consulted.")));
    ASSIGN_OR_RETURN(auto mlp,
                     RenderPointwise("Lookup", model.transformers[block].mlp));
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
      RenderPointwise("Lookup", model.language_modeling_head, &names));
  RETURN_IF_ERROR(install(
      "language_modeling_head.cc", head, "Map", "GeneratedLanguageModelingHead",
      "Final residual symbol -> named vocabulary token.\n"
      "No unobserved input is assigned a default prediction.",
      true));
  if (!model.state_relabeling.empty()) {
    std::string mapping =
        "# Pure within-boundary renaming relative to generator input.\n"
        "old_state_id\tnew_state_id\tboundary_index\n";
    for (const auto& row : model.state_relabeling)
      absl::StrAppend(&mapping, row.old_id, "\t", row.new_id, "\t",
                      row.boundary, "\n");
    files["state_relabeling.tsv"] = std::move(mapping);
  }
  ASSIGN_OR_RETURN(files["generated_transition_test.cc"],
                   RenderTransitionTest(model, qualified_names));
  return absl::OkStatus();
}

}  // namespace pluto::llm::discretized::generator
