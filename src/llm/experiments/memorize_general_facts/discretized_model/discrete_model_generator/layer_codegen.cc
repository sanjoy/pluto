#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/layer_codegen.h"

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/discretize_attention.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/discretize_map.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/discretize_position_embedding.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/discretize_transition_tests.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/utils.h"
#include "src/util/status_macros.h"

namespace pluto::llm::discretized::generator {

absl::Status RenderLayerSources(const CapturedModel& model,
                                const std::vector<std::string>& token_names,
                                bool compact, FileMap& files) {
  TokenNames names;
  std::vector<std::string> qualified_names;
  for (size_t index = 0; index < token_names.size(); ++index) {
    qualified_names.push_back(absl::StrCat("vocab::", token_names[index]));
    names[index] = qualified_names.back();
  }
  auto install = [&](const std::string& filename, absl::string_view source,
                     absl::string_view interface, absl::string_view factory,
                     absl::string_view description, bool vocabulary = false) {
    files[filename] = Source(TransitionObject(source, interface, factory),
                             "\"tables.h\"", description, vocabulary);
  };
  ASSIGN_OR_RETURN(auto entry,
                   RenderPositionEmbedding(model.position_embedding, "Lookup",
                                           names, compact));
  install(
      "entry.cc", entry.source, "PositionEmbedding",
      "GeneratedPositionEmbedding",
      compact
          ? "Entry: compact token and absolute position -> residual symbol.\n"
            "A default symbol plus exceptional positions describes each "
            "token;\n"
            "support masks reject every token/position absent from the source."
          : "Entry: compact token and absolute position -> residual symbol.\n"
            "Exact sorted lookup rejects every unobserved token/position.",
      true);
  for (size_t block = 0; block < model.transformers.size(); ++block) {
    ASSIGN_OR_RETURN(
        auto attention,
        RenderAttention(model.transformers[block].attention, "Lookup", compact));
    install(absl::StrCat("attention_", block, ".cc"), attention.source,
            "CausalAttention", absl::StrCat("GeneratedAttention", block),
            absl::StrCat(
                "Block ", block, ": exact causal-history decision program.\n",
                compact ? "Shared suffixes and sequence checks compress "
                          "this boundary only.\n"
                        : "Exact sorted lookup rejects unknown "
                          "causal histories.\n",
                "No neighboring layer, sentence identity or "
                "future token is consulted."));
    ASSIGN_OR_RETURN(auto mlp, RenderMap(model.transformers[block].mlp,
                                         "Lookup", nullptr, compact));
    install(absl::StrCat("mlp_", block, ".cc"), mlp.source, "Map",
            absl::StrCat("GeneratedMlp", block),
            absl::StrCat(
                "Block ", block, ": pointwise MLP residual transition.\n",
                compact ? "Symbol renaming may expose a guarded offset. This "
                          "function and\n"
                        : "Exact sorted lookup implements this function. This "
                          "function and\n",
                "both boundary alphabets remain separate; it is not a layer "
                "bypass."));
  }
  ASSIGN_OR_RETURN(auto head, RenderMap(model.language_modeling_head, "Lookup",
                                        &names, compact));
  install("language_modeling_head.cc", head.source, "Map",
          "GeneratedLanguageModelingHead",
          "Final residual symbol -> named vocabulary token.\n"
          "No unobserved input is assigned a default prediction.",
          true);
  if (!compact)
    return absl::OkStatus();
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
