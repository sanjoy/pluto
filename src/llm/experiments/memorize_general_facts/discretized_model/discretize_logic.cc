#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_logic.h"

#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_attention_logic.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_pointwise.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_transition_tests.h"
#include "src/util/status_macros.h"

namespace pluto::llm::discretized::generator {
namespace {

absl::string_view RepresentationName(TransitionRepresentation representation) {
  switch (representation) {
    case TransitionRepresentation::kEmpty:
      return "empty";
    case TransitionRepresentation::kGuardedAffine:
      return "guarded_affine";
    case TransitionRepresentation::kSparseAffineSupportMask:
      return "sparse_affine_support_mask";
    case TransitionRepresentation::kAffineRangesAndSwitch:
      return "affine_ranges_and_switch";
    case TransitionRepresentation::kGuardedOutputArray:
      return "guarded_output_array";
    case TransitionRepresentation::kExactTokenPositionSwitch:
      return "exact_token_position_switch";
    case TransitionRepresentation::kPackedSupportPatternsAndExceptions:
      return "packed_support_patterns_and_exceptions";
    case TransitionRepresentation::kSharedSuffixControlFlow:
      return "shared_suffix_control_flow";
  }
  return "unknown";
}

template <class T>
void AppendOptional(std::string& report, absl::string_view name,
                    const std::optional<T>& value) {
  if (!value)
    return;
  if constexpr (std::is_same_v<T, bool>)
    absl::StrAppend(&report, "  ", name, ": ", *value ? "true" : "false", "\n");
  else
    absl::StrAppend(&report, "  ", name, ": ", *value, "\n");
}

void AppendStatistics(std::string& report, const TransitionStatistics& stats) {
  absl::StrAppend(
      &report, "  representation: ", RepresentationName(stats.representation),
      "\n  rows: ", stats.rows, "\n  source_bytes: ", stats.source_bytes, "\n");
#define APPEND_MEASUREMENT(field) AppendOptional(report, #field, stats.field)
  if (stats.strategy)
    absl::StrAppend(&report, "  strategy: ",
                    *stats.strategy == AttentionStrategy::kHybrid
                        ? "hybrid"
                        : "control_flow",
                    "\n");
  APPEND_MEASUREMENT(table_bytes);
  APPEND_MEASUREMENT(affine_ranges);
  APPEND_MEASUREMENT(switch_cases);
  APPEND_MEASUREMENT(supported_span);
  APPEND_MEASUREMENT(named_anchor);
  APPEND_MEASUREMENT(named_outputs);
  APPEND_MEASUREMENT(tokens);
  APPEND_MEASUREMENT(patterns);
  APPEND_MEASUREMENT(position_bits);
  APPEND_MEASUREMENT(token_only_defaults);
  APPEND_MEASUREMENT(position_exceptions);
  APPEND_MEASUREMENT(named_exception_tokens);
  APPEND_MEASUREMENT(flat_key_scalars);
  APPEND_MEASUREMENT(flat_scalars);
  APPEND_MEASUREMENT(trie_nodes);
  APPEND_MEASUREMENT(nodes);
  APPEND_MEASUREMENT(edges);
  APPEND_MEASUREMENT(unary_nodes);
  APPEND_MEASUREMENT(branch_nodes);
  APPEND_MEASUREMENT(branch_edges);
  APPEND_MEASUREMENT(control_blocks);
  APPEND_MEASUREMENT(helpers);
  APPEND_MEASUREMENT(helper_node_limit);
  APPEND_MEASUREMENT(entry_cases);
  APPEND_MEASUREMENT(scalar_estimate);
  APPEND_MEASUREMENT(literal_sequence_patterns);
  APPEND_MEASUREMENT(literal_sequence_steps);
  APPEND_MEASUREMENT(literal_sequence_calls);
  APPEND_MEASUREMENT(literal_sequence_word_bits);
#undef APPEND_MEASUREMENT
}

}  // namespace

absl::Status RenderCompact(const SymbolicModel& model,
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
    AppendStatistics(report, rendered.stats);
    absl::StrAppend(
        &report, "  unformatted_source_bytes_before: ", before,
        "\n  unformatted_source_bytes_after: ", found->second.size(), "\n");
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
  absl::StrAppend(&report, "\nWithin-boundary relabeling\n");
  if (model.stats.pointwise_relabeling) {
    const auto& stats = *model.stats.pointwise_relabeling;
    absl::StrAppend(
        &report,
        "  eligible_layers: ", absl::StrJoin(stats.eligible_layers, ", "),
        "\n  skipped_layers: ", absl::StrJoin(stats.skipped_layers, ", "),
        "\n  changed_states: ", stats.changed_states,
        "\n  layer_boundaries_preserved: ",
        stats.layer_boundaries_preserved ? "true" : "false",
        "\n  vocabulary_aligned_final_boundaries: ",
        stats.vocabulary_aligned_final_boundaries ? "true" : "false", "\n");
    AppendOptional(report, "last_attention_base", stats.last_attention_base);
    AppendOptional(report, "final_state_base", stats.final_state_base);
    AppendOptional(report, "reserved_range_size", stats.reserved_range_size);
  }
  files["transition_patterns.txt"] = std::move(report);
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
