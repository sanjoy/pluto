// Exploratory suffix decoder, not a universal decoder of trained knowledge.
// Required side information: initialization, trained weights, architecture,
// compact vocabulary/tokenizer, and the task's known five-token prompt length.
// The frozen sign rule selects a token SET. Default Hamiltonian search assumes
// each selected non-EOS ID occurs once; greedy_history separately permits
// repeated predictions. Neither mode receives multiplicities or missing IDs.
// A separate exploratory report ranks a candidate prompt SET, assuming five
// distinct prompt IDs disjoint from that suffix set; it does not infer order
// and never supplies candidate prompt tokens to inference or path search.
// No corpus, prompt text, gold labels, or known token order are read.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/cuda/executor.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/gpt2_detokenizer.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/one_shot_memorizer/embedding_sign_readout.h"
#include "src/llm/experiments/one_shot_memorizer/greedy_suffix_readout.h"
#include "src/llm/experiments/one_shot_memorizer/position_path.h"
#include "src/llm/experiments/one_shot_memorizer/token_trace.h"
#include "src/llm/gpt2.h"
#include "src/llm/layer.h"
#include "src/util/status_macros.h"

ABSL_FLAG(
    std::string, initial_checkpoint, "",
    "Original step_0; only embedding weights and vocabulary mapping read");
ABSL_FLAG(std::string, checkpoint, "",
          "Trained single-fact checkpoint directory");
ABSL_FLAG(std::string, tokenizer, "", "Base GPT-2 tokenizer directory");
ABSL_FLAG(std::string, output_dir, "",
          "Fresh directory for decoded paths and scores");
ABSL_FLAG(int, layers, 8, "Transformer block count");
ABSL_FLAG(int, model_width, 16, "Residual width");
ABSL_FLAG(int, attention_heads, 1, "Attention head count");
ABSL_FLAG(int, feed_forward_width, 64, "MLP expansion width");
ABSL_FLAG(std::string, search_mode, "hamiltonian",
          "hamiltonian (original exact all-tokens-once search) or "
          "greedy_history (post-hoc unrestricted continuation with coverage)");
ABSL_FLAG(
    std::string, score_mode, "position_aware",
    "position_aware (primary), fixed_position, or mean_positions; the latter "
    "two are post-hoc controls, not additional independent replications");

namespace pluto::llm::one_shot_memorizer {
namespace {
namespace fs = std::filesystem;
constexpr int kKnownTaskPromptCount = 5;

// Escape bytes without letting token text introduce TSV delimiters or invalid
// UTF-8. This is presentation only and never influences scores or ordering.
std::string DisplayBytes(absl::string_view text) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string result;
  for (unsigned char ch : text)
    if (ch == '\\')
      result += "\\\\";
    else if (ch == '\n')
      result += "\\n";
    else if (ch == '\r')
      result += "\\r";
    else if (ch == '\t')
      result += "\\t";
    else if (ch < 32 || ch >= 127) {
      result += "\\x";
      result += kHex[ch >> 4];
      result += kHex[ch & 15];
    } else
      result += static_cast<char>(ch);
  return result;
}

absl::StatusOr<std::string> Decode(
    absl::Span<const int> compact,
    const tokenizer::CompactVocabularyTokenizer& vocabulary,
    const tokenizer::Gpt2Detokenizer& detokenizer) {
  std::vector<int> original;
  original.reserve(compact.size());
  for (int token : compact) {
    ASSIGN_OR_RETURN(int id, vocabulary.OriginalId(token));
    original.push_back(id);
  }
  return detokenizer.Decode(original);
}

absl::StatusOr<std::string> ReadMappingBytes(const fs::path& checkpoint) {
  std::ifstream input(checkpoint / "compact_vocabulary.tsv", std::ios::binary);
  if (!input)
    return absl::NotFoundError("missing checkpoint vocabulary map");
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

absl::StatusOr<std::vector<float>> ReadEmbedding(const fs::path& checkpoint,
                                                 size_t elements) {
  static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
  std::ifstream input(checkpoint / "weight_0.bin",
                      std::ios::binary | std::ios::ate);
  if (!input || input.tellg() != static_cast<std::streamoff>(elements * 4))
    return absl::InvalidArgumentError("missing or wrong-sized embedding");
  std::vector<float> values(elements);
  input.seekg(0);
  input.read(reinterpret_cast<char*>(values.data()), elements * sizeof(float));
  if (!input)
    return absl::DataLossError("short embedding read");
  return values;
}

std::string GreedyRejectionReason(const GreedySuffixCandidate& candidate) {
  if (candidate.accepted())
    return "accepted";
  std::string reason;
  const auto add = [&](absl::string_view part) {
    if (!reason.empty())
      reason += ',';
    reason.append(part);
  };
  if (candidate.hit_token_limit)
    add("token_limit");
  if (!candidate.terminated_with_eos)
    add("missing_EOS");
  if (!candidate.all_tokens_selected)
    add("outside_selected_set");
  if (!candidate.covers_selected_tokens)
    add("incomplete_coverage");
  return reason;
}

std::string JoinTokenIds(absl::Span<const int> tokens) {
  std::string result;
  for (int token : tokens) {
    if (!result.empty())
      result += ',';
    result += std::to_string(token);
  }
  return result;
}

// Write per-start successes AND failures. Selected IDs only seed starts and
// filter completed candidates; no logit mask, forced suffix, or gold length
// enters a forward. Prompt candidates remain a separate, unused diagnostic.
absl::StatusOr<size_t> WriteGreedyHistoryReport(
    cuda::Executor& executor, const Layer& model,
    const tokenizer::CompactVocabularyTokenizer& vocabulary,
    const tokenizer::Gpt2Detokenizer& detokenizer,
    absl::Span<const int> selected, const fs::path& directory,
    std::ostream& paths, std::ostream& manifest) {
  const TokenTraceOptions trace_options{
      .vocabulary_size = vocabulary.vocab_size(),
      .padding_token = vocabulary.eos_token_id()};
  const GreedySuffixLogits evaluate =
      [&](absl::Span<const int> history) -> absl::StatusOr<std::vector<float>> {
    ASSIGN_OR_RETURN(auto trace,
                     TraceNextToken(executor, model, history, trace_options));
    return std::move(trace.logits);
  };
  ASSIGN_OR_RETURN(
      auto result,
      ReadGreedySuffix({.vocabulary_size = vocabulary.vocab_size(),
                        .eos_token = vocabulary.eos_token_id(),
                        .selected_token_ids = selected,
                        .prompt_token_count = kKnownTaskPromptCount},
                       evaluate));
  std::ofstream candidates(directory / "greedy_candidates.tsv");
  std::ofstream steps(directory / "greedy_steps.tsv");
  if (!candidates || !steps)
    return absl::UnknownError("cannot create greedy-history reports");
  candidates << std::setprecision(17);
  steps << std::setprecision(17);
  candidates
      << "start_compact\tlog_score\tterminated_EOS\tall_tokens_selected\t"
         "all_selected_covered\tdistinct_covered\taccepted\thit_token_limit\t"
         "nonEOS_length\treason\tcompact_ids\tdecoded_bytes\n";
  steps << "start_compact\tstep\tabsolute_query_position\tnext_compact\t"
           "log_probability\tappended\n";
  for (const auto& candidate : result.candidates) {
    ASSIGN_OR_RETURN(auto text,
                     Decode(candidate.token_ids, vocabulary, detokenizer));
    candidates << candidate.start_token << '\t' << candidate.log_score << '\t'
               << candidate.terminated_with_eos << '\t'
               << candidate.all_tokens_selected << '\t'
               << candidate.covers_selected_tokens << '\t'
               << candidate.distinct_selected_covered << '\t'
               << candidate.accepted() << '\t' << candidate.hit_token_limit
               << '\t'
               << candidate.token_ids.size() - candidate.terminated_with_eos
               << '\t' << GreedyRejectionReason(candidate) << '\t'
               << JoinTokenIds(candidate.token_ids) << '\t'
               << DisplayBytes(text) << '\n';
    for (size_t index = 0; index < candidate.steps.size(); ++index) {
      const auto& step = candidate.steps[index];
      steps << candidate.start_token << '\t' << index << '\t'
            << kKnownTaskPromptCount + index << '\t' << step.token << '\t'
            << step.log_probability << '\t' << step.appended << '\n';
    }
  }
  for (size_t rank = 0; rank < result.accepted_order.size(); ++rank) {
    const auto& candidate = result.candidates[result.accepted_order[rank]];
    const auto& best = result.candidates[result.accepted_order.front()];
    ASSIGN_OR_RETURN(auto text,
                     Decode(candidate.token_ids, vocabulary, detokenizer));
    paths << rank + 1 << '\t' << candidate.log_score << '\t'
          << best.log_score - candidate.log_score << '\t'
          << JoinTokenIds(candidate.token_ids) << '\t' << DisplayBytes(text)
          << '\n';
    std::cout << "rank=" << rank + 1 << " score=" << candidate.log_score
              << " decoded=" << DisplayBytes(text) << std::endl;
  }
  if (result.accepted_order.empty())
    std::cout
        << "No accepted greedy-history continuation; see greedy_candidates.tsv"
        << std::endl;
  manifest << "greedy_candidate_count\t" << result.candidates.size()
           << "\ngreedy_accepted_count\t" << result.accepted_order.size()
           << "\nmaximum_nonEOS_suffix_tokens\t" << result.max_non_eos_tokens
           << '\n';
  candidates.close();
  steps.close();
  if (!candidates || !steps)
    return absl::DataLossError("failed writing greedy-history reports");
  return result.model_query_count;
}

absl::Status PositionPathProbe() {
  const std::string search_mode = absl::GetFlag(FLAGS_search_mode);
  if (search_mode != "hamiltonian" && search_mode != "greedy_history")
    return absl::InvalidArgumentError(
        "search_mode must be hamiltonian or greedy_history");
  const bool greedy_history = search_mode == "greedy_history";
  const std::string score_mode = absl::GetFlag(FLAGS_score_mode);
  PositionPathScoreMode mode;
  if (score_mode == "position_aware")
    mode = PositionPathScoreMode::kPositionAware;
  else if (score_mode == "fixed_position")
    mode = PositionPathScoreMode::kFixedPosition;
  else if (score_mode == "mean_positions")
    mode = PositionPathScoreMode::kMeanPositions;
  else
    return absl::InvalidArgumentError(
        "score_mode must be position_aware, fixed_position, or mean_positions");
  if (greedy_history && mode != PositionPathScoreMode::kPositionAware)
    return absl::InvalidArgumentError(
        "non-default score_mode is only valid with hamiltonian search");
  const fs::path initial = absl::GetFlag(FLAGS_initial_checkpoint);
  const fs::path checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const fs::path directory = absl::GetFlag(FLAGS_output_dir);
  if (initial.empty() || checkpoint.empty() || directory.empty() ||
      absl::GetFlag(FLAGS_tokenizer).empty())
    return absl::InvalidArgumentError(
        "initial_checkpoint, checkpoint, tokenizer, fresh output_dir required");
  ASSIGN_OR_RETURN(auto initial_map, ReadMappingBytes(initial));
  ASSIGN_OR_RETURN(auto trained_map, ReadMappingBytes(checkpoint));
  if (initial_map != trained_map)
    return absl::InvalidArgumentError("initial/trained vocabulary maps differ");
  ASSIGN_OR_RETURN(
      auto base, tokenizer::Gpt2Tokenizer::Load(absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto detokenizer, tokenizer::Gpt2Detokenizer::Load(
                                         absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto vocabulary,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *base, checkpoint / "compact_vocabulary.tsv"));
  if (base->eos_token_id() != vocabulary->original_eos_token_id() ||
      detokenizer->vocab_size() != base->vocab_size() ||
      detokenizer->eos_token_id() != base->eos_token_id())
    return absl::InvalidArgumentError("checkpoint/tokenizer identity differs");
  const Gpt2Config config{
      .transformer_block_count = absl::GetFlag(FLAGS_layers),
      .model_width = absl::GetFlag(FLAGS_model_width),
      .attention_heads = absl::GetFlag(FLAGS_attention_heads),
      .feed_forward_width = absl::GetFlag(FLAGS_feed_forward_width),
      .vocabulary_size = vocabulary->vocab_size(),
      .pad_vocabulary = false};
  RETURN_IF_ERROR(config.Validate());
  const size_t elements =
      static_cast<size_t>(config.model_width) * config.vocabulary_size;
  ASSIGN_OR_RETURN(auto initial_embedding, ReadEmbedding(initial, elements));
  ASSIGN_OR_RETURN(auto trained_embedding, ReadEmbedding(checkpoint, elements));
  ASSIGN_OR_RETURN(auto decoded, ComputeEmbeddingSignReadout(
                                     initial_embedding, trained_embedding,
                                     config.model_width));
  ASSIGN_OR_RETURN(
      auto prompt_candidates,
      RankEmbeddingResidualCandidates(initial_embedding, trained_embedding,
                                      config.model_width));
  const int eos = vocabulary->eos_token_id();
  const auto& selected = decoded.delta_fp32.selected_ids;
  if (!std::binary_search(selected.begin(), selected.end(), eos))
    return absl::FailedPreconditionError(
        "frozen selected set does not contain EOS");
  std::vector<int> nodes;
  for (int token : selected)
    if (token != eos)
      nodes.push_back(token);
  const int n = nodes.size();
  if (n < 1 || n > kMaxPositionPathNodes ||
      kKnownTaskPromptCount + (greedy_history ? 2 * n : n) > kGpt2ContextLength)
    return absl::FailedPreconditionError(
        "selected non-EOS count must be 1..16");
  const int query_positions =
      mode == PositionPathScoreMode::kFixedPosition ? 1 : n;
  const auto token_text = [&](int token) -> absl::StatusOr<std::string> {
    ASSIGN_OR_RETURN(auto text, Decode(absl::Span<const int>(&token, 1),
                                       *vocabulary, *detokenizer));
    return DisplayBytes(text);
  };
  std::error_code error;
  if (!fs::create_directory(directory, error))
    return absl::InvalidArgumentError("output_dir must be fresh: " +
                                      error.message());
  std::ofstream manifest(directory / "manifest.tsv");
  std::ofstream choices(directory / "selected_tokens.tsv");
  std::ofstream prompts(directory / "prompt_candidates.tsv");
  std::ofstream scores;
  std::ofstream paths(directory / "paths.tsv");
  std::ofstream path_edges;
  std::vector<std::ofstream*> reports{&manifest, &choices, &prompts, &paths};
  if (!greedy_history) {
    scores.open(directory / "edge_scores.tsv");
    path_edges.open(directory / "path_edges.tsv");
    reports.push_back(&scores);
    reports.push_back(&path_edges);
  }
  if (std::any_of(reports.begin(), reports.end(),
                  [](auto* stream) { return !*stream; }))
    return absl::UnknownError("cannot create position path report");
  for (auto* stream : reports)
    *stream << std::setprecision(17);
  manifest
      << "initial_checkpoint\t" << initial.string() << "\ncheckpoint\t"
      << checkpoint.string() << "\nexploratory\ttrue\n"
      << "selection_rule\tdelta_fp32_dot_mean_lt_0\n"
      << "prompt_candidate_rule\trank_squared_norm_delta_minus_mean_excluding_"
         "negative_sign_rows\n"
      << "prompt_candidate_exploratory_posthoc\ttrue\n"
      << "prompt_candidate_assumption\tfive_distinct_prompt_IDs_disjoint_from_"
         "sign_selected_suffix_EOS_IDs\n"
      << "prompt_candidate_assumption_verified\tfalse\n"
      << "prompt_candidate_order_inferred\tfalse\n"
      << "prompt_candidates_used_for_inference\tfalse\n"
      << "prompt_candidate_report_limit\t20\n"
      << "prompt_candidate_set_size\t"
      << std::min(prompt_candidates.size(),
                  static_cast<size_t>(kKnownTaskPromptCount))
      << '\n'
      << "corpus_read\tfalse\ngold_labels_read\tfalse\n"
      << "known_task_prompt_count\t" << kKnownTaskPromptCount << '\n'
      << "selected_non_eos_count\t" << n << '\n';
  if (greedy_history) {
    manifest
        << "search_mode\tgreedy_history\nposthoc_search_control\ttrue\n"
           "query_policy\tactual_history_after_five_EOS_and_each_selected_"
           "start\n"
           "normalization\tfull_vocabulary_softmax_no_selected_mask\n"
           "score_definition\tsum_appended_successor_and_EOS_logprob\n"
           "path_policy\tEOS_terminated_all_selected_covered_no_outside_tokens_"
           "repeats_allowed\n"
           "start_policy\tuniform_score_zero_seed_not_scored\n"
           "multiplicity_inference\tgenerated_not_supplied\n"
           "prompt_order_supplied\tfalse\n"
           "generation_cap_rule\ttwice_selected_nonEOS_count_not_gold_length\n";
  } else {
    manifest
        << "score_mode\t" << score_mode << '\n'
        << "posthoc_score_control\t"
        << (mode != PositionPathScoreMode::kPositionAware) << '\n'
        << "queried_absolute_positions\t5.." << 4 + query_positions << '\n'
        << "query_policy\tEOS_prefix_then_selected_token_at_position_5_plus_j\n"
        << "normalization\tfull_vocabulary_softmax\n"
        << "score_definition\t"
        << (mode == PositionPathScoreMode::kMeanPositions
                ? "mean_log_probabilities_no_renormalization"
                : "log_probability")
        << '\n'
        << "path_policy\teach_selected_non_eos_ID_once_then_EOS\n"
        << "start_policy\tuniform_score_zero_no_known_first_token\n"
        << "multiplicity_inference\tunsupported\n";
  }
  manifest << "corpus_suffix_length_used\tfalse\n"
           << "side_information\tinitial_weights_trained_weights_architecture_"
              "vocabulary_known_prompt_count\n"
           << "tokenizer\t" << DisplayBytes(absl::GetFlag(FLAGS_tokenizer))
           << '\n'
           << "model_width\t" << config.model_width << '\n'
           << "layers\t" << config.transformer_block_count << '\n'
           << "attention_heads\t" << config.attention_heads << '\n'
           << "feed_forward_width\t" << config.feed_forward_width << '\n';
  choices << "compact_id\toriginal_id\tsign_score\tis_eos\ttoken_bytes\n";
  for (int token : selected) {
    ASSIGN_OR_RETURN(auto text, token_text(token));
    choices << token << '\t' << vocabulary->original_token_ids()[token] << '\t'
            << decoded.delta_fp32.scores[token] << '\t' << (token == eos)
            << '\t' << text << '\n';
  }
  // Rank is evidence strength, NOT a recovered position in the prompt. The
  // five-token task supplies the set size, never the IDs or their ordering.
  prompts << "residual_rank\tcompact_id\toriginal_id\tresidual_squared_norm\t"
             "sign_score\tin_top5_candidate_set\ttoken_bytes\n";
  for (size_t rank = 0; rank < std::min<size_t>(20, prompt_candidates.size());
       ++rank) {
    const auto& candidate = prompt_candidates[rank];
    const int token = candidate.row_id;
    ASSIGN_OR_RETURN(auto text, token_text(token));
    prompts << rank + 1 << '\t' << token << '\t'
            << vocabulary->original_token_ids()[token] << '\t'
            << candidate.residual_squared_norm << '\t'
            << decoded.delta_fp32.scores[token] << '\t'
            << (rank < static_cast<size_t>(kKnownTaskPromptCount)) << '\t'
            << text << '\n';
  }
  if (!greedy_history) {
    scores
        << "suffix_index\tabsolute_query_position\tsource_compact\t"
           "destination_compact\tis_eos\tlog_probability\tquery_top1_compact\n";
    path_edges << "path_rank\tsuffix_index\tsuffix_absolute_position\t"
                  "source_compact\tsource_bytes\tdestination_compact\t"
                  "destination_bytes\tedge_log_score\texp_edge_log_score\n";
  }
  paths << "rank\tlog_score\tgap_from_best\tcompact_path\tdecoded_bytes\n";
  choices.flush();
  prompts.flush();
  manifest.flush();
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto model, CreateGpt2(*executor, DataType::BF16, 0, config));
  RETURN_IF_ERROR(
      ReadFromDirectory(*executor, *model, checkpoint.string(), false));
  if (greedy_history) {
    ASSIGN_OR_RETURN(
        size_t queries,
        WriteGreedyHistoryReport(*executor, *model, *vocabulary, *detokenizer,
                                 selected, directory, paths, manifest));
    manifest << "query_count\t" << queries << "\ncomplete\ttrue\n";
    for (auto* stream : reports) {
      stream->close();
      if (!*stream)
        return absl::DataLossError("failed writing greedy-history report");
    }
    return absl::OkStatus();
  }
  const TokenTraceOptions options{.vocabulary_size = config.vocabulary_size,
                                  .padding_token = eos};
  std::vector<double> raw_edges(static_cast<size_t>(query_positions) * n *
                                (n + 1));
  for (int position = 0; position < query_positions; ++position) {
    for (int source = 0; source < n; ++source) {
      // Constant EOS history and a candidate from the selected SET only. We
      // never supply the previously inferred path, original prompt, or labels.
      std::vector<int> input(kKnownTaskPromptCount + position + 1, eos);
      input.back() = nodes[source];
      ASSIGN_OR_RETURN(auto result,
                       TraceNextToken(*executor, *model, input, options));
      const double maximum = result.logits[result.predicted_token];
      double denominator = 0;
      for (float logit : result.logits)
        denominator += std::exp(static_cast<double>(logit) - maximum);
      const double log_denominator = std::log(denominator);
      for (int destination = 0; destination <= n; ++destination) {
        const int next = destination == n ? eos : nodes[destination];
        const double log_probability =
            (static_cast<double>(result.logits[next]) - maximum) -
            log_denominator;
        raw_edges[(position * n + source) * (n + 1) + destination] =
            log_probability;
        scores << position << '\t' << kKnownTaskPromptCount + position << '\t'
               << nodes[source] << '\t' << next << '\t' << (next == eos) << '\t'
               << log_probability << '\t' << result.predicted_token << '\n';
      }
    }
    scores.flush();
    std::cout << "scored suffix_index=" << position << " of " << query_positions
              << std::endl;
  }
  // These optional controls were introduced after observing the primary
  // position-aware results. They change only the edge scores, never the
  // selected set, terminal, uniform starting score, or all-tokens-once rule.
  ASSIGN_OR_RETURN(auto edges, PreparePositionPathScores(n, raw_edges, mode));
  ASSIGN_OR_RETURN(auto solved, SolvePositionPaths(n, edges));
  for (size_t rank = 0; rank < solved.size(); ++rank) {
    const auto& result = solved[rank];
    std::vector<int> tokens;
    for (int index : result.nodes)
      tokens.push_back(nodes[index]);
    tokens.push_back(eos);
    std::string path;
    for (int token : tokens) {
      if (!path.empty())
        path += ',';
      path += std::to_string(token);
    }
    ASSIGN_OR_RETURN(auto text, Decode(tokens, *vocabulary, *detokenizer));
    paths << rank + 1 << '\t' << result.score << '\t'
          << solved.front().score - result.score << '\t' << path << '\t'
          << DisplayBytes(text) << '\n';
    std::cout << "rank=" << rank + 1 << " score=" << result.score
              << " decoded=" << DisplayBytes(text) << std::endl;
    for (int position = 0; position < n; ++position) {
      const int source = result.nodes[position];
      const int destination =
          position == n - 1 ? n : result.nodes[position + 1];
      const double score =
          edges[(position * n + source) * (n + 1) + destination];
      ASSIGN_OR_RETURN(auto source_text, token_text(tokens[position]));
      ASSIGN_OR_RETURN(auto next_text, token_text(tokens[position + 1]));
      path_edges << rank + 1 << '\t' << position << '\t'
                 << kKnownTaskPromptCount + position << '\t' << tokens[position]
                 << '\t' << source_text << '\t' << tokens[position + 1] << '\t'
                 << next_text << '\t' << score << '\t' << std::exp(score)
                 << '\n';
    }
  }
  manifest << "query_count\t" << n * query_positions << "\ncomplete\ttrue\n";
  for (auto* stream : reports) {
    stream->close();
    if (!*stream)
      return absl::DataLossError("failed writing position path report");
  }
  return absl::OkStatus();
}
}  // namespace
}  // namespace pluto::llm::one_shot_memorizer

int main(int argc, char** argv) {
  if (absl::ParseCommandLine(argc, argv).size() != 1)
    return 2;
  const auto status = pluto::llm::one_shot_memorizer::PositionPathProbe();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
