// Prefix-only activation tracing and isolated causal interventions. Corpus
// suffixes supply reporting labels, never inputs to a traced model forward.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_detokenizer.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/one_shot_memorizer/feature_capture.h"
#include "src/llm/experiments/one_shot_memorizer/sentence_ablation.h"
#include "src/llm/experiments/one_shot_memorizer/token_trace.h"
#include "src/llm/experiments/one_shot_memorizer/trace_readout_attribution.h"
#include "src/llm/gpt2.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "", "Exact memorized checkpoint directory");
ABSL_FLAG(std::string, tokenizer, "", "Base GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt",
          "Full corpus");
ABSL_FLAG(std::string, output_dir, "",
          "Fresh directory for prefix trace reports");
ABSL_FLAG(
    std::vector<std::string>, lines,
    (std::vector<std::string>{"80", "406", "411", "1", "258", "631"}),
    "Distinct one-based corpus lines to trace from their first five tokens");
ABSL_FLAG(int, expected_sentences, 1024, "Expected corpus line count");
ABSL_FLAG(int, layers, 8, "Transformer block count");
ABSL_FLAG(int, model_width, 16, "Residual width");
ABSL_FLAG(int, attention_heads, 1, "Attention head count");
ABSL_FLAG(int, feed_forward_width, 64, "MLP expansion width");
ABSL_FLAG(
    bool, branch_ablations, true,
    "Independently zero each attention/MLP branch at all five prefix rows");
ABSL_FLAG(
    bool, capital_donor_patches, true,
    "Transplant residual and Q/K/V rows between France/Greece/Peru prompts");
ABSL_FLAG(bool, readout_lens, true,
          "Read each residual through the original final LayerNorm and head");
ABSL_FLAG(std::vector<std::string>, mlp_neuron_ablation_blocks,
          (std::vector<std::string>{}),
          "Distinct zero-based MLP blocks: individually zero each GELU channel "
          "at query row 4 only; empty disables these interventions");
ABSL_FLAG(std::vector<std::string>, mlp_neuron_ablation_channels,
          (std::vector<std::string>{}),
          "Distinct zero-based GELU channels to zero independently in the "
          "selected MLP blocks; empty selects every feed-forward channel");

namespace pluto::llm::one_shot_memorizer {
namespace {

namespace fs = std::filesystem;
constexpr size_t kPrefixTokens = 5;
// Match the actual float constant passed to LayerNormLayer by gpt2.cc,
// rather than silently replacing it with the slightly different double 1e-5.
constexpr double kReadoutEpsilon = static_cast<double>(1e-5f);

struct ReadoutParameters {
  int width = 0;
  std::vector<float> effective_embeddings;  // Logical token rows, rounded BF16.
  std::vector<float> gamma;  // Original final LayerNorm FP32 master values.
  std::vector<float> beta;
  std::vector<float> final_mlp_weights;  // Effective BF16 [expansion, width].
  std::vector<float> final_mlp_bias;     // Original FP32 contraction bias.
};

struct ReadoutAccounting {
  int target = 0;
  int rival = 0;  // Strongest other token in the actual baseline final logits.
  std::vector<TokenTraceSite>
      residual_sites;  // Initial state, then increments.
  TraceReadoutAttribution attribution;
  double gpu_logit_margin = 0;  // Difference of stored FP32 logits, in double.
  double gpu_head_residual =
      0;                    // GPU logit margin minus actual-normalized dot.
  int final_mlp_block = 0;  // Zero-based block owning the accounted projection.
  TokenTraceSite final_gelu_site;
  TokenTraceSite final_projection_site;
  std::vector<float> final_gelu_values;  // Captured BF16 query-row operands.
  DenseProjectionReadoutAttribution final_mlp;
  // Observed residual increment minus captured contraction output, along the
  // same direction. Keeps residual-add rounding separate from projection math.
  double final_mlp_residual_add_residual = 0;
};

struct Sentence {
  // One-based source line, retained independently of selection.
  size_t line = 0;
  std::string text;
  // Unpadded corpus encoding; only the first five run.
  std::vector<int> tokens;
};

struct ScoredPrediction {
  int predicted = 0;
  int target = 0;
  size_t target_rank = 0;  // One-based rank, with lower ID breaking exact ties.
  double target_probability = 0;
  double target_margin = 0;  // Target logit minus the largest other logit.
  std::vector<int> top_ids;
  std::vector<double> top_probabilities;
};

struct Baseline {
  const Sentence* sentence = nullptr;
  std::string prefix_text;
  TokenTraceResult trace;
  ReadoutAccounting readout;
};

struct Condition {
  size_t line = 0;
  std::string name;
  size_t donor_line = 0;  // Zero means no donor.
  std::string site;
  std::string source_site;  // Nonempty for donor copies, including the lens.
  std::string rows;
  size_t first_row = 0;  // Source and destination use the same absolute rows.
  size_t row_count = 0;  // Zero only when there is no intervention.
  size_t first_channel = 0;
  size_t channel_count =
      0;            // Actual count, never the patch API's zero sentinel.
  int block = -1;   // Set explicitly for optional single-neuron interventions.
  int neuron = -1;  // GELU channel index; -1 for other condition kinds.
  bool identity_control = false;  // This condition must preserve exact logits.
  ScoredPrediction prediction;
  int donor_target = -1;
  double donor_target_probability = 0;
  double target_probability_change = 0;
};

std::string EscapeHtml(absl::string_view text) {
  std::string result;
  for (char ch : text)
    switch (ch) {
      case '&':
        result += "&amp;";
        break;
      case '<':
        result += "&lt;";
        break;
      case '>':
        result += "&gt;";
        break;
      case '"':
        result += "&quot;";
        break;
      case '\'':
        result += "&#39;";
        break;
      default:
        result += ch;
    }
  return result;
}

// Token pieces may contain tabs, newlines, or partial UTF-8 byte sequences.
// Preserve valid text as far as possible, but escape all non-ASCII bytes so
// both TSV and self-contained HTML remain unambiguous and valid UTF-8.
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

const char* DataTypeName(DataType type) {
  switch (type) {
    case DataType::FP16:
      return "FP16";
    case DataType::BF16:
      return "BF16";
    case DataType::FP8:
      return "FP8";
    case DataType::FP32:
      return "FP32";
    case DataType::INT32:
      return "INT32";
  }
  return "unknown";
}

std::string SiteName(const TokenTraceSite& site) {
  std::string result;
  for (const auto& scope : site.scope)
    absl::StrAppend(&result, scope, "/");
  return absl::StrCat(result, site.layer_name, "#", site.occurrence, ":output_",
                      site.output_index);
}

bool SameSite(const TokenTraceSite& left, const TokenTraceSite& right) {
  return left.scope == right.scope && left.layer_name == right.layer_name &&
         left.occurrence == right.occurrence &&
         left.output_index == right.output_index;
}

absl::StatusOr<const TokenTraceActivation*> FindSite(
    const TokenTraceResult& trace, const TokenTraceSite& site) {
  const TokenTraceActivation* found = nullptr;
  for (const auto& activation : trace.activations)
    if (SameSite(activation.site, site)) {
      if (found != nullptr)
        return absl::InternalError("duplicate captured activation identity");
      found = &activation;
    }
  if (found == nullptr)
    return absl::NotFoundError(
        absl::StrCat("missing trace site: ", SiteName(site)));
  return found;
}

absl::StatusOr<ReadoutParameters> CopyReadoutParameters(
    cuda::Executor& executor, const Layer& model, const Gpt2Config& config) {
  ASSIGN_OR_RETURN(
      auto layout,
      BuildGpt2ParameterLayout(
          {.vocabulary_size = config.vocabulary_size,
           .model_width = config.model_width,
           .feed_forward_width = config.feed_forward_width,
           .context_length = kGpt2ContextLength,
           .transformer_block_count = config.transformer_block_count}));
  const auto weights = model.weights();
  // ComposedLayer does not expose children. Its flattened recipe inventory
  // ends in final gamma, final beta, and the head's alias of token embeddings.
  // Check the entire inventory, including that alias, before using indices.
  if (layout.size() < 4 || weights.size() != layout.size() + 1 ||
      layout[layout.size() - 2].name != "final_layer_norm.gamma" ||
      layout.back().name != "final_layer_norm.beta" ||
      weights.back().data() != weights.front().data() ||
      weights.back().size_bytes() != weights.front().size_bytes() ||
      &weights.back().executor() != &executor)
    return absl::FailedPreconditionError(
        "readout accounting requires the tied GPT-2 parameter inventory");
  const std::string final_mlp_prefix =
      absl::StrCat("transformer_block_", config.transformer_block_count - 1,
                   ".mlp.contraction.");
  if (config.transformer_block_count <= 0 ||
      layout[layout.size() - 4].name != final_mlp_prefix + "weight" ||
      layout[layout.size() - 3].name != final_mlp_prefix + "bias" ||
      layout[layout.size() - 4].element_count !=
          static_cast<size_t>(config.feed_forward_width) * config.model_width ||
      layout[layout.size() - 3].element_count !=
          static_cast<size_t>(config.model_width))
    return absl::FailedPreconditionError(
        "readout accounting requires the final GPT-2 MLP contraction");
  absl::flat_hash_set<const void*> seen;
  for (size_t index = 0; index < layout.size(); ++index)
    if (weights[index].size_bytes() !=
            layout[index].element_count * sizeof(float) ||
        &weights[index].executor() != &executor ||
        !seen.insert(weights[index].data()).second)
      return absl::FailedPreconditionError(absl::StrCat(
          "readout accounting parameter inventory mismatch at ", index));
  ASSIGN_OR_RETURN(auto gamma, cuda::PageLockedHostArray<float>::Allocate(
                                   executor, config.model_width));
  ASSIGN_OR_RETURN(auto beta, cuda::PageLockedHostArray<float>::Allocate(
                                  executor, config.model_width));
  ASSIGN_OR_RETURN(
      auto final_mlp_bias,
      cuda::PageLockedHostArray<float>::Allocate(executor, config.model_width));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(gamma.data(), weights[layout.size() - 2].data(),
                      gamma.size_bytes(), cudaMemcpyDeviceToHost,
                      executor.stream()),
      "copy original final LayerNorm gamma"));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(beta.data(), weights[layout.size() - 1].data(),
                      beta.size_bytes(), cudaMemcpyDeviceToHost,
                      executor.stream()),
      "copy original final LayerNorm beta"));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(final_mlp_bias.data(), weights[layout.size() - 3].data(),
                      final_mlp_bias.size_bytes(), cudaMemcpyDeviceToHost,
                      executor.stream()),
      "copy original final MLP contraction bias"));
  // This helper uses pinned D2H, rounds the actual master embeddings to BF16,
  // and synchronizes the same stream, also completing the three copies above.
  ASSIGN_OR_RETURN(
      auto embeddings,
      CopyEffectiveBf16Readout(executor, weights.front(), config.model_width,
                               config.vocabulary_size));
  // The same row-major conversion also captures the effective dense operands:
  // one contraction-weight row per GELU neuron, not one row per vocabulary ID.
  ASSIGN_OR_RETURN(
      auto final_mlp_weights,
      CopyEffectiveBf16Readout(executor, weights[layout.size() - 4],
                               config.model_width, config.feed_forward_width));
  for (int channel = 0; channel < config.model_width; ++channel)
    if (!std::isfinite(gamma[channel]) || !std::isfinite(beta[channel]) ||
        !std::isfinite(final_mlp_bias[channel]))
      return absl::InvalidArgumentError(
          "nonfinite final normalization or MLP bias parameter");
  return ReadoutParameters{
      .width = config.model_width,
      .effective_embeddings = std::move(embeddings),
      .gamma = std::vector<float>(gamma.begin(), gamma.end()),
      .beta = std::vector<float>(beta.begin(), beta.end()),
      .final_mlp_weights = std::move(final_mlp_weights),
      .final_mlp_bias =
          std::vector<float>(final_mlp_bias.begin(), final_mlp_bias.end())};
}

absl::StatusOr<absl::Span<const float>> QueryRow(
    const TokenTraceActivation& activation, int width) {
  constexpr size_t kQuery = kPrefixTokens - 1;
  if (width <= 0 || activation.data_type != DataType::BF16 ||
      activation.channels != static_cast<size_t>(width) ||
      activation.first_row > kQuery ||
      kQuery - activation.first_row >= activation.row_count ||
      activation.values.size() / activation.channels != activation.row_count ||
      activation.values.size() % activation.channels != 0)
    return absl::InvalidArgumentError(
        "readout accounting requires a captured BF16 query row");
  return absl::MakeConstSpan(activation.values)
      .subspan((kQuery - activation.first_row) * activation.channels,
               activation.channels);
}

absl::StatusOr<ReadoutAccounting> AccountReadout(
    const TokenTraceResult& trace, const Gpt2Config& config,
    const ReadoutParameters& parameters, int target) {
  if (target < 0 || target >= config.vocabulary_size ||
      config.vocabulary_size < 2 ||
      trace.logits.size() != static_cast<size_t>(config.vocabulary_size))
    return absl::InvalidArgumentError(
        "invalid baseline logits for readout accounting");
  int rival = target == 0 ? 1 : 0;
  for (int token = 0; token < config.vocabulary_size; ++token) {
    if (!std::isfinite(trace.logits[token]))
      return absl::InvalidArgumentError(
          "nonfinite baseline logit in readout accounting");
    if (token != target &&
        (trace.logits[token] > trace.logits[rival] ||
         (trace.logits[token] == trace.logits[rival] && token < rival)))
      rival = token;
  }
  ReadoutAccounting result{.target = target, .rival = rival};
  result.residual_sites.push_back({.scope = {"gpt2"},
                                   .layer_name = "PositionEmbeddingLayer",
                                   .occurrence = 0});
  for (int block = 0; block < config.transformer_block_count; ++block)
    for (int occurrence = 0; occurrence < 2; ++occurrence)
      result.residual_sites.push_back(
          {.scope = {"gpt2", absl::StrCat("transformer_block_", block)},
           .layer_name = "ResidualLayer",
           .occurrence = occurrence});
  std::vector<absl::Span<const float>> rows;
  for (const auto& site : result.residual_sites) {
    ASSIGN_OR_RETURN(const auto* activation, FindSite(trace, site));
    ASSIGN_OR_RETURN(auto row, QueryRow(*activation, config.model_width));
    rows.push_back(row);
  }
  ASSIGN_OR_RETURN(const auto* final_norm,
                   FindSite(trace, {.scope = {"gpt2"},
                                    .layer_name = "LayerNormLayer",
                                    .occurrence = 0}));
  ASSIGN_OR_RETURN(auto actual_normalized,
                   QueryRow(*final_norm, config.model_width));
  const auto embeddings = absl::MakeConstSpan(parameters.effective_embeddings);
  ASSIGN_OR_RETURN(
      result.attribution,
      ComputeTraceReadoutAttribution(
          rows, parameters.gamma, parameters.beta,
          embeddings.subspan(static_cast<size_t>(target) * parameters.width,
                             parameters.width),
          embeddings.subspan(static_cast<size_t>(rival) * parameters.width,
                             parameters.width),
          actual_normalized, kReadoutEpsilon));
  // Subtract saved FP32 GPU logits in double. The residual isolates the head's
  // finite-precision dot products from the already-rounded normalized input;
  // there is no extra simulated FP32 subtraction here.
  result.gpu_logit_margin =
      static_cast<double>(trace.logits[target]) - trace.logits[rival];
  result.gpu_head_residual =
      result.gpu_logit_margin - result.attribution.actual_normalized_margin;
  result.final_mlp_block = config.transformer_block_count - 1;
  const std::vector<std::string> final_mlp_scope{
      "gpt2", absl::StrCat("transformer_block_", result.final_mlp_block),
      "ResidualLayer", "mlp"};
  result.final_gelu_site = {
      .scope = final_mlp_scope, .layer_name = "GeluLayer", .occurrence = 0};
  result.final_projection_site = {.scope = final_mlp_scope,
                                  .layer_name = "FullyConnectedLayer",
                                  .occurrence = 1};
  ASSIGN_OR_RETURN(const auto* gelu, FindSite(trace, result.final_gelu_site));
  ASSIGN_OR_RETURN(auto gelu_row, QueryRow(*gelu, config.feed_forward_width));
  ASSIGN_OR_RETURN(const auto* projection,
                   FindSite(trace, result.final_projection_site));
  ASSIGN_OR_RETURN(auto projection_row,
                   QueryRow(*projection, config.model_width));
  ASSIGN_OR_RETURN(
      result.final_mlp,
      ComputeDenseProjectionReadoutAttribution(
          gelu_row, parameters.final_mlp_weights, parameters.final_mlp_bias,
          projection_row, result.attribution.direction));
  result.final_gelu_values.assign(gelu_row.begin(), gelu_row.end());
  result.final_mlp_residual_add_residual =
      result.attribution.boundaries.back().total -
      result.final_mlp.actual_directional_sum;
  if (!std::isfinite(result.final_mlp_residual_add_residual))
    return absl::InvalidArgumentError(
        "nonfinite final MLP residual accounting");
  return result;
}

absl::StatusOr<ScoredPrediction> Score(absl::Span<const float> logits,
                                       int target) {
  if (logits.size() < 3 || target < 0 ||
      static_cast<size_t>(target) >= logits.size() ||
      logits.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
    return absl::InvalidArgumentError(
        "invalid token-trace scoring shape/target");
  for (float value : logits)
    if (!std::isfinite(value))
      return absl::InvalidArgumentError("nonfinite token-trace logit");
  std::vector<int> order(logits.size());
  std::iota(order.begin(), order.end(), 0);
  auto greater = [&](int left, int right) {
    return logits[left] > logits[right] ||
           (logits[left] == logits[right] && left < right);
  };
  std::partial_sort(order.begin(), order.begin() + 3, order.end(), greater);
  const double maximum = logits[order[0]];
  double denominator = 0;
  double rival = -std::numeric_limits<double>::infinity();
  size_t rank = 1;
  for (size_t token = 0; token < logits.size(); ++token) {
    denominator += std::exp(static_cast<double>(logits[token]) - maximum);
    if (static_cast<int>(token) != target) {
      rival = std::max(rival, static_cast<double>(logits[token]));
      rank += greater(static_cast<int>(token), target);
    }
  }
  ScoredPrediction result{
      .predicted = order[0],
      .target = target,
      .target_rank = rank,
      .target_probability = std::exp(logits[target] - maximum) / denominator,
      .target_margin = static_cast<double>(logits[target]) - rival};
  for (size_t rank = 0; rank < 3; ++rank) {
    result.top_ids.push_back(order[rank]);
    result.top_probabilities.push_back(std::exp(logits[order[rank]] - maximum) /
                                       denominator);
  }
  return result;
}

absl::Span<const int> Prefix(const Sentence& sentence) {
  return absl::MakeConstSpan(sentence.tokens).first(kPrefixTokens);
}

int Target(const Sentence& sentence, int eos) {
  return sentence.tokens.size() == kPrefixTokens
             ? eos
             : sentence.tokens[kPrefixTokens];
}

absl::StatusOr<std::vector<Sentence>> ReadSentences(absl::string_view text) {
  std::vector<Sentence> result;
  while (!text.empty()) {
    const size_t end = text.find('\n');
    absl::string_view line = text.substr(0, end);
    if (!line.empty() && line.back() == '\r')
      line.remove_suffix(1);
    if (line.empty() ||
        line.find_first_not_of(" \t\r") == absl::string_view::npos)
      return absl::InvalidArgumentError("empty or blank corpus line");
    result.push_back({.line = result.size() + 1, .text = std::string(line)});
    if (end == absl::string_view::npos)
      break;
    text.remove_prefix(end + 1);
  }
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

absl::Status WriteCaptures(const fs::path& directory,
                           const std::vector<Baseline>& baselines,
                           const std::vector<std::string>& token_text,
                           absl::Span<const int> original_ids) {
  std::ofstream prefixes(directory / "prefix_tokens.tsv");
  std::ofstream activations(directory / "activations.tsv");
  std::ofstream attention(directory / "attention.tsv");
  if (!prefixes || !activations || !attention)
    return absl::UnknownError("cannot open token-trace capture files");
  prefixes
      << "line_1based\tposition_0based\tcompact_id\toriginal_id\ttoken_bytes\n";
  activations << std::setprecision(std::numeric_limits<float>::max_digits10)
              << "line_1based\tsite\tdtype\tposition_0based\tchannel\tvalue\n";
  attention
      << std::setprecision(std::numeric_limits<float>::max_digits10)
      << "line_1based\tsite\thead\tquery_0based\tkey_0based\tprobability\n";
  for (const auto& baseline : baselines) {
    const size_t line = baseline.sentence->line;
    for (size_t position = 0; position < kPrefixTokens; ++position) {
      const int id = baseline.sentence->tokens[position];
      prefixes << line << '\t' << position << '\t' << id << '\t'
               << original_ids[id] << '\t' << token_text[id] << '\n';
    }
    for (const auto& activation : baseline.trace.activations) {
      if (activation.channels == 0 || activation.row_count == 0 ||
          activation.row_count > kPrefixTokens ||
          activation.first_row > kPrefixTokens - activation.row_count ||
          activation.values.size() / activation.channels !=
              activation.row_count ||
          activation.values.size() % activation.channels != 0)
        return absl::InternalError("invalid captured activation dimensions");
      for (size_t row = 0; row < activation.row_count; ++row)
        for (size_t channel = 0; channel < activation.channels; ++channel)
          activations << line << '\t' << SiteName(activation.site) << '\t'
                      << DataTypeName(activation.data_type) << '\t'
                      << activation.first_row + row << '\t' << channel << '\t'
                      << activation.values[row * activation.channels + channel]
                      << '\n';
    }
    for (const auto& weights : baseline.trace.attention) {
      if (weights.prefix_length != kPrefixTokens || weights.heads == 0 ||
          weights.probabilities.size() / (kPrefixTokens * kPrefixTokens) !=
              weights.heads ||
          weights.probabilities.size() % (kPrefixTokens * kPrefixTokens) != 0)
        return absl::InternalError("invalid captured attention dimensions");
      for (size_t head = 0; head < weights.heads; ++head)
        for (size_t query = 0; query < kPrefixTokens; ++query)
          for (size_t key = 0; key <= query; ++key)
            attention << line << '\t' << SiteName(weights.site) << '\t' << head
                      << '\t' << query << '\t' << key << '\t'
                      << weights.probabilities[(head * kPrefixTokens + query) *
                                                   kPrefixTokens +
                                               key]
                      << '\n';
    }
  }
  for (std::ofstream* file : {&prefixes, &activations, &attention}) {
    file->close();
    if (!*file)
      return absl::UnknownError("cannot write token-trace capture files");
  }
  return absl::OkStatus();
}

std::string ReadoutBoundaryName(size_t boundary) {
  if (boundary == 0)
    return "embedding_plus_position";
  return absl::StrCat("block_", (boundary - 1) / 2,
                      boundary % 2 == 1 ? "_post_attention" : "_post_mlp");
}

absl::Status WriteReadoutAccounting(
    const fs::path& directory, const std::vector<Baseline>& baselines,
    const ReadoutParameters& parameters,
    const std::vector<std::string>& token_text) {
  std::ofstream summary(directory / "readout_margin_summary.tsv");
  std::ofstream boundaries(directory / "readout_boundaries.tsv");
  std::ofstream dimensions(directory / "readout_dimensions.tsv");
  std::ofstream boundary_dimensions(directory /
                                    "readout_boundary_dimensions.tsv");
  std::ofstream mlp_summary(directory / "readout_final_mlp_summary.tsv");
  std::ofstream mlp_neurons(directory / "readout_final_mlp_neurons.tsv");
  for (std::ofstream* file :
       {&summary, &boundaries, &dimensions, &boundary_dimensions, &mlp_summary,
        &mlp_neurons}) {
    if (!*file)
      return absl::UnknownError("cannot open readout accounting report files");
    *file << std::setprecision(std::numeric_limits<double>::max_digits10);
  }
  summary
      << "line_1based\tquery_row_0based\ttarget_compact_id\ttarget_"
         "bytes\trival_compact_id\trival_bytes\ttarget_gpu_logit\trival_gpu_"
         "logit\tgpu_logit_margin\tideal_margin\taccounted_margin\taccounting_"
         "residual\tactual_normalized_margin\tnormalization_residual\tgpu_head_"
         "residual\tgpu_minus_ideal\tfinal_mean\tfinal_"
         "variance\tdenominator\tepsilon\tbeta_contribution\n";
  boundaries << "line_1based\tboundary_0based\tboundary\tsite\tkind\tfixed_"
                "final_margin_term\n";
  dimensions
      << "line_1based\tdimension\ttarget_embedding_bf16\trival_embedding_"
         "bf16\tembedding_difference\tgamma_fp32\tbeta_fp32\tfixed_final_"
         "direction\tbeta_margin_term\tactual_normalized_dimension_term\n";
  boundary_dimensions
      << "line_1based\tboundary_0based\tboundary\tdimension\tfixed_final_"
         "margin_term\n";
  mlp_summary
      << "line_1based\tblock_0based\tquery_row_0based\tgelu_site\tprojection_"
         "site\tneuron_sum\tbias_margin_term\tideal_projection_margin_term\t"
         "actual_projection_margin_term\tprojection_arithmetic_residual\t"
         "observed_residual_increment_term\tresidual_add_arithmetic_residual\n";
  mlp_neurons
      << "line_1based\tblock_0based\tneuron_0based\tgelu_bf16\t"
         "effective_weight_row_dot_fixed_direction\tfixed_final_margin_term\n";
  for (const auto& baseline : baselines) {
    const auto& readout = baseline.readout;
    const auto& attribution = readout.attribution;
    const size_t line = baseline.sentence->line;
    const auto& projection = readout.final_mlp;
    mlp_summary << line << '\t' << readout.final_mlp_block << '\t'
                << kPrefixTokens - 1 << '\t'
                << SiteName(readout.final_gelu_site) << '\t'
                << SiteName(readout.final_projection_site) << '\t'
                << std::accumulate(projection.neuron_contributions.begin(),
                                   projection.neuron_contributions.end(), 0.0)
                << '\t' << projection.bias_contribution << '\t'
                << projection.ideal_directional_sum << '\t'
                << projection.actual_directional_sum << '\t'
                << projection.rounding_residual << '\t'
                << attribution.boundaries.back().total << '\t'
                << readout.final_mlp_residual_add_residual << '\n';
    for (size_t neuron = 0; neuron < projection.neuron_contributions.size();
         ++neuron)
      mlp_neurons << line << '\t' << readout.final_mlp_block << '\t' << neuron
                  << '\t' << readout.final_gelu_values[neuron] << '\t'
                  << projection.neuron_directions[neuron] << '\t'
                  << projection.neuron_contributions[neuron] << '\n';
    summary << line << '\t' << kPrefixTokens - 1 << '\t' << readout.target
            << '\t' << token_text[readout.target] << '\t' << readout.rival
            << '\t' << token_text[readout.rival] << '\t'
            << baseline.trace.logits[readout.target] << '\t'
            << baseline.trace.logits[readout.rival] << '\t'
            << readout.gpu_logit_margin << '\t' << attribution.ideal_margin
            << '\t' << attribution.accounted_margin << '\t'
            << attribution.accounting_residual << '\t'
            << attribution.actual_normalized_margin << '\t'
            << attribution.normalization_residual << '\t'
            << readout.gpu_head_residual << '\t'
            << readout.gpu_logit_margin - attribution.ideal_margin << '\t'
            << attribution.final_mean << '\t' << attribution.final_variance
            << '\t' << attribution.normalization_denominator << '\t'
            << kReadoutEpsilon << '\t' << attribution.beta_contribution << '\n';
    for (size_t boundary = 0; boundary < attribution.boundaries.size();
         ++boundary) {
      const auto& contribution = attribution.boundaries[boundary];
      boundaries << line << '\t' << boundary << '\t'
                 << ReadoutBoundaryName(boundary) << '\t'
                 << SiteName(readout.residual_sites[boundary]) << '\t'
                 << (boundary == 0 ? "initial_state"
                                   : "observed_residual_increment")
                 << '\t' << contribution.total << '\n';
      for (size_t dimension = 0;
           dimension < contribution.dimension_contributions.size(); ++dimension)
        boundary_dimensions
            << line << '\t' << boundary << '\t' << ReadoutBoundaryName(boundary)
            << '\t' << dimension << '\t'
            << contribution.dimension_contributions[dimension] << '\n';
    }
    for (size_t dimension = 0; dimension < attribution.direction.size();
         ++dimension)
      dimensions
          << line << '\t' << dimension << '\t'
          << parameters
                 .effective_embeddings[static_cast<size_t>(readout.target) *
                                           parameters.width +
                                       dimension]
          << '\t'
          << parameters
                 .effective_embeddings[static_cast<size_t>(readout.rival) *
                                           parameters.width +
                                       dimension]
          << '\t' << attribution.embedding_difference[dimension] << '\t'
          << parameters.gamma[dimension] << '\t' << parameters.beta[dimension]
          << '\t' << attribution.direction[dimension] << '\t'
          << parameters.beta[dimension] *
                 attribution.embedding_difference[dimension]
          << '\t' << attribution.actual_dimension_contributions[dimension]
          << '\n';
  }
  for (std::ofstream* file :
       {&summary, &boundaries, &dimensions, &boundary_dimensions, &mlp_summary,
        &mlp_neurons}) {
    file->close();
    if (!*file)
      return absl::UnknownError("cannot write readout accounting report files");
  }
  return absl::OkStatus();
}

void WriteReadoutHtml(std::ostream& out, const Baseline& baseline,
                      const std::vector<std::string>& token_text) {
  const auto& readout = baseline.readout;
  const auto& attribution = readout.attribution;
  out << "<h3>Fixed-final-normalizer margin accounting</h3><p>Target "
         "<code>&quot;"
      << EscapeHtml(token_text[readout.target])
      << "&quot;</code> minus final strongest rival <code>&quot;"
      << EscapeHtml(token_text[readout.rival])
      << "&quot;</code>. This rival "
         "and the final normalization are held fixed for every term. "
         "Positive terms favor this target over this rival only. "
         "<strong>This is linear accounting of an observed trajectory, not "
         "a causal effect or an early-layer prediction.</strong></p>"
         "<p>Using the actual BF16 embedding operands, let "
         "<code>a = center(gamma * (E_target - E_rival)) / "
         "sqrt(var(x_final) + epsilon)</code>. The initial term is "
         "<code>a dot x_initial</code>; subsequent terms are "
         "<code>a dot (x_current - x_previous)</code>. These observed residual "
         "differences include residual-add rounding and are not raw branch "
         "outputs. Add <code>beta dot (E_target - E_rival)</code> once. "
         "No extra model forward is needed for this accounting.</p>"
         "<table><tr><th>Observed boundary / numerical check</th><th>Margin "
         "term</th></tr>";
  for (size_t boundary = 0; boundary < attribution.boundaries.size();
       ++boundary)
    out << "<tr><td>" << ReadoutBoundaryName(boundary)
        << (boundary == 0 ? " (initial state)" : " (increment)") << "</td><td>"
        << attribution.boundaries[boundary].total << "</td></tr>";
  out << "<tr><td>Final LayerNorm beta term</td><td>"
      << attribution.beta_contribution
      << "</td></tr><tr><th>Sum of boundary terms + beta</th><th>"
      << attribution.accounted_margin
      << "</th></tr><tr><td>Literal ideal final LayerNorm/head margin</td><td>"
      << attribution.ideal_margin
      << "</td></tr><tr><td>Actual captured normalized row, double-dot "
         "margin</td><td>"
      << attribution.actual_normalized_margin
      << "</td></tr><tr><th>Actual GPU logit margin</th><th>"
      << readout.gpu_logit_margin
      << "</th></tr><tr><td>Accounting closure: sum - ideal</td><td>"
      << attribution.accounting_residual
      << "</td></tr><tr><td>Normalization residual: actual-normalized - "
         "ideal</td><td>"
      << attribution.normalization_residual
      << "</td></tr><tr><td>Head arithmetic residual: GPU - "
         "actual-normalized</td><td>"
      << readout.gpu_head_residual
      << "</td></tr></table><p>The normalization residual includes FP32 "
         "normalization/affine arithmetic and BF16 output rounding. The head "
         "residual compares GPU dot products against double arithmetic on "
         "those same actual normalized values; saved FP32 GPU logits are "
         "subtracted in double. Numerical residuals are displayed separately, "
         "never assigned to an invented layer contribution. Epsilon is the "
         "recipe's float <code>1e-5f</code>, expanded to double.</p>"
         "<details><summary>Actual final-normalized contribution by "
         "dimension</summary>"
         "<p>Each actual dimension term is the captured normalized value "
         "times the effective target-minus-rival embedding value. These terms "
         "already include beta: do not add the beta term again. All boundary "
         "and dimension terms are also written to the readout_*.tsv files.</p>"
         "<table><tr><th>Dimension</th><th>Effective embedding difference</th>"
         "<th>Fixed final direction</th><th>Actual normalized dimension "
         "term</th></tr>";
  for (size_t dimension = 0; dimension < attribution.direction.size();
       ++dimension)
    out << "<tr><td>" << dimension << "</td><td>"
        << attribution.embedding_difference[dimension] << "</td><td>"
        << attribution.direction[dimension] << "</td><td>"
        << attribution.actual_dimension_contributions[dimension]
        << "</td></tr>";
  out << "</table></details>";
  const auto& projection = readout.final_mlp;
  out << "<h3>Final MLP projection: contributions from individual neurons</h3>"
         "<p>At query row 4 in block "
      << readout.final_mlp_block
      << ", neuron <code>j</code> contributes "
         "<code>GELU[j] * dot(W2[j, :], a)</code> along the same fixed final "
         "direction <code>a</code> above. GELU and W2 use their actual BF16 "
         "operands; the bias remains FP32. This decomposes the observed "
         "projection arithmetic. <strong>It is not a neuron deletion effect, "
         "a claim of exclusive fact ownership, or evidence of an independent "
         "fact stored in a weight row.</strong> No inference is rerun.</p>"
         "<table><tr><th>Projection / numerical check</th><th>Margin term</th>"
         "</tr><tr><td>Sum of neuron terms</td><td>"
      << std::accumulate(projection.neuron_contributions.begin(),
                         projection.neuron_contributions.end(), 0.0)
      << "</td></tr><tr><td>FP32 contraction bias term</td><td>"
      << projection.bias_contribution
      << "</td></tr><tr><td>Ideal projection sum</td><td>"
      << projection.ideal_directional_sum
      << "</td></tr><tr><td>Actual captured contraction output</td><td>"
      << projection.actual_directional_sum
      << "</td></tr><tr><td>Projection arithmetic residual: actual - ideal</td>"
         "<td>"
      << projection.rounding_residual
      << "</td></tr><tr><td>Residual-add arithmetic residual: observed "
         "increment - actual contraction</td><td>"
      << readout.final_mlp_residual_add_residual
      << "</td></tr><tr><th>Observed final residual increment</th><th>"
      << attribution.boundaries.back().total
      << "</th></tr></table><p>The first numerical residual includes "
         "projection accumulation, bias addition, and BF16 output rounding. "
         "The second isolates the residual addition. Neither is assigned to "
         "a neuron. TSV files retain every term.</p>"
         "<details><summary>All final MLP neurons, sorted by absolute "
         "contribution</summary><table><tr><th>Neuron</th><th>GELU value</th>"
         "<th>W2 row dot fixed direction</th><th>Signed margin term</th></tr>";
  std::vector<size_t> neurons(projection.neuron_contributions.size());
  std::iota(neurons.begin(), neurons.end(), 0);
  std::sort(neurons.begin(), neurons.end(), [&](size_t left, size_t right) {
    const double a = std::abs(projection.neuron_contributions[left]);
    const double b = std::abs(projection.neuron_contributions[right]);
    return a > b || (a == b && left < right);
  });
  for (size_t neuron : neurons)
    out << "<tr><td>" << neuron << "</td><td>"
        << readout.final_gelu_values[neuron] << "</td><td>"
        << projection.neuron_directions[neuron] << "</td><td>"
        << projection.neuron_contributions[neuron] << "</td></tr>";
  out << "</table></details>";
}

absl::Status WriteHtml(const fs::path& directory, const fs::path& checkpoint,
                       const std::vector<Baseline>& baselines,
                       const std::vector<Condition>& conditions,
                       const std::vector<std::string>& token_text,
                       bool complete) {
  std::ofstream out(directory / "report.html", std::ios::trunc);
  if (!out)
    return absl::UnknownError("cannot open token-trace HTML report");
  out << std::setprecision(7)
      << "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
         "<meta name=\"viewport\" "
         "content=\"width=device-width,initial-scale=1\">"
         "<title>Five-token checkpoint trace</title><style>"
         "body{font:14px system-ui,sans-serif;margin:2rem}"
         "table{border-collapse:collapse;margin:1rem "
         "0;font-variant-numeric:tabular-nums}"
         "td,th{border:1px solid #ddd;padding:.35rem;text-align:right}"
         "td:first-child,th:first-child{text-align:left}th{background:#eee}"
         "code{white-space:pre-wrap}details{margin:.5rem 0}.wide{overflow:auto}"
         "</style><body><h1>Five-token checkpoint trace</h1><p>"
      << (complete ? "Complete." : "Partial: interventions still pending.")
      << " Checkpoint: <code>" << EscapeHtml(checkpoint.string())
      << "</code></p>"
         "<p>Every forward receives exactly the first five tokens and EOS in "
         "all later positions. Only position 4's actual language-model-head "
         "logits are scored: it predicts token 6 (or EOS for a five-token "
         "fact). "
         "No gold suffix is fed to the model. These are one-token diagnostics "
         "and causal interventions, "
         "not full-completion or exclusive-fact-ownership claims.</p>"
         "<p>Branch zeroing removes the branch output at all five prefix rows, "
         "retaining its residual skip. Donor transplants replace one selected "
         "row/channel slice from an independently traced capital prompt. Each "
         "condition starts a fresh forward with unchanged weights; conditions "
         "are never accumulated. Logit margin is target minus best rival. "
         "Ranks break ties by lower token ID. Softmax uses all compact "
         "tokens. A late residual transplant can transfer an already computed "
         "answer; it does not locate where a fact is stored. Full query-row "
         "restoration before versus after a pointwise MLP can commute with "
         "that MLP, so branch ablations are a separate necessity test.</p>"
         "<p><strong>No cross-position copies:</strong> country-row donors "
         "copy row 3 to row 3, and query-row donors copy row 4 to row 4. "
         "A row-3 intervention can affect the scored row 4 only through "
         "remaining causal attention; it never directly copies row 3 into "
         "row 4. Q/K/V patches replace only the indicated slice of that "
         "same row. Optional <code>zero_mlp_neuron_*</code> conditions zero "
         "one GELU channel at query row 4 only, preserving all prior rows.</p>"
         "<p><strong>Diagnostic readout lens:</strong> conditions named "
         "<code>readout_lens_*</code>, when enabled, copy an earlier row-4 "
         "residual into "
         "the last block's post-MLP query row, then apply the checkpoint's "
         "original final LayerNorm and language-model head. Earlier blocks "
         "still execute, but their final query residual is overwritten. "
         "These are counterfactual readouts, <strong>not actual early-layer "
         "predictions or tests of causal necessity</strong>. A poor early "
         "readout does not establish that the answer is absent from that "
         "representation. The final post-MLP self-copy must reproduce the "
         "baseline logits bit-for-bit.</p>"
         "<p>Uninstrumented and captured baseline query logits were checked "
         "bit-for-bit. An identity donor-copy control checks unchanged logits. "
         "Positions, blocks, heads, channels, and site occurrences are "
         "zero-based; "
         "corpus line numbers and token ranks are one-based. Byte escapes "
         "preserve token spelling, including leading spaces.</p>";
  for (const auto& baseline : baselines) {
    out << "<h2>Line " << baseline.sentence->line << ": <code>"
        << EscapeHtml(DisplayBytes(baseline.prefix_text))
        << "</code></h2>"
           "<p>Reference text (not a model input): "
        << EscapeHtml(baseline.sentence->text) << "</p><p>Prefix pieces: ";
    for (size_t row = 0; row < kPrefixTokens; ++row)
      out << row << ": <code>&quot;"
          << EscapeHtml(token_text[baseline.sentence->tokens[row]])
          << "&quot;</code> ";
    out << "</p>";
    WriteReadoutHtml(out, baseline, token_text);
    out << "<h3>Compact readout and branch summary</h3>"
           "<p>Readouts diagnose compatibility with the final head; branch "
           "zeroing tests the effect of removing that branch at all prefix "
           "rows. These are different questions, not interchangeable "
           "measures of where a fact is stored.</p><div class=\"wide\"><table>"
           "<tr><th>Test</th><th>Condition</th><th>Prediction</th>"
           "<th>Target probability</th><th>Target rank</th><th>Target "
           "margin</th></tr>";
    for (const auto& condition : conditions) {
      if (condition.line != baseline.sentence->line)
        continue;
      const char* kind =
          condition.name == "baseline"                  ? "Actual baseline"
          : condition.name.starts_with("readout_lens_") ? "Diagnostic readout"
          : condition.name.starts_with("zero_block_")   ? "Branch zero"
                                                        : nullptr;
      if (kind == nullptr)
        continue;
      const auto& score = condition.prediction;
      out << "<tr><td>" << kind << "</td><td>" << EscapeHtml(condition.name)
          << "</td><td><code>&quot;" << EscapeHtml(token_text[score.predicted])
          << "&quot;</code></td><td>" << score.target_probability << "</td><td>"
          << score.target_rank << "</td><td>" << score.target_margin
          << "</td></tr>";
    }
    const auto condition_count = std::count_if(
        conditions.begin(), conditions.end(), [&](const Condition& condition) {
          return condition.line == baseline.sentence->line;
        });
    out << "</table></div><details><summary>All " << condition_count
        << " recorded conditions, including controls and donor/neuron "
           "interventions</summary><div "
           "class=\"wide\"><table><thead><tr><th>Condition</th>"
           "<th>Site / rows / channels</th><th>Prediction</th><th>Target</th>"
           "<th>Target rank</th><th>Target probability</th><th>Probability "
           "change</th>"
           "<th>Target margin</th><th>Top 3</th><th>Donor target "
           "probability</th>"
           "</tr></thead><tbody>";
    for (const auto& condition : conditions) {
      if (condition.line != baseline.sentence->line)
        continue;
      const auto& score = condition.prediction;
      out << "<tr><td>" << EscapeHtml(condition.name);
      if (condition.identity_control)
        out << " <strong>[bitwise identity control]</strong>";
      out << "</td><td><code>" << EscapeHtml(condition.site) << "</code> ";
      if (!condition.source_site.empty())
        out << "from <code>" << EscapeHtml(condition.source_site) << "</code> ";
      out << EscapeHtml(condition.rows);
      if (condition.channel_count != 0)
        out << " channels [" << condition.first_channel << ','
            << condition.first_channel + condition.channel_count << ')';
      out << "</td><td><code>&quot;" << EscapeHtml(token_text[score.predicted])
          << "&quot;</code> (" << score.predicted << ")</td><td><code>&quot;"
          << EscapeHtml(token_text[score.target]) << "&quot;</code></td><td>"
          << score.target_rank << "</td><td>" << score.target_probability
          << "</td><td>" << condition.target_probability_change << "</td><td>"
          << score.target_margin << "</td><td>";
      for (size_t rank = 0; rank < score.top_ids.size(); ++rank)
        out << (rank == 0 ? "" : "; ") << "<code>&quot;"
            << EscapeHtml(token_text[score.top_ids[rank]]) << "&quot;</code> "
            << score.top_probabilities[rank];
      out << "</td><td>";
      if (condition.donor_target >= 0)
        out << "line " << condition.donor_line << ": <code>&quot;"
            << EscapeHtml(token_text[condition.donor_target])
            << "&quot;</code> " << condition.donor_target_probability;
      out << "</td></tr>";
    }
    out << "</tbody></table></div></details><details><summary>Causal attention "
           "matrices"
           " (baseline only)</summary>";
    for (const auto& weights : baseline.trace.attention)
      for (size_t head = 0; head < weights.heads; ++head) {
        out << "<p><code>" << EscapeHtml(SiteName(weights.site))
            << "</code>, head " << head
            << "</p><table><tr><th>query / key</th>";
        for (size_t key = 0; key < kPrefixTokens; ++key)
          out << "<th>" << key << "</th>";
        out << "</tr>";
        for (size_t query = 0; query < kPrefixTokens; ++query) {
          out << "<tr><th>" << query << "</th>";
          for (size_t key = 0; key < kPrefixTokens; ++key) {
            out << "<td>";
            if (key <= query)
              out << weights.probabilities[(head * kPrefixTokens + query) *
                                               kPrefixTokens +
                                           key];
            out << "</td>";
          }
          out << "</tr>";
        }
        out << "</table>";
      }
    out << "</details><details><summary>Raw intermediate activation vectors "
           "(baseline only)</summary><p>Every prefix row is shown below. Head "
           "and outer-model logits are summarized by the prediction table; "
           "all real vocabulary logits are included in activations.tsv. "
           "Raw head/root snapshots also include physical padded vocabulary "
           "lanes; these never enter prediction ranks or probabilities. "
           "Values are physical-dtype values expanded for printing.</p>";
    for (const auto& activation : baseline.trace.activations) {
      if (activation.site.layer_name == "LanguageModelingHeadLayer" ||
          activation.site.scope.empty())
        continue;
      out << "<details><summary><code>" << EscapeHtml(SiteName(activation.site))
          << "</code> " << DataTypeName(activation.data_type) << " ["
          << activation.row_count << ',' << activation.channels
          << "]</summary><table><tr><th>Position</th><th>Channels in "
             "order</th></tr>";
      for (size_t row = 0; row < activation.row_count; ++row) {
        out << "<tr><td>" << activation.first_row + row << "</td><td><code>";
        for (size_t channel = 0; channel < activation.channels; ++channel)
          out << (channel == 0 ? "" : ", ")
              << activation.values[row * activation.channels + channel];
        out << "</code></td></tr>";
      }
      out << "</table></details>";
    }
    out << "</details>";
  }
  out << "</body></html>\n";
  out.close();
  if (!out)
    return absl::UnknownError("cannot write token-trace HTML report");
  return absl::OkStatus();
}

absl::Status Run() {
  const fs::path checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const fs::path directory = absl::GetFlag(FLAGS_output_dir);
  if (checkpoint.empty() || directory.empty() ||
      absl::GetFlag(FLAGS_tokenizer).empty() ||
      absl::GetFlag(FLAGS_expected_sentences) <= 0 ||
      absl::GetFlag(FLAGS_layers) <= 0)
    return absl::InvalidArgumentError(
        "checkpoint, tokenizer, fresh output_dir and positive dimensions "
        "required");
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  ASSIGN_OR_RETURN(auto sentences, ReadSentences(corpus.text()));
  if (sentences.size() !=
      static_cast<size_t>(absl::GetFlag(FLAGS_expected_sentences)))
    return absl::InvalidArgumentError(
        "corpus line count differs from expected_sentences");
  std::vector<size_t> selected;
  for (const std::string& value : absl::GetFlag(FLAGS_lines)) {
    size_t line = 0;
    if (!absl::SimpleAtoi(value, &line) || line == 0 ||
        line > sentences.size() ||
        std::find(selected.begin(), selected.end(), line - 1) != selected.end())
      return absl::InvalidArgumentError(
          "lines must be distinct one-based corpus indices");
    selected.push_back(line - 1);
  }
  if (selected.empty())
    return absl::InvalidArgumentError("at least one selected line is required");
  // Validate the optional sweep before creating an Executor or touching the
  // GPU, using the same strict distinct-integer policy as selected lines.
  std::vector<int> neuron_blocks;
  for (const auto& text : absl::GetFlag(FLAGS_mlp_neuron_ablation_blocks)) {
    int block = -1;
    if (!absl::SimpleAtoi(text, &block) || block < 0 ||
        block >= absl::GetFlag(FLAGS_layers) ||
        std::find(neuron_blocks.begin(), neuron_blocks.end(), block) !=
            neuron_blocks.end())
      return absl::InvalidArgumentError(
          "mlp_neuron_ablation_blocks must be distinct integers in [0,layers)");
    neuron_blocks.push_back(block);
  }
  std::vector<int> neuron_channels;
  for (const auto& text : absl::GetFlag(FLAGS_mlp_neuron_ablation_channels)) {
    int channel = -1;
    if (!absl::SimpleAtoi(text, &channel) || channel < 0 ||
        channel >= absl::GetFlag(FLAGS_feed_forward_width) ||
        std::find(neuron_channels.begin(), neuron_channels.end(), channel) !=
            neuron_channels.end())
      return absl::InvalidArgumentError(
          "mlp_neuron_ablation_channels must be distinct integers in "
          "[0,feed_forward_width)");
    neuron_channels.push_back(channel);
  }
  const bool all_neuron_channels = neuron_channels.empty();
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
    return absl::InvalidArgumentError(
        "checkpoint/tokenizer vocabulary identity differs");
  const Gpt2Config config{
      .transformer_block_count = absl::GetFlag(FLAGS_layers),
      .model_width = absl::GetFlag(FLAGS_model_width),
      .attention_heads = absl::GetFlag(FLAGS_attention_heads),
      .feed_forward_width = absl::GetFlag(FLAGS_feed_forward_width),
      .vocabulary_size = vocabulary->vocab_size(),
      .pad_vocabulary = false};
  RETURN_IF_ERROR(config.Validate());
  if (all_neuron_channels) {
    neuron_channels.resize(config.feed_forward_width);
    std::iota(neuron_channels.begin(), neuron_channels.end(), 0);
  }
  std::error_code error;
  if (!fs::create_directory(directory, error))
    return absl::InvalidArgumentError(
        absl::StrCat("output_dir must be fresh: ", error.message()));
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  // Encoding the full corpus validates checkpoint vocabulary compatibility.
  // The only data passed to TraceNextToken below is an explicit five-ID span.
  for (auto& sentence : sentences) {
    ASSIGN_OR_RETURN(auto tokens, vocabulary->Encode(*executor, sentence.text));
    if (tokens.size() < kPrefixTokens || tokens.size() > kGpt2ContextLength)
      return absl::InvalidArgumentError(
          absl::StrCat("line ", sentence.line, " has an invalid token count"));
    sentence.tokens.assign(tokens.begin(), tokens.end());
  }
  std::vector<std::string> token_text;
  for (int id = 0; id < vocabulary->vocab_size(); ++id) {
    ASSIGN_OR_RETURN(auto text, Decode(absl::Span<const int>(&id, 1),
                                       *vocabulary, *detokenizer));
    token_text.push_back(DisplayBytes(text));
  }
  // Find donors by corpus contents, not assumptions about corpus line order.
  std::vector<size_t> capitals;
  if (absl::GetFlag(FLAGS_capital_donor_patches)) {
    for (const char* country : {"France", "Greece", "Peru"}) {
      const std::string expected =
          absl::StrCat("The capital of ", country, " is");
      size_t found = sentences.size();
      for (size_t index = 0; index < sentences.size(); ++index)
        if (sentences[index].text.starts_with(expected + " ")) {
          if (found != sentences.size())
            return absl::InvalidArgumentError(
                "ambiguous capital donor sentence");
          found = index;
        }
      if (found == sentences.size())
        return absl::NotFoundError(
            absl::StrCat("capital donor absent from corpus: ", country));
      ASSIGN_OR_RETURN(auto prefix, Decode(Prefix(sentences[found]),
                                           *vocabulary, *detokenizer));
      if (prefix != expected)
        return absl::InvalidArgumentError(
            "capital donor must occupy exactly five prefix tokens");
      capitals.push_back(found);
      if (std::find(selected.begin(), selected.end(), found) == selected.end())
        selected.push_back(found);
    }
    for (size_t row = 0; row < kPrefixTokens; ++row)
      for (size_t donor = 1; donor < capitals.size(); ++donor)
        if ((row == 3) == (sentences[capitals[0]].tokens[row] ==
                           sentences[capitals[donor]].tokens[row]))
          return absl::InvalidArgumentError(
              "capital prompts must differ only at country position 3");
  }
  ASSIGN_OR_RETURN(auto model, CreateGpt2(*executor, DataType::BF16, 0, config));
  RETURN_IF_ERROR(
      ReadFromDirectory(*executor, *model, checkpoint.string(), false));
  ASSIGN_OR_RETURN(auto readout_parameters,
                   CopyReadoutParameters(*executor, *model, config));
  const TokenTraceOptions plain{.vocabulary_size = config.vocabulary_size,
                                .padding_token = vocabulary->eos_token_id()};
  std::vector<Baseline> baselines;
  for (size_t index : selected) {
    const Sentence& sentence = sentences[index];
    std::cout << "Capturing line " << sentence.line << std::endl;
    ASSIGN_OR_RETURN(auto plain_result,
                     TraceNextToken(*executor, *model, Prefix(sentence), plain));
    TokenTraceOptions capture = plain;
    capture.capture_activations = true;
    capture.capture_attention = true;
    ASSIGN_OR_RETURN(auto trace, TraceNextToken(*executor, *model,
                                                Prefix(sentence), capture));
    if (trace.query_row != kPrefixTokens - 1 ||
        trace.logits.size() != static_cast<size_t>(config.vocabulary_size) ||
        trace.attention.size() !=
            static_cast<size_t>(config.transformer_block_count))
      return absl::InternalError(
          "captured prefix does not match GPT-2 dimensions");
    for (const auto& attention : trace.attention)
      if (attention.heads != static_cast<size_t>(config.attention_heads))
        return absl::InternalError(
            "captured attention head count differs from configuration");
    if (plain_result.logits.size() != trace.logits.size() ||
        std::memcmp(plain_result.logits.data(), trace.logits.data(),
                    trace.logits.size() * sizeof(float)) != 0)
      return absl::FailedPreconditionError(
          "captured baseline differs from uninstrumented logits");
    if (trace.predicted_token != Target(sentence, vocabulary->eos_token_id()))
      return absl::FailedPreconditionError(absl::StrCat(
          "clean checkpoint does not correctly predict the sixth token of "
          "line ",
          sentence.line,
          "; cannot interpret these as answer-preservation tests"));
    ASSIGN_OR_RETURN(auto prefix_text,
                     Decode(Prefix(sentence), *vocabulary, *detokenizer));
    ASSIGN_OR_RETURN(
        auto readout,
        AccountReadout(trace, config, readout_parameters,
                       Target(sentence, vocabulary->eos_token_id())));
    baselines.push_back({.sentence = &sentence,
                         .prefix_text = std::move(prefix_text),
                         .trace = std::move(trace),
                         .readout = std::move(readout)});
  }
  RETURN_IF_ERROR(WriteCaptures(directory, baselines, token_text,
                                vocabulary->original_token_ids()));
  RETURN_IF_ERROR(WriteReadoutAccounting(directory, baselines,
                                         readout_parameters, token_text));
  std::ofstream manifest(directory / "manifest.tsv");
  std::ofstream table(directory / "conditions.tsv");
  std::ofstream tops(directory / "top_tokens.tsv");
  if (!manifest || !table || !tops)
    return absl::UnknownError("cannot create token-trace summary files");
  manifest
      << "checkpoint\t" << DisplayBytes(checkpoint.string()) << "\ncorpus\t"
      << DisplayBytes(absl::GetFlag(FLAGS_corpus)) << "\ntokenizer\t"
      << DisplayBytes(absl::GetFlag(FLAGS_tokenizer))
      << "\nprefix_token_count\t5\ncontext_length\t" << kGpt2ContextLength
      << "\nvocabulary\t" << config.vocabulary_size << "\nlayers\t"
      << config.transformer_block_count << "\nmodel_width\t"
      << config.model_width << "\nattention_heads\t" << config.attention_heads
      << "\nfeed_forward_width\t" << config.feed_forward_width
      << "\ncompute_type\tBF16\nfuture_input\tEOS_only\nbranch_ablations\t"
      << absl::GetFlag(FLAGS_branch_ablations) << "\ncapital_donor_patches\t"
      << absl::GetFlag(FLAGS_capital_donor_patches) << "\nreadout_lens\t"
      << absl::GetFlag(FLAGS_readout_lens)
      << "\nhook_vs_no_hook_logits\tbitwise_equal_for_every_selected_prefix\n";
  manifest << "raw_head_channels\tinclude_physical_padding\n"
              "prediction_vocabulary\treal_compact_ids_only\n"
              "capital_donor_selection\tappend_missing_donor_baselines_when_"
              "enabled\n";
  manifest
      << std::setprecision(std::numeric_limits<double>::max_digits10)
      << "readout_accounting\tfixed_final_normalizer_not_causal\n"
         "readout_accounting_rival\tstrongest_other_actual_baseline_logit\n"
         "readout_parameter_source\tvalidated_recipe_inventory_pinned_D2H\n"
         "readout_epsilon_recipe_float\t"
      << kReadoutEpsilon
      << "\nreadout_increments\tdifferences_of_observed_rounded_residuals\n"
         "gpu_logit_margin_subtraction\tsaved_FP32_logits_subtracted_in_"
         "double\n"
         "final_mlp_neuron_accounting\tfixed_final_direction_not_causal\n"
         "final_mlp_operands\tcaptured_BF16_GELU_effective_BF16_W2_FP32_bias\n"
         "final_mlp_numerical_residuals\tprojection_arithmetic_and_residual_"
         "add_"
         "reported_separately\n";
  for (const auto& baseline : baselines)
    manifest << "traced_line_1based\t" << baseline.sentence->line << '\n';
  manifest << "mlp_neuron_ablation_block_count\t" << neuron_blocks.size()
           << "\nmlp_neuron_ablation_row_0based\t4\n";
  for (int block : neuron_blocks)
    manifest << "mlp_neuron_ablation_block_0based\t" << block << '\n';
  manifest << "mlp_neuron_ablation_channel_selection\t"
           << (all_neuron_channels ? "all" : "explicit")
           << "\nmlp_neuron_ablation_channel_count\t" << neuron_channels.size()
           << '\n';
  for (int channel : neuron_channels)
    manifest << "mlp_neuron_ablation_channel_0based\t" << channel << '\n';
  manifest.flush();
  if (!manifest)
    return absl::UnknownError("cannot write token-trace manifest");
  table << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "line_1based\tcondition\tdonor_line_1based\tsite\tsource_"
           "site\trows\tfirst_row_0based\trow_count\tfirst_"
           "channel\tchannel_count\tblock\tneuron\tidentity_control\tpredicted_"
           "compact_id\ttarget_compact_"
           "id\ttarget_rank\ttarget_probability\ttarget_probability_"
           "change\ttarget_margin\tdonor_target_compact_id\tdonor_target_"
           "probability\n";
  tops << std::setprecision(std::numeric_limits<double>::max_digits10)
       << "line_1based\tcondition\trank\tcompact_id\toriginal_"
          "id\tprobability\ttoken_bytes\n";
  std::vector<Condition> conditions;
  auto record = [&](const Baseline& baseline, Condition condition,
                    const TokenTraceResult& result) -> absl::Status {
    condition.line = baseline.sentence->line;
    ASSIGN_OR_RETURN(condition.prediction,
                     Score(result.logits, Target(*baseline.sentence,
                                                 vocabulary->eos_token_id())));
    if (condition.prediction.predicted != result.predicted_token)
      return absl::InternalError("trace library and report top-1 disagree");
    ASSIGN_OR_RETURN(auto baseline_score,
                     Score(baseline.trace.logits, condition.prediction.target));
    condition.target_probability_change =
        condition.prediction.target_probability -
        baseline_score.target_probability;
    if (condition.donor_target >= 0) {
    ASSIGN_OR_RETURN(auto donor_score,
                     Score(result.logits, condition.donor_target));
      condition.donor_target_probability = donor_score.target_probability;
    }
    const auto& score = condition.prediction;
    table << condition.line << '\t' << condition.name << '\t'
          << condition.donor_line << '\t' << condition.site << '\t'
          << condition.source_site << '\t' << condition.rows << '\t'
          << condition.first_row << '\t' << condition.row_count << '\t'
          << condition.first_channel << '\t' << condition.channel_count << '\t'
          << condition.block << '\t' << condition.neuron << '\t'
          << condition.identity_control << '\t' << score.predicted << '\t'
          << score.target << '\t' << score.target_rank << '\t'
          << score.target_probability << '\t'
          << condition.target_probability_change << '\t' << score.target_margin
          << '\t' << condition.donor_target << '\t'
          << condition.donor_target_probability << '\n';
    for (size_t rank = 0; rank < score.top_ids.size(); ++rank) {
      const int token = score.top_ids[rank];
      tops << condition.line << '\t' << condition.name << '\t' << rank + 1
           << '\t' << token << '\t' << vocabulary->original_token_ids()[token]
           << '\t' << score.top_probabilities[rank] << '\t' << token_text[token]
           << '\n';
    }
    table.flush();
    tops.flush();
    if (!table || !tops)
      return absl::UnknownError("cannot write token-trace prediction files");
    std::cout << "line " << condition.line << ' ' << condition.name << " -> \""
              << token_text[score.predicted]
              << "\" target_rank=" << score.target_rank
              << " target_probability=" << score.target_probability
              << " margin=" << score.target_margin << std::endl;
    conditions.push_back(std::move(condition));
    return absl::OkStatus();
  };
  auto intervene = [&](const Baseline& baseline, const TokenTracePatch& patch,
                       Condition condition, bool identity) -> absl::Status {
    ASSIGN_OR_RETURN(const auto* recipient,
                     FindSite(baseline.trace, patch.site));
    if (patch.first_channel >= recipient->channels ||
        patch.channel_count > recipient->channels - patch.first_channel)
      return absl::InvalidArgumentError("invalid driver patch channel range");
    condition.channel_count = patch.channel_count == 0
                                  ? recipient->channels - patch.first_channel
                                  : patch.channel_count;
    // Derive report coordinates from the actual patch rather than trusting
    // hand-written labels. Donor source and recipient use identical row IDs.
    condition.first_row = patch.rows == TokenTraceRows::kQuery
                              ? kPrefixTokens - 1
                          : patch.rows == TokenTraceRows::kOne ? patch.row
                                                               : 0;
    condition.row_count =
        patch.rows == TokenTraceRows::kAllPrefix ? kPrefixTokens : 1;
    condition.rows = patch.rows == TokenTraceRows::kAllPrefix
                         ? "all_prefix"
                         : absl::StrCat("row_", condition.first_row);
    condition.identity_control = identity;
    TokenTraceOptions options = plain;
    options.patches = absl::Span<const TokenTracePatch>(&patch, 1);
    ASSIGN_OR_RETURN(
        auto result,
        TraceNextToken(*executor, *model, Prefix(*baseline.sentence), options));
    if (result.query_row != kPrefixTokens - 1 ||
        result.logits.size() != static_cast<size_t>(config.vocabulary_size))
      return absl::InternalError("intervention changed the scored query shape");
    if (identity &&
        (result.logits.size() != baseline.trace.logits.size() ||
         std::memcmp(result.logits.data(), baseline.trace.logits.data(),
                     result.logits.size() * sizeof(float)) != 0))
      return absl::FailedPreconditionError(
          "identity or structurally disconnected donor patch changed logits");
    condition.site = SiteName(patch.site);
    if (patch.donor != nullptr)
      condition.source_site = SiteName(patch.donor->site);
    condition.first_channel = patch.first_channel;
    return record(baseline, std::move(condition), result);
  };
  for (const auto& baseline : baselines)
    RETURN_IF_ERROR(record(baseline, {.name = "baseline"}, baseline.trace));
  for (const auto& baseline : baselines) {
    const TokenTraceSite position{.scope = {"gpt2"},
                                  .layer_name = "PositionEmbeddingLayer",
                                  .occurrence = 0};
    ASSIGN_OR_RETURN(const auto* embedding, FindSite(baseline.trace, position));
    const TokenTracePatch identity{.site = embedding->site,
                                   .rows = TokenTraceRows::kAllPrefix,
                                   .replacement = TokenTraceReplacement::kDonor,
                                   .donor = embedding};
    RETURN_IF_ERROR(intervene(
        baseline, identity,
        {.name = "identity_donor_control", .rows = "all_prefix"}, true));
  }
  RETURN_IF_ERROR(WriteHtml(directory, checkpoint, baselines, conditions,
                            token_text, false));
  for (const auto& baseline : baselines) {
    if (absl::GetFlag(FLAGS_readout_lens)) {
      const TokenTraceSite destination{
          .scope = {"gpt2", absl::StrCat("transformer_block_",
                                         config.transformer_block_count - 1)},
          .layer_name = "ResidualLayer",
          .occurrence = 1};
      ASSIGN_OR_RETURN(const auto* final_residual,
                       FindSite(baseline.trace, destination));
      std::vector<std::pair<TokenTraceSite, std::string>> sources;
      sources.push_back({{.scope = {"gpt2"},
                          .layer_name = "PositionEmbeddingLayer",
                          .occurrence = 0},
                         "readout_lens_embedding"});
      for (int block = 0; block < config.transformer_block_count; ++block)
        for (int occurrence = 0; occurrence < 2; ++occurrence)
          sources.push_back(
              {{.scope = {"gpt2", absl::StrCat("transformer_block_", block)},
                .layer_name = "ResidualLayer",
                .occurrence = occurrence},
               absl::StrCat(
                   "readout_lens_block_", block,
                   occurrence == 0 ? "_post_attention" : "_post_mlp")});
      for (const auto& [source_site, name] : sources) {
        ASSIGN_OR_RETURN(const auto* source,
                         FindSite(baseline.trace, source_site));
        if (source->channels != static_cast<size_t>(config.model_width) ||
            source->channels != final_residual->channels ||
            source->data_type != final_residual->data_type)
          return absl::InternalError(
              "readout-lens residual shape/dtype differs");
        // Overwrite only the last residual's query row. The original final
        // LayerNorm and tied output head then provide the readout, without
        // constructing substitute layers or treating an early state as an
        // actual prediction. Other positions cannot reach this row anymore.
        const TokenTracePatch patch{
            .site = final_residual->site,
            .rows = TokenTraceRows::kQuery,
            .channel_count = static_cast<size_t>(config.model_width),
            .replacement = TokenTraceReplacement::kDonor,
            .donor = source};
        RETURN_IF_ERROR(
            intervene(baseline, patch,
                      {.name = name,
                       .donor_line = baseline.sentence->line,
                       .rows = "query_row_4"},
                      SameSite(source->site, final_residual->site)));
      }
    }
    if (absl::GetFlag(FLAGS_branch_ablations))
      for (int block = 0; block < config.transformer_block_count; ++block)
        for (const char* branch : {"attention", "mlp"}) {
          const TokenTraceSite site{
              .scope = {"gpt2", absl::StrCat("transformer_block_", block),
                        "ResidualLayer"},
              .layer_name = branch,
              .occurrence = 0};
          ASSIGN_OR_RETURN(const auto* activation,
                           FindSite(baseline.trace, site));
          const TokenTracePatch patch{.site = activation->site,
                                      .rows = TokenTraceRows::kAllPrefix};
          RETURN_IF_ERROR(intervene(
              baseline, patch,
              {.name = absl::StrCat("zero_block_", block, "_", branch),
               .rows = "all_prefix"},
              false));
        }
    for (int block : neuron_blocks) {
      const TokenTraceSite site{
          .scope = {"gpt2", absl::StrCat("transformer_block_", block),
                    "ResidualLayer", "mlp"},
          .layer_name = "GeluLayer",
          .occurrence = 0};
      ASSIGN_OR_RETURN(const auto* activation, FindSite(baseline.trace, site));
      if (activation->channels !=
          static_cast<size_t>(config.feed_forward_width))
        return absl::InternalError(
            "GELU width differs from neuron-sweep configuration");
      for (int neuron : neuron_channels) {
        // This is deliberately query-only. Zeroing this channel at preceding
        // positions would mix its local role with later attention transport.
        const TokenTracePatch patch{
            .site = activation->site,
            .rows = TokenTraceRows::kQuery,
            .first_channel = static_cast<size_t>(neuron),
            .channel_count = 1};
        RETURN_IF_ERROR(
            intervene(baseline, patch,
                      {.name = absl::StrCat("zero_mlp_neuron_block_", block,
                                            "_channel_", neuron),
                       .block = block,
                       .neuron = neuron},
                      false));
      }
    }
    const size_t index = baseline.sentence->line - 1;
    if (std::find(capitals.begin(), capitals.end(), index) != capitals.end())
      for (const auto& donor : baselines) {
        const size_t donor_index = donor.sentence->line - 1;
        if (donor_index == index || std::find(capitals.begin(), capitals.end(),
                                              donor_index) == capitals.end())
          continue;
        for (int block = 0; block < config.transformer_block_count; ++block) {
          // A residual output hook is after its own scope exits. Its two
          // occurrences distinguish post-attention and post-MLP within a block.
          const std::string block_name =
              absl::StrCat("transformer_block_", block);
          std::vector<TokenTraceSite> sites;
          for (int occurrence = 0; occurrence < 2; ++occurrence)
            sites.push_back({.scope = {"gpt2", block_name},
                             .layer_name = "ResidualLayer",
                             .occurrence = occurrence});
          sites.push_back(
              {.scope = {"gpt2", block_name, "ResidualLayer", "attention"},
               .layer_name = "FullyConnectedLayer",
               .occurrence = 0});
          for (size_t kind = 0; kind < sites.size(); ++kind) {
            ASSIGN_OR_RETURN(const auto* recipient_activation,
                             FindSite(baseline.trace, sites[kind]));
            ASSIGN_OR_RETURN(const auto* donor_activation,
                             FindSite(donor.trace, sites[kind]));
            const size_t expected_channels =
                static_cast<size_t>(config.model_width) * (kind == 2 ? 3 : 1);
            if (recipient_activation->channels != expected_channels ||
                donor_activation->channels != expected_channels)
              return absl::InternalError(
                  "donor patch site width differs from GPT-2 recipe");
            for (size_t row : {size_t{3}, size_t{4}})
              for (size_t slice = 0; slice < (kind == 2 ? 3u : 1u); ++slice) {
                const char* label = kind == 0    ? "post_attention"
                                    : kind == 1  ? "post_mlp"
                                    : slice == 0 ? "Q"
                                    : slice == 1 ? "K"
                                                 : "V";
                const TokenTracePatch patch{
                    .site = recipient_activation->site,
                    .rows = TokenTraceRows::kOne,
                    .row = row,
                    .first_channel = slice * config.model_width,
                    .channel_count = static_cast<size_t>(config.model_width),
                    .replacement = TokenTraceReplacement::kDonor,
                    .donor = donor_activation};
                // Before block 0 attention, query position 4 is the same
                // token/position in every capital prompt, hence identical QKV.
                // After the final attention, row 3 has no remaining path to
                // query row 4; changing Q3 cannot affect attention output4
                // either. These negative controls catch wrong-row patching.
                const bool structural_identity =
                    (block == 0 && row == 4 && kind == 2) ||
                    (block == config.transformer_block_count - 1 && row == 3 &&
                     (kind != 2 || slice == 0));
                RETURN_IF_ERROR(intervene(
                    baseline, patch,
                    {.name = absl::StrCat("donor_line_", donor.sentence->line,
                                          "_block_", block, "_", label, "_row_",
                                          row),
                     .donor_line = donor.sentence->line,
                     .rows = absl::StrCat("row_", row),
                     .donor_target =
                         Target(*donor.sentence, vocabulary->eos_token_id())},
                    structural_identity));
              }
          }
        }
      }
    RETURN_IF_ERROR(WriteHtml(directory, checkpoint, baselines, conditions,
                              token_text, false));
  }
  RETURN_IF_ERROR(WriteHtml(directory, checkpoint, baselines, conditions,
                            token_text, true));
  manifest << "identity_donor_copy_logits\tbitwise_equal_for_every_selected_"
              "prefix\n";
  if (absl::GetFlag(FLAGS_readout_lens))
    manifest << "readout_lens_final_self_copy_logits\tbitwise_equal_for_every_"
                "selected_prefix\n";
  manifest << "recorded_condition_count\t" << conditions.size()
           << "\nmlp_neuron_ablation_condition_count\t"
           << std::count_if(conditions.begin(), conditions.end(),
                            [](const Condition& condition) {
                              return condition.neuron >= 0;
                            })
           << "\ncomplete\ttrue\n";
  for (std::ofstream* file : {&manifest, &table, &tops}) {
    file->close();
    if (!*file)
      return absl::UnknownError("cannot close token-trace summary files");
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer

int main(int argc, char** argv) {
  if (absl::ParseCommandLine(argc, argv).size() != 1) {
    std::cerr << "unexpected positional arguments\n";
    return 1;
  }
  const auto status = pluto::llm::one_shot_memorizer::Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
