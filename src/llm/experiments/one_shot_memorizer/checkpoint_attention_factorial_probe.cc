// Fixed prefix-only attention routing/value factorial for three capital
// prompts. Uses unchanged production FlashAttention through ordinary
// token-trace patches. Target IDs score outputs only. This is a selected-case
// intervention, not a claim of general semantic ownership or a search for a
// sufficient circuit.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
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
#include "src/llm/experiments/one_shot_memorizer/attention_factorial.h"
#include "src/llm/experiments/one_shot_memorizer/feature_capture.h"
#include "src/llm/experiments/one_shot_memorizer/sentence_ablation.h"
#include "src/llm/experiments/one_shot_memorizer/token_trace.h"
#include "src/llm/gpt2.h"
#include "src/llm/layer.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "",
          "Exact original 8-layer width16 checkpoint");
ABSL_FLAG(std::string, tokenizer, "", "Base GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt",
          "Original fixed 1,024-fact corpus");
ABSL_FLAG(std::string, output_dir, "",
          "Fresh directory for attention factorial reports");
namespace pluto::llm::one_shot_memorizer {
namespace {

namespace fs = std::filesystem;

// Escape arbitrary token bytes without losing tab-separated report structure.
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
absl::StatusOr<std::vector<std::string>> ReadCorpusLines(
    absl::string_view text) {
  std::vector<std::string> lines;
  while (!text.empty()) {
    const size_t end = text.find('\n');
    auto line = text.substr(0, end);
    if (!line.empty() && line.back() == '\r')
      line.remove_suffix(1);
    if (line.empty() || line.find_first_not_of(" \t\r") == line.npos)
      return absl::InvalidArgumentError("empty or blank corpus line");
    lines.emplace_back(line);
    if (end == text.npos)
      break;
    text.remove_prefix(end + 1);
  }
  return lines;
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
bool SameSite(const TokenTraceSite& a, const TokenTraceSite& b) {
  return a.scope == b.scope && a.layer_name == b.layer_name &&
         a.occurrence == b.occurrence && a.output_index == b.output_index;
}
absl::StatusOr<const TokenTraceActivation*> FindSite(
    const TokenTraceResult& trace, const TokenTraceSite& site) {
  const TokenTraceActivation* found = nullptr;
  for (const auto& a : trace.activations)
    if (SameSite(a.site, site)) {
      if (found != nullptr)
        return absl::InternalError("duplicate trace site");
      found = &a;
    }
  if (found == nullptr)
    return absl::NotFoundError("missing trace activation");
  return found;
}
// Score against every real vocabulary rival, never only the two capital labels.
struct Prediction {
  double target_probability = 0;
  double target_margin = 0;
};
absl::StatusOr<Prediction> Score(absl::Span<const float> logits, int target) {
  if (logits.size() < 2 || target < 0 ||
      static_cast<size_t>(target) >= logits.size())
    return absl::InvalidArgumentError("invalid scoring shape or target");
  for (float value : logits)
    if (!std::isfinite(value))
      return absl::DataLossError("nonfinite logit");
  const double maximum = *std::max_element(logits.begin(), logits.end());
  double denominator = 0, rival = -std::numeric_limits<double>::infinity();
  for (size_t token = 0; token < logits.size(); ++token) {
    denominator += std::exp(static_cast<double>(logits[token]) - maximum);
    if (static_cast<int>(token) != target)
      rival = std::max(rival, static_cast<double>(logits[token]));
  }
  return Prediction{std::exp(logits[target] - maximum) / denominator,
                    static_cast<double>(logits[target]) - rival};
}
using Row = std::array<double, 16>;
// Prefix IDs alone enter inference. The target only validates and scores it.
struct CapitalCase {
  size_t line;  // One-based original corpus line.
  int target;   // Expected sixth-token compact ID, never a model input.
  std::vector<int> prefix;  // Exactly five observed input token IDs.
  TokenTraceResult trace;   // Unmodified captured baseline, owning donor bytes.
};
TokenTraceSite AttnSite(int block, std::string leaf, int occurrence = 0) {
  return {.scope = {"gpt2", absl::StrCat("transformer_block_", block),
                    "ResidualLayer", "attention"},
          .layer_name = std::move(leaf),
          .occurrence = occurrence};
}
TokenTraceSite ResidualSite(int block) {
  return {.scope = {"gpt2", absl::StrCat("transformer_block_", block)},
          .layer_name = "ResidualLayer",
          .occurrence = 0};
}
bool FloatBytesEqual(absl::Span<const float> a, absl::Span<const float> b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
absl::StatusOr<const TokenTraceAttention*> FindAttention(
    const TokenTraceResult& result, int block) {
  const auto site = AttnSite(block, "AttentionLayer");
  for (const auto& a : result.attention)
    if (SameSite(a.site, site)) {
      if (a.heads != 1 || a.prefix_length != 5 || a.probabilities.size() != 25)
        return absl::InternalError("unexpected probability shape");
      return &a;
    }
  return absl::NotFoundError("attention not captured");
}
absl::StatusOr<Row> QueryRow(const TokenTraceResult& t,
                             const TokenTraceSite& site) {
  ASSIGN_OR_RETURN(auto a, FindSite(t, site));
  if (a->data_type != DataType::BF16 || a->channels != 16 ||
      a->first_row != 0 || a->row_count != 5 || a->values.size() != 80 ||
      a->bytes.size() != 160)
    return absl::InternalError("unexpected query activation shape");
  Row r;
  for (int i = 0; i < 16; ++i)
    r[i] = a->values[4 * 16 + i];
  return r;
}
double Norm(absl::Span<const double> r) {
  double x = 0;
  for (double v : r)
    x += v * v;
  return std::sqrt(x);
}
Row Project(absl::Span<const double> r, const std::vector<float>& w) {
  Row y{};
  for (int i = 0; i < 16; ++i)
    for (int j = 0; j < 16; ++j)
      y[j] += r[i] * w[i * 16 + j];
  return y;
}
absl::StatusOr<std::vector<std::vector<uint8_t>>> SnapshotWeights(
    cuda::Executor& e, const Layer& m) {
  std::vector<std::vector<uint8_t>> result;
  for (const auto& w : m.weights()) {
    ASSIGN_OR_RETURN(
        auto h, cuda::PageLockedHostArray<uint8_t>::Allocate(e, w.size_bytes()));
    RETURN_IF_ERROR(
        cuda::CudaStatus(cudaMemcpyAsync(h.data(), w.data(), w.size_bytes(),
                                         cudaMemcpyDeviceToHost, e.stream()),
                         "copy weight control"));
    RETURN_IF_ERROR(e.Synchronize());
    result.emplace_back(h.begin(), h.end());
  }
  return result;
}
absl::Status RunFactorial() {
  const fs::path checkpoint = absl::GetFlag(FLAGS_checkpoint),
                 out = absl::GetFlag(FLAGS_output_dir);
  if (checkpoint.empty() || out.empty() ||
      absl::GetFlag(FLAGS_tokenizer).empty())
    return absl::InvalidArgumentError("paths required");
  std::error_code error;
  if (!fs::create_directory(out, error))
    return absl::InvalidArgumentError("output directory must be fresh");
  std::ofstream conditions(out / "conditions.tsv"),
      probabilities(out / "probabilities.tsv"), vectors(out / "vectors.tsv"),
      decomposition(out / "decomposition.tsv"),
      summary(out / "decomposition_summary.tsv"),
      controls(out / "controls.tsv"), manifest(out / "manifest.tsv"),
      tokens(out / "tokens.tsv");
  std::ofstream logits_file(out / "query_logits.f32", std::ios::binary);
  static_assert(std::endian::native == std::endian::little &&
                sizeof(float) == 4);
  if (!logits_file)
    return absl::DataLossError("cannot write logits report");
  for (auto* f : {&conditions, &probabilities, &vectors, &decomposition,
                  &summary, &controls, &manifest, &tokens}) {
    if (!*f)
      return absl::DataLossError("cannot write report");
    *f << std::setprecision(17);
  }
  manifest << "checkpoint\t" << DisplayBytes(checkpoint.string())
           << "\ntokenizer\t" << DisplayBytes(absl::GetFlag(FLAGS_tokenizer))
           << "\ncorpus\t" << DisplayBytes(absl::GetFlag(FLAGS_corpus))
           << "\nprotocol\tthree_fixed_capital_prefixes_eight_blocks_six_"
              "ordered_pairs_four_corners\nintervention\twhole_QK_or_V_family_"
              "all_five_prefix_rows\ninput\tfirst_five_tokens_only_no_future_"
              "target\ncomplete\tfalse\n";
  manifest.flush();
  conditions
      << "recipient\tdonor\tblock\tcorner\twinner\twinner_text\trecipient_"
         "target\tdonor_target\trecipient_probability\tdonor_"
         "probability\trecipient_margin\tdonor_margin\toutcome\n";
  probabilities << "recipient\tdonor\tblock\tcorner\tkey\tprobability\n";
  vectors << "recipient\tdonor\tblock\tcorner\tsite\tchannel\tvalue\n";
  decomposition
      << "recipient\tdonor\tblock\tchannel\ttotal\trouting\tvalues\tinteraction"
         "\tideal_routing\tideal_values\tideal_interaction\tideal_shared_"
         "prefix_routing\tideal_closure_rounding\n";
  summary << "recipient\tdonor\tblock\ttotal_norm\trouting_norm\tvalues_"
             "norm\tinteraction_norm\trouting_over_total\tvalues_over_"
             "total\tinteraction_over_total\tprojected_total_norm\tprojected_"
             "routing_norm\tprojected_values_norm\tprojected_interaction_"
             "norm\tideal_shared_prefix_routing_norm\tideal_closure_rounding_"
             "norm\n";
  controls << "recipient\tdonor\tblock\tcorner\tcontrol\tbitwise_equal\n";
  tokens << "line\tposition\tcompact_id\ttext\n";
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  ASSIGN_OR_RETURN(auto lines, ReadCorpusLines(corpus.text()));
  if (lines.size() != 1024)
    return absl::InvalidArgumentError("expected original corpus");
  ASSIGN_OR_RETURN(
      auto base, tokenizer::Gpt2Tokenizer::Load(absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto detok, tokenizer::Gpt2Detokenizer::Load(
                                   absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto vocab,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *base, checkpoint / "compact_vocabulary.tsv"));
  if (vocab->vocab_size() != 4475 || vocab->eos_token_id() != 4474 ||
      vocab->original_eos_token_id() != base->eos_token_id() ||
      detok->vocab_size() != base->vocab_size() ||
      detok->eos_token_id() != base->eos_token_id())
    return absl::InvalidArgumentError("vocabulary mismatch");
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  const Gpt2Config config{.transformer_block_count = 8,
                          .model_width = 16,
                          .attention_heads = 1,
                          .feed_forward_width = 64,
                          .vocabulary_size = 4475,
                          .pad_vocabulary = false};
  ASSIGN_OR_RETURN(auto model, CreateGpt2(*executor, DataType::BF16, 0, config));
  RETURN_IF_ERROR(
      ReadFromDirectory(*executor, *model, checkpoint.string(), false));
  ASSIGN_OR_RETURN(auto original_weights, SnapshotWeights(*executor, *model));
  ASSIGN_OR_RETURN(auto layout, BuildGpt2ParameterLayout(
                                    {.vocabulary_size = 4475,
                                     .model_width = 16,
                                     .feed_forward_width = 64,
                                     .context_length = kGpt2ContextLength,
                                     .transformer_block_count = 8}));
  std::array<std::vector<float>, 8> projection;
  for (int b = 0; b < 8; ++b) {
    size_t index = 0;
    const std::string name =
        absl::StrCat("transformer_block_", b, ".attention.output.weight");
    while (index < layout.size() && layout[index].name != name)
      ++index;
    if (index == layout.size() || layout[index].element_count != 256)
      return absl::InternalError("missing projection");
    ASSIGN_OR_RETURN(
        projection[b],
        CopyEffectiveBf16Readout(*executor, model->weights()[index], 16, 16));
  }
  const TokenTraceOptions plain{.vocabulary_size = 4475, .padding_token = 4474};
  TokenTraceOptions capture = plain;
  capture.capture_activations = true;
  capture.capture_attention = true;
  // Include ordinary replays and the master snapshot in the control inventory.
  constexpr std::array<std::pair<absl::string_view, size_t>, 8>
      expected_controls{{{"same_source_QKV", 24},
                         {"probability_source", 192},
                         {"master_weights_unchanged", 1},
                         {"capture_vs_plain", 3},
                         {"full_QKV_donor_attention_bytes", 48},
                         {"ordinary_post", 192},
                         {"ordinary_corner", 48},
                         {"shared_first_three_QKV", 48}}};
  std::array<size_t, 8> control_counts{};
  auto check = [&](size_t r, size_t d, int b, int c, absl::string_view what,
                   bool equal) -> absl::Status {
    size_t control_index = 0;
    while (control_index < expected_controls.size() &&
           expected_controls[control_index].first != what)
      ++control_index;
    if (control_index == expected_controls.size())
      return absl::InternalError("unregistered control");
    ++control_counts[control_index];
    controls << r << '\t' << d << '\t' << b << '\t' << c << '\t' << what << '\t'
             << equal << '\n';
    if (!equal)
      return absl::FailedPreconditionError(absl::StrCat(
          "failed ", what, " r", r, " d", d, " block", b, " corner", c));
    return absl::OkStatus();
  };
  std::vector<CapitalCase> cases;
  for (const auto& [line, target] : std::array<std::pair<size_t, int>, 3>{
           {{80, 1806}, {406, 3417}, {411, 4235}}}) {
    ASSIGN_OR_RETURN(auto ids, vocab->Encode(*executor, lines[line - 1]));
    if (ids.size() < 6 || ids[5] != target)
      return absl::InvalidArgumentError("fixed capital target mismatch");
    CapitalCase c{.line = line,
                  .target = target,
                  .prefix = std::vector<int>(ids.begin(), ids.begin() + 5)};
    const std::array<int, 5> expected_prefix{216, 1145, 82,
                                             line == 80    ? 1516
                                             : line == 406 ? 2373
                                                           : 3626,
                                             113};
    if (!std::equal(c.prefix.begin(), c.prefix.end(), expected_prefix.begin()))
      return absl::InvalidArgumentError("fixed capital prefix IDs differ");
    ASSIGN_OR_RETURN(c.trace,
                     TraceNextToken(*executor, *model, c.prefix, capture));
    ASSIGN_OR_RETURN(auto ordinary,
                     TraceNextToken(*executor, *model, c.prefix, plain));
    RETURN_IF_ERROR(check(line, line, -1, 0, "capture_vs_plain",
                          FloatBytesEqual(ordinary.logits, c.trace.logits)));
    if (c.trace.predicted_token != target)
      return absl::FailedPreconditionError("baseline wrong");
    for (int p = 0; p < 6; ++p) {
      const int id = ids[p];
      ASSIGN_OR_RETURN(auto text,
                       Decode(absl::Span<const int>(&id, 1), *vocab, *detok));
      tokens << line << '\t' << p << '\t' << id << '\t' << DisplayBytes(text)
             << '\n';
    }
    cases.push_back(std::move(c));
  }
  for (int b = 0; b < 8; ++b)
    for (const auto& r : cases) {
      ASSIGN_OR_RETURN(auto source,
                       FindSite(r.trace, AttnSite(b, "FullyConnectedLayer")));
      TokenTracePatch identity{.site = source->site,
                               .rows = TokenTraceRows::kAllPrefix,
                               .replacement = TokenTraceReplacement::kDonor,
                               .donor = source};
      auto opts = capture;
      opts.patches = absl::Span<const TokenTracePatch>(&identity, 1);
      ASSIGN_OR_RETURN(auto result,
                       TraceNextToken(*executor, *model, r.prefix, opts));
      RETURN_IF_ERROR(check(r.line, r.line, b, 3, "same_source_QKV",
                            FloatBytesEqual(result.logits, r.trace.logits)));
    }
  size_t count = 0;
  for (int b = 0; b < 8; ++b)
    for (const auto& r : cases)
      for (const auto& d : cases)
        if (r.line != d.line) {
          ASSIGN_OR_RETURN(
              auto rqkv, FindSite(r.trace, AttnSite(b, "FullyConnectedLayer")));
          ASSIGN_OR_RETURN(
              auto dqkv, FindSite(d.trace, AttnSite(b, "FullyConnectedLayer")));
          if (rqkv->channels != 48 || dqkv->channels != 48 ||
              rqkv->data_type != DataType::BF16 ||
              dqkv->data_type != DataType::BF16 || rqkv->first_row != 0 ||
              dqkv->first_row != 0 || rqkv->row_count != 5 ||
              dqkv->row_count != 5 || rqkv->values.size() != 240 ||
              dqkv->values.size() != 240 || rqkv->bytes.size() != 5 * 48 * 2 ||
              dqkv->bytes.size() != 5 * 48 * 2)
            return absl::InternalError("QKV shape mismatch");
          RETURN_IF_ERROR(check(
              r.line, d.line, b, 0, "shared_first_three_QKV",
              std::memcmp(rqkv->bytes.data(), dqkv->bytes.data(), 3 * 48 * 2) ==
                  0));
          ASSIGN_OR_RETURN(auto rp, FindAttention(r.trace, b));
          ASSIGN_OR_RETURN(auto dp, FindAttention(d.trace, b));
          std::array<Row, 4> corner_rows;
          // The entire causal prefix is patched: earlier rows can change too
          // and be consumed by later blocks. This is NOT a query-only edge
          // intervention. Bits: zero swaps the whole Q/K family; one swaps the
          // whole V family.
          for (int corner = 0; corner < 4; ++corner) {
            std::vector<TokenTracePatch> patches;
            if (corner & 1)
              patches.push_back({.site = dqkv->site,
                                 .rows = TokenTraceRows::kAllPrefix,
                                 .first_channel = 0,
                                 .channel_count = 32,
                                 .replacement = TokenTraceReplacement::kDonor,
                                 .donor = dqkv});
            if (corner & 2)
              patches.push_back({.site = dqkv->site,
                                 .rows = TokenTraceRows::kAllPrefix,
                                 .first_channel = 32,
                                 .channel_count = 16,
                                 .replacement = TokenTraceReplacement::kDonor,
                                 .donor = dqkv});
            auto opts = capture;
            opts.patches = patches;
            ASSIGN_OR_RETURN(auto result,
                             TraceNextToken(*executor, *model, r.prefix, opts));
            ASSIGN_OR_RETURN(auto p, FindAttention(result, b));
            RETURN_IF_ERROR(check(
                r.line, d.line, b, corner, "probability_source",
                FloatBytesEqual(p->probabilities, (corner & 1)
                                                      ? dp->probabilities
                                                      : rp->probabilities)));
            if (corner == 0)
              RETURN_IF_ERROR(
                  check(r.line, d.line, b, corner, "ordinary_corner",
                        FloatBytesEqual(result.logits, r.trace.logits)));
            if (corner == 3) {
              ASSIGN_OR_RETURN(auto a,
                               FindSite(result, AttnSite(b, "AttentionLayer")));
              ASSIGN_OR_RETURN(auto da,
                               FindSite(d.trace, AttnSite(b, "AttentionLayer")));
              RETURN_IF_ERROR(check(r.line, d.line, b, corner,
                                    "full_QKV_donor_attention_bytes",
                                    a->bytes == da->bytes));
            }
            ASSIGN_OR_RETURN(corner_rows[corner],
                             QueryRow(result, AttnSite(b, "AttentionLayer")));
            ASSIGN_OR_RETURN(auto residual, QueryRow(result, ResidualSite(b)));
            ASSIGN_OR_RETURN(
                auto projected,
                QueryRow(result, AttnSite(b, "FullyConnectedLayer", 1)));
            logits_file.write(
                reinterpret_cast<const char*>(result.logits.data()),
                result.logits.size() * sizeof(float));
            ASSIGN_OR_RETURN(auto rs, Score(result.logits, r.target));
            ASSIGN_OR_RETURN(auto ds, Score(result.logits, d.target));
            const int winner = result.predicted_token;
            ASSIGN_OR_RETURN(
                auto winner_text,
                Decode(absl::Span<const int>(&winner, 1), *vocab, *detok));
            conditions << r.line << '\t' << d.line << '\t' << b << '\t'
                       << corner << '\t' << winner << '\t'
                       << DisplayBytes(winner_text) << '\t' << r.target << '\t'
                       << d.target << '\t' << rs.target_probability << '\t'
                       << ds.target_probability << '\t' << rs.target_margin
                       << '\t' << ds.target_margin << '\t'
                       << (winner == r.target   ? "recipient"
                           : winner == d.target ? "donor"
                                                : "third")
                       << '\n';
            for (int k = 0; k < 5; ++k)
              probabilities << r.line << '\t' << d.line << '\t' << b << '\t'
                            << corner << '\t' << k << '\t'
                            << p->probabilities[20 + k] << '\n';
            for (int c = 0; c < 16; ++c) {
              vectors << r.line << '\t' << d.line << '\t' << b << '\t' << corner
                      << "\tattention\t" << c << '\t' << corner_rows[corner][c]
                      << '\n';
              vectors << r.line << '\t' << d.line << '\t' << b << '\t' << corner
                      << "\tprojection\t" << c << '\t' << projected[c] << '\n';
              vectors << r.line << '\t' << d.line << '\t' << b << '\t' << corner
                      << "\tresidual\t" << c << '\t' << residual[c] << '\n';
            }
            ASSIGN_OR_RETURN(auto post,
                             TraceNextToken(*executor, *model, r.prefix, plain));
            RETURN_IF_ERROR(
                check(r.line, d.line, b, corner, "ordinary_post",
                      FloatBytesEqual(post.logits, r.trace.logits)));
            ++count;
          }
          ASSIGN_OR_RETURN(auto observed, DecomposeAttentionFactorial(
                                              {corner_rows[0], corner_rows[1],
                                               corner_rows[2], corner_rows[3]}));
          std::array<float, 80> recipient_values, donor_values;
          for (int key = 0; key < 5; ++key)
            for (int c = 0; c < 16; ++c) {
              recipient_values[key * 16 + c] = rqkv->values[key * 48 + 32 + c];
              donor_values[key * 16 + c] = dqkv->values[key * 48 + 32 + c];
            }
          ASSIGN_OR_RETURN(
              auto ideal,
              DecomposeIdealAttention(
                  absl::MakeConstSpan(rp->probabilities).subspan(20, 5),
                  absl::MakeConstSpan(dp->probabilities).subspan(20, 5),
                  recipient_values, donor_values, 16, 3));
          const auto& total = observed.total;
          const auto& routing = observed.routing;
          const auto& values = observed.values;
          const auto& interaction = observed.interaction;
          const auto& ir = ideal.routing;
          const auto& iv = ideal.values;
          const auto& ii = ideal.interaction;
          const auto& shared = ideal.shared_prefix_routing;
          Row rounding{};
          for (int c = 0; c < 16; ++c) {
            rounding[c] = total[c] - ir[c] - iv[c] - ii[c];
            decomposition << r.line << '\t' << d.line << '\t' << b << '\t' << c
                          << '\t' << total[c] << '\t' << routing[c] << '\t'
                          << values[c] << '\t' << interaction[c] << '\t'
                          << ir[c] << '\t' << iv[c] << '\t' << ii[c] << '\t'
                          << shared[c] << '\t' << rounding[c] << '\n';
          }
          // Ratios are not additive shares. Projected components below are
          // ideal linear projections; actual rounded projection outputs are in
          // vectors.tsv.
          const double norm = Norm(total);
          auto ratio = [&](double component) -> std::string {
            if (norm == 0)
              return "undefined";
            std::ostringstream text;
            text << std::setprecision(17) << component / norm;
            return text.str();
          };
          summary << r.line << '\t' << d.line << '\t' << b << '\t' << norm
                  << '\t' << Norm(routing) << '\t' << Norm(values) << '\t'
                  << Norm(interaction) << '\t' << ratio(Norm(routing)) << '\t'
                  << ratio(Norm(values)) << '\t' << ratio(Norm(interaction))
                  << '\t' << Norm(Project(total, projection[b])) << '\t'
                  << Norm(Project(routing, projection[b])) << '\t'
                  << Norm(Project(values, projection[b])) << '\t'
                  << Norm(Project(interaction, projection[b])) << '\t'
                  << Norm(shared) << '\t' << Norm(rounding) << '\n';
          std::cout << "block " << b << " recipient " << r.line << " donor "
                    << d.line << " done\n";
        }
  ASSIGN_OR_RETURN(auto final_weights, SnapshotWeights(*executor, *model));
  RETURN_IF_ERROR(check(0, 0, -1, 0, "master_weights_unchanged",
                        original_weights == final_weights));
  if (count != 192)
    return absl::InternalError("unexpected factorial condition count");
  size_t total_controls = 0;
  for (size_t i = 0; i < expected_controls.size(); ++i) {
    if (control_counts[i] != expected_controls[i].second)
      return absl::InternalError("unexpected factorial control count");
    manifest << "control_" << expected_controls[i].first << '\t'
             << control_counts[i] << '\n';
    total_controls += control_counts[i];
  }
  manifest << "corners\t0=recipient_QK_recipient_V,1=donor_QK_recipient_V,2="
              "recipient_QK_donor_V,3=donor_QK_donor_V\nconditions\t"
           << count << "\ncontrols\t" << total_controls
           << "\nquery_logits_format\tlittle_endian_float32_row_major_192_by_"
              "4475_in_conditions_tsv_order"
           << "\nratio_convention\tnorms_are_not_additive_shares_zero_total_is_"
              "undefined"
           << "\nprojection_convention\tmathematical_linear_projection_of_"
              "observed_differences_not_a_separate_GPU_intervention"
           << "\nweights\tunchanged\n";
  logits_file.flush();
  if (!logits_file)
    return absl::DataLossError("logit write failed");
  for (auto* f : {&conditions, &probabilities, &vectors, &decomposition,
                  &summary, &controls, &manifest, &tokens}) {
    f->flush();
    if (!*f)
      return absl::DataLossError("report write failed");
  }
  // Do not publish a successful completion marker until all report writes
  // have succeeded, including the complete real-vocabulary logit matrix.
  manifest << "complete\ttrue\n";
  manifest.flush();
  if (!manifest)
    return absl::DataLossError("manifest write failed");
  return absl::OkStatus();
}
}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
int main(int argc, char** argv) {
  const auto arguments = absl::ParseCommandLine(argc, argv);
  if (arguments.size() != 1) {
    std::cerr << "INVALID_ARGUMENT: unexpected positional arguments\n";
    return 1;
  }
  auto status = pluto::llm::one_shot_memorizer::RunFactorial();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
