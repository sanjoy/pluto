// A fixed, selected-case donor intervention, not a general semantic decoder.
// The eight cases were frozen before donor outcomes. All 184 cached-tail rows
// must match independent full-model patches bitwise; controls are not optional.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
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
#include "src/llm/experiments/one_shot_memorizer/feature_donor.h"
#include "src/llm/experiments/one_shot_memorizer/feature_subset.h"
#include "src/llm/experiments/one_shot_memorizer/mlp_subset_tail.h"
#include "src/llm/experiments/one_shot_memorizer/token_trace.h"
#include "src/llm/gpt2.h"
#include "src/llm/layer.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "",
          "Exact original 8-layer width16 checkpoint");
ABSL_FLAG(std::string, tokenizer, "", "Base GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt",
          "Original fixed 1,024-fact corpus");
ABSL_FLAG(std::string, output_dir, "", "Fresh directory for donor reports");

namespace pluto::llm::one_shot_memorizer {
namespace {
namespace fs = std::filesystem;
constexpr int kFeatures = 64;
constexpr FeatureSubset kFullSubset = ~FeatureSubset{0};

constexpr FeatureSubset Feature(int index) { return FeatureSubset{1} << index; }

// Fixed named pairs selected from shared minimum-size masks before donor
// outcomes. Expected IDs validate the original corpus/vocabulary; they score
// predictions but NEVER enter inference. Same-target cases are disruption
// controls, not evidence for answer transfer.
struct ProtocolPair {
  int id;
  const char* name;
  FeatureSubset subset;  // Donor channels; other channels follow background.
  size_t a, b;  // One-based corpus lines; both directions are always evaluated.
  // Expected sixth-token compact IDs, checked after encoding.
  int target_a, target_b;
};
constexpr std::array<ProtocolPair, 8> kProtocolPairs{{
    {1, "sun_geological", Feature(34), 43, 110, 1303, 4246},
    {2, "gravity_flavor", Feature(56), 10, 412, 2722, 2296},
    {3, "butter_360", Feature(35), 583, 542, 2245, 2503},
    {4, "babylon_thunder", Feature(7) | Feature(34), 312, 499, 3742, 3160},
    {5, "circle_dna", Feature(34) | Feature(35), 102, 493, 2242, 1995},
    {6, "lima_plugs", Feature(34) | Feature(56), 411, 939, 4235, 4061},
    {7, "same_sun_control", Feature(34), 43, 138, 1303, 1303},
    {8, "same_same_control", Feature(34), 40, 848, 462, 462},
}};

// Keep text fields unambiguous even for partial token bytes or special paths.
std::string EscapeTsv(absl::string_view text) {
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

struct Prediction {
  int winner = 0;  // Stable argmax: the lowest ID wins equal logits.
  int rival = 0;   // Strongest non-target class, with the same tie rule.
  size_t target_rank = 1;
  double target_margin = 0;
};

// Inspect recipient and donor targets independently while retaining the same
// lowest-ID argmax rule as ordinary inference. Softmax is computed separately.
absl::StatusOr<Prediction> InspectLogits(absl::Span<const float> logits,
                                         int target) {
  if (logits.size() < 2 || target < 0 ||
      static_cast<size_t>(target) >= logits.size())
    return absl::InvalidArgumentError("invalid subset-scoring shape or target");
  Prediction result{.rival = target == 0 ? 1 : 0};
  for (size_t token = 0; token < logits.size(); ++token) {
    if (!std::isfinite(logits[token]))
      return absl::DataLossError("nonfinite subset logit");
    if (logits[token] > logits[result.winner])
      result.winner = static_cast<int>(token);
    if (static_cast<int>(token) != target) {
      if (logits[token] > logits[result.rival])
        result.rival = static_cast<int>(token);
      result.target_rank +=
          logits[token] > logits[target] ||
          (logits[token] == logits[target] && static_cast<int>(token) < target);
    }
  }
  result.target_margin =
      static_cast<double>(logits[target]) - logits[result.rival];
  return result;
}

double TargetProbability(absl::Span<const float> logits, int target,
                         int winner) {
  const double maximum = logits[winner];
  double denominator = 0;
  for (float value : logits)
    denominator += std::exp(static_cast<double>(value) - maximum);
  return std::exp(static_cast<double>(logits[target]) - maximum) / denominator;
}

// This is byte-preserving extraction, not a float decode/re-encode. Match the
// full hook identity so identically named layers elsewhere cannot be selected.
absl::Status ExtractBf16Query(const TokenTraceResult& trace,
                              const TokenTraceSite& site,
                              absl::Span<uint16_t> destination) {
  const TokenTraceActivation* found = nullptr;
  for (const auto& activation : trace.activations)
    if (activation.site.scope == site.scope &&
        activation.site.layer_name == site.layer_name &&
        activation.site.occurrence == site.occurrence &&
        activation.site.output_index == site.output_index) {
      if (found != nullptr)
        return absl::FailedPreconditionError(
            "ambiguous cached activation site");
      found = &activation;
    }
  if (found == nullptr || found->data_type != DataType::BF16 ||
      found->channels != destination.size() ||
      found->first_row > trace.query_row ||
      trace.query_row - found->first_row >= found->row_count ||
      found->bytes.size() !=
          found->row_count * destination.size() * sizeof(uint16_t))
    return absl::FailedPreconditionError(
        "cached activation has wrong site/type/shape");
  const size_t offset = (trace.query_row - found->first_row) *
                        destination.size() * sizeof(uint16_t);
  std::memcpy(destination.data(), found->bytes.data() + offset,
              destination.size() * sizeof(uint16_t));
  return absl::OkStatus();
}

bool SameLogits(absl::Span<const float> first, absl::Span<const float> second) {
  return first.size() == second.size() &&
         std::memcmp(first.data(), second.data(),
                     first.size() * sizeof(float)) == 0;
}

using HostBytes = cuda::PageLockedHostArray<uint8_t>;

// Snapshot all exposed tensors, including the tied alias. Kept host storage is
// pinned; every CUDA transfer completes before the snapshots are inspected.
absl::StatusOr<std::vector<HostBytes>> CopyMasterBytes(cuda::Executor& executor,
                                                       const Layer& model) {
  std::vector<HostBytes> result;
  for (const auto& weight : model.weights()) {
    ASSIGN_OR_RETURN(auto host,
                     HostBytes::Allocate(executor, weight.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), weight.data(), host.size_bytes(),
                        cudaMemcpyDeviceToHost, executor.stream()),
        "snapshot model master bytes"));
    result.push_back(std::move(host));
  }
  RETURN_IF_ERROR(executor.Synchronize());
  return result;
}

std::string ChannelList(FeatureSubset subset) {
  std::string text;
  for (int feature = 0; feature < kFeatures; ++feature)
    if (subset & (FeatureSubset{1} << feature))
      absl::StrAppend(&text, text.empty() ? "" : ",", feature);
  return text.empty() ? "-" : text;
}

// Read-only baseline snapshot; interventions always reuse these originals.
struct CachedFact {
  size_t line;
  int target;
  std::vector<int> prefix;
  std::string prefix_text, target_text;
  std::array<uint16_t, 64> gelu;  // Original query-row physical BF16 values.
  std::array<uint16_t, 16> residual;  // Query row before the final MLP.
  std::vector<float> logits;  // Ordinary full-model real-vocabulary logits.
};
// One predeclared row. GELU may be mixed, but residual is always recipient.
// IDs/text are reporting metadata, never inference inputs.
struct Condition {
  std::string phase, background;
  int pair_id;
  size_t recipient, donor;
  FeatureSubset subset;
  std::array<uint16_t, 64>
      physical_gelu;  // Exact bytes used by both forward paths.
};

absl::Status RunDonorProtocol() {
  const auto& pairs = kProtocolPairs;
  const fs::path checkpoint = absl::GetFlag(FLAGS_checkpoint),
                 output = absl::GetFlag(FLAGS_output_dir);
  if (checkpoint.empty() || output.empty() ||
      absl::GetFlag(FLAGS_tokenizer).empty())
    return absl::InvalidArgumentError("checkpoint/tokenizer/output required");
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  ASSIGN_OR_RETURN(auto corpus_lines, ReadCorpusLines(corpus.text()));
  if (corpus_lines.size() != 1024)
    return absl::InvalidArgumentError("expected original 1024-fact corpus");
  ASSIGN_OR_RETURN(
      auto base, tokenizer::Gpt2Tokenizer::Load(absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto detokenizer, tokenizer::Gpt2Detokenizer::Load(
                                         absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto vocabulary,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *base, checkpoint / "compact_vocabulary.tsv"));
  if (vocabulary->vocab_size() != 4475 || vocabulary->eos_token_id() != 4474 ||
      base->eos_token_id() != 50256 ||
      vocabulary->original_eos_token_id() != base->eos_token_id())
    return absl::InvalidArgumentError("unexpected vocabulary");
  std::error_code error;
  if (!fs::create_directory(output, error))
    return absl::InvalidArgumentError("output must be fresh");
  std::ofstream report(output / "conditions.tsv"),
      manifest(output / "manifest.tsv"), facts_output(output / "facts.tsv"),
      amplitudes(output / "amplitudes.tsv"), controls(output / "controls.tsv");
  for (auto* out :
       {&report, &manifest, &facts_output, &amplitudes, &controls}) {
    if (!*out)
      return absl::DataLossError("failed to create report");
    *out << std::setprecision(17);
  }
  manifest << "protocol\t" << "fixed_eight_pairs_v1" << "\ncheckpoint\t"
           << checkpoint.string()
           << "\nselection\tfixed_posthoc_cases_before_donor_"
              "outcomes\ncondition_count_expected\t184\nquery_"
              "row\t4\nweights\tunchanged\ncomplete\tfalse\n";
  for (const auto& pair : kProtocolPairs)
    manifest << "pair_" << pair.id << '\t' << pair.name << ';' << pair.a << ';'
             << pair.b << ';' << ChannelList(pair.subset) << ';'
             << pair.target_a << ';' << pair.target_b << '\n';
  manifest
      << "tokenizer\t" << EscapeTsv(absl::GetFlag(FLAGS_tokenizer))
      << "\ncorpus\t" << EscapeTsv(absl::GetFlag(FLAGS_corpus))
      << "\narchitecture\tlayers8_width16_heads1_ff64_vocabulary4475_BF16"
      << "\nclaim_scope\tselected_case_off_manifold_conditional_intervention\n";
  manifest.flush();
  report
      << "index\tphase\tpair_id\tbackground\trecipient_line\tdonor_line\tkeep_"
         "channels\trecipient_target\tdonor_target\twinner\twinner_"
         "text\trecipient_probability\tdonor_probability\trecipient_"
         "logit\tdonor_logit\trecipient_rank\tdonor_rank\trecipient_"
         "rival\trecipient_rival_text\trecipient_margin\tdonor_rival\tdonor_"
         "margin\trecipient_minus_donor_margin\trecipient_fixed_self_"
         "rival\trecipient_margin_vs_fixed_self_rival\tbitwise_fullmodel_"
         "equal\tbitwise_post_control\toutcome\n";
  facts_output << "line\tprefix_text\ttarget\ttarget_text\n";
  amplitudes << "line\tchannel\tbf16_bits\n";
  controls << "line\tcontrol\tbitwise_equal\n";
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
  ASSIGN_OR_RETURN(auto original_weights, CopyMasterBytes(*executor, *model));
  ASSIGN_OR_RETURN(auto tail,
                   FinalMlpSubsetTail::Create(*executor, *model, config));
  const TokenTraceOptions plain{.vocabulary_size = 4475, .padding_token = 4474};
  TokenTraceOptions capture = plain;
  capture.capture_activations = true;
  const TokenTraceSite gelu_site{
      .scope = {"gpt2", "transformer_block_7", "ResidualLayer", "mlp"},
      .layer_name = "GeluLayer",
      .occurrence = 0};
  const TokenTraceSite residual_site{.scope = {"gpt2", "transformer_block_7"},
                                     .layer_name = "ResidualLayer",
                                     .occurrence = 0};
  std::map<size_t, int> expected_targets;
  for (const auto& p : pairs) {
    for (const auto& [id, target] : std::array<std::pair<size_t, int>, 2>{
             {{p.a, p.target_a}, {p.b, p.target_b}}}) {
      if (expected_targets.contains(id) && expected_targets.at(id) != target)
        return absl::InvalidArgumentError("inconsistent protocol targets");
      expected_targets[id] = target;
    }
  }
  std::map<size_t, CachedFact> facts;
  for (const auto& [id, expected] : expected_targets) {
    ASSIGN_OR_RETURN(auto tokens,
                     vocabulary->Encode(*executor, corpus_lines[id - 1]));
    if (tokens.size() < 6 || tokens.size() > kGpt2ContextLength)
      return absl::InvalidArgumentError(
          "expected five-token prompt plus target");
    CachedFact f{
        .line = id,
        .target = tokens[5],
        .prefix = std::vector<int>(tokens.begin(), tokens.begin() + 5)};
    if (f.target != expected)
      return absl::InvalidArgumentError(
          "protocol target does not match corpus");
    ASSIGN_OR_RETURN(auto baseline,
                     TraceNextToken(*executor, *model, f.prefix, plain));
    ASSIGN_OR_RETURN(auto traced,
                     TraceNextToken(*executor, *model, f.prefix, capture));
    if (!SameLogits(baseline.logits, traced.logits) ||
        baseline.predicted_token != f.target)
      return absl::FailedPreconditionError("baseline/capture mismatch");
    f.logits = std::move(baseline.logits);
    RETURN_IF_ERROR(
        ExtractBf16Query(traced, gelu_site, absl::MakeSpan(f.gelu)));
    RETURN_IF_ERROR(
        ExtractBf16Query(traced, residual_site, absl::MakeSpan(f.residual)));
    std::vector<int> original_prefix;
    for (int t : f.prefix) {
      ASSIGN_OR_RETURN(int original, vocabulary->OriginalId(t));
      original_prefix.push_back(original);
    }
    ASSIGN_OR_RETURN(f.prefix_text, detokenizer->Decode(original_prefix));
    ASSIGN_OR_RETURN(int original_target, vocabulary->OriginalId(f.target));
    ASSIGN_OR_RETURN(f.target_text, detokenizer->Decode({&original_target, 1}));
    facts_output << id << '\t' << EscapeTsv(f.prefix_text) << '\t' << f.target
                 << '\t' << EscapeTsv(f.target_text) << '\n';
    for (int n = 0; n < 64; ++n)
      amplitudes << id << '\t' << n << '\t' << f.gelu[n] << '\n';
    controls << id << "\tcapture_plain\t1\n";
    facts.emplace(id, std::move(f));
  }
  std::vector<Condition> conditions;
  conditions.reserve(184);
  auto add = [&](std::string phase, int pair_id, std::string background,
                 size_t recipient, size_t donor,
                 FeatureSubset subset) -> absl::Status {
    Condition c{std::move(phase),
                std::move(background),
                pair_id,
                recipient,
                donor,
                subset,
                {}};
    ASSIGN_OR_RETURN(auto physical,
                     MakeFeatureDonorRow(facts.at(recipient).gelu,
                                         facts.at(donor).gelu, subset,
                                         c.background == "intact"
                                             ? FeatureDonorBackground::kIntact
                                             : FeatureDonorBackground::kSparse));
    std::copy(physical.begin(), physical.end(), c.physical_gelu.begin());
    conditions.push_back(std::move(c));
    return absl::OkStatus();
  };
  const std::vector<FeatureSubset> common = {
      uint64_t{1} << 34,
      uint64_t{1} << 56,
      uint64_t{1} << 35,
      (uint64_t{1} << 7) | (uint64_t{1} << 34),
      (uint64_t{1} << 34) | (uint64_t{1} << 35),
      (uint64_t{1} << 34) | (uint64_t{1} << 56),
      0,
      kFullSubset};
  for (const auto& [id, f] : facts)
    for (auto mask : common)
      RETURN_IF_ERROR(add("common_panel", 0, "sparse", id, id, mask));
  for (const auto& p : pairs)
    for (const auto& background : {"sparse", "intact"})
      for (size_t recipient : {p.a, p.b})
        for (size_t donor : {p.a, p.b})
          RETURN_IF_ERROR(
              add("donor_swap", p.id, background, recipient, donor, p.subset));
  if (conditions.size() != 184)
    return absl::InternalError("wrong frozen condition count");
  // Every condition and all15-prefix full identity bookends use one batch
  // geometry. The arrays remain alive until Evaluate synchronizes its copy.
  std::vector<FinalMlpSubsetInput> batch;
  for (const auto& [id, f] : facts)
    batch.push_back({f.gelu, f.residual, kFullSubset});
  for (const auto& c : conditions)
    batch.push_back(
        {c.physical_gelu, facts.at(c.recipient).residual, kFullSubset});
  for (const auto& [id, f] : facts)
    batch.push_back({f.gelu, f.residual, kFullSubset});
  ASSIGN_OR_RETURN(auto logits, tail->Evaluate(*executor, batch));
  const int stride = tail->logit_stride();
  auto row = [&](size_t r) {
    return absl::Span<const float>(logits.data() + r * stride, 4475);
  };
  size_t fact_index = 0;
  for (const auto& [id, f] : facts) {
    for (size_t r :
         {fact_index, facts.size() + conditions.size() + fact_index}) {
      if (!SameLogits(row(r), f.logits))
        return absl::FailedPreconditionError(
            "batched full-mask bookend differs");
      controls << id << "\ttail_full_bookend\t1\n";
    }
    ++fact_index;
  }
  std::map<std::pair<size_t, FeatureSubset>, size_t> common_index;
  for (size_t i = 0; i < 120; ++i)
    common_index[{conditions[i].recipient, conditions[i].subset}] = i;
  for (size_t i = 0; i < conditions.size(); ++i) {
    const auto& c = conditions[i];
    const auto& recipient = facts.at(c.recipient);
    const auto& donor = facts.at(c.donor);
    const auto values = row(facts.size() + i);
    // Independent real full-model patch, not another invocation of the cached
    // tail. Replace only the query GELU64 with the exact same physical bytes.
    TokenTraceActivation replacement{.site = gelu_site,
                                     .data_type = DataType::BF16,
                                     .first_row = 4,
                                     .row_count = 1,
                                     .channels = 64};
    replacement.bytes.resize(128);
    std::memcpy(replacement.bytes.data(), c.physical_gelu.data(), 128);
    const TokenTracePatch patch{.site = gelu_site,
                                .rows = TokenTraceRows::kQuery,
                                .channel_count = 64,
                                .replacement = TokenTraceReplacement::kDonor,
                                .donor = &replacement};
    TokenTraceOptions patched = plain;
    patched.patches = absl::Span<const TokenTracePatch>(&patch, 1);
    ASSIGN_OR_RETURN(
        auto independent,
        TraceNextToken(*executor, *model, recipient.prefix, patched));
    if (!SameLogits(values, independent.logits))
      return absl::FailedPreconditionError(
          absl::StrCat("cached/full-model mismatch at condition ", i));
    ASSIGN_OR_RETURN(auto restored,
                     TraceNextToken(*executor, *model, recipient.prefix, plain));
    if (!SameLogits(restored.logits, recipient.logits))
      return absl::FailedPreconditionError(
          "post-patch ordinary output differs");
    if (c.recipient == c.donor) {
      const auto expected =
          c.background == "intact"
              ? absl::MakeConstSpan(recipient.logits)
              : row(facts.size() + common_index.at({c.recipient, c.subset}));
      if (!SameLogits(values, expected))
        return absl::FailedPreconditionError("self donor condition differs");
    }
    ASSIGN_OR_RETURN(auto recipient_score,
                     InspectLogits(values, recipient.target));
    ASSIGN_OR_RETURN(auto donor_score, InspectLogits(values, donor.target));
    const auto self_values =
        c.background == "intact"
            ? absl::MakeConstSpan(recipient.logits)
            : row(facts.size() + common_index.at({c.recipient, c.subset}));
    ASSIGN_OR_RETURN(auto self_score,
                     InspectLogits(self_values, recipient.target));
    ASSIGN_OR_RETURN(int winner_original,
                     vocabulary->OriginalId(recipient_score.winner));
    ASSIGN_OR_RETURN(auto winner_text,
                     detokenizer->Decode({&winner_original, 1}));
    ASSIGN_OR_RETURN(int rival_original,
                     vocabulary->OriginalId(recipient_score.rival));
    ASSIGN_OR_RETURN(auto rival_text, detokenizer->Decode({&rival_original, 1}));
    const std::string outcome =
        recipient.target == donor.target
            ? (recipient_score.winner == recipient.target
                   ? "same_target_preserved"
                   : "same_target_disrupted")
        : recipient_score.winner == recipient.target ? "recipient_preserved"
        : recipient_score.winner == donor.target     ? "donor_wins"
                                                     : "third_token";
    report << i << '\t' << c.phase << '\t' << c.pair_id << '\t' << c.background
           << '\t' << c.recipient << '\t' << c.donor << '\t'
           << ChannelList(c.subset) << '\t' << recipient.target << '\t'
           << donor.target << '\t' << recipient_score.winner << '\t'
           << EscapeTsv(winner_text) << '\t'
           << TargetProbability(values, recipient.target,
                                recipient_score.winner)
           << '\t'
           << TargetProbability(values, donor.target, donor_score.winner)
           << '\t' << values[recipient.target] << '\t' << values[donor.target]
           << '\t' << recipient_score.target_rank << '\t'
           << donor_score.target_rank << '\t' << recipient_score.rival << '\t'
           << EscapeTsv(rival_text) << '\t' << recipient_score.target_margin
           << '\t' << donor_score.rival << '\t' << donor_score.target_margin
           << '\t' << double(values[recipient.target]) - values[donor.target]
           << '\t' << self_score.rival << '\t'
           << double(values[recipient.target]) - values[self_score.rival]
           << "\t1\t1\t" << outcome << '\n';
    if (i % 16 == 0) {
      report.flush();
      std::cout << "validated_conditions=" << i + 1 << "/184" << std::endl;
    }
  }
  ASSIGN_OR_RETURN(auto final_weights, CopyMasterBytes(*executor, *model));
  if (final_weights.size() != original_weights.size())
    return absl::InternalError("weight inventory changed");
  for (size_t i = 0; i < original_weights.size(); ++i)
    if (original_weights[i].size() != final_weights[i].size() ||
        std::memcmp(original_weights[i].data(), final_weights[i].data(),
                    original_weights[i].size()) != 0)
      return absl::FailedPreconditionError("model weights changed");
  manifest << "case_count\t15\ncommon_panel_conditions\t120\ndonor_"
              "conditions\t64\nfullmodel_all_logits_bitwise\t184\npost_patch_"
              "plain_bitwise\t184\ntail_fullmask_bookends\t30\nsource_master_"
              "bytes_unchanged\ttrue\ncomplete\ttrue\n";
  for (auto* out :
       {&report, &manifest, &facts_output, &amplitudes, &controls}) {
    out->close();
    if (!*out)
      return absl::DataLossError("failed closing report");
  }
  return absl::OkStatus();
}
}  // namespace
}  // namespace pluto::llm::one_shot_memorizer
int main(int argc, char** argv) {
  if (absl::ParseCommandLine(argc, argv).size() != 1)
    return 2;
  const auto status = pluto::llm::one_shot_memorizer::RunDonorProtocol();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
