// Fixed three-capital compatibility experiment. Donate the query residual
// after block 6, then independently donate country K/V in block 7. Labels only
// score outputs; forward receives exactly five prefix tokens and EOS padding.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_detokenizer.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/one_shot_memorizer/token_trace.h"
#include "src/llm/gpt2.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "", "Original width-16/depth-8 checkpoint");
ABSL_FLAG(std::string, tokenizer, "", "Original GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt",
          "Original corpus; fixed lines 80/406/411");
ABSL_FLAG(std::string, output_dir, "", "Fresh output directory");

namespace pluto::llm::one_shot_memorizer {
namespace {
constexpr int kVocabulary = 4475;
constexpr size_t kCountry = 3;
constexpr size_t kQuery = 4;

TokenTraceSite BoundarySite() {
  return {.scope = {"gpt2", "transformer_block_6"},
          .layer_name = "ResidualLayer",
          .occurrence = 1};
}
TokenTraceSite AttentionSite(absl::string_view leaf) {
  return {
      .scope = {"gpt2", "transformer_block_7", "ResidualLayer", "attention"},
      .layer_name = std::string(leaf),
      .occurrence = 0};
}
TokenTraceSite MlpSite(absl::string_view leaf) {
  return {.scope = {"gpt2", "transformer_block_7", "ResidualLayer", "mlp"},
          .layer_name = std::string(leaf),
          .occurrence = 0};
}
bool SameSite(const TokenTraceSite& a, const TokenTraceSite& b) {
  return a.scope == b.scope && a.layer_name == b.layer_name &&
         a.occurrence == b.occurrence && a.output_index == b.output_index;
}
std::string SiteName(const TokenTraceSite& site) {
  std::string result;
  for (const auto& scope : site.scope)
    absl::StrAppend(&result, scope, "/");
  return absl::StrCat(result, site.layer_name, "#", site.occurrence, ":output_",
                      site.output_index);
}
std::string Escape(absl::string_view text) {
  const char* hex = "0123456789abcdef";
  std::string result;
  for (unsigned char c : text)
    if (c == '\\')
      result += "\\\\";
    else if (c < 32 || c >= 127) {
      result += "\\x";
      result += hex[c >> 4];
      result += hex[c & 15];
    } else
      result += static_cast<char>(c);
  return result;
}
absl::StatusOr<const TokenTraceActivation*> Activation(
    const TokenTraceResult& result, const TokenTraceSite& site, size_t width) {
  const TokenTraceActivation* found = nullptr;
  for (const auto& value : result.activations)
    if (SameSite(value.site, site)) {
      if (found != nullptr)
        return absl::DataLossError("duplicate compatibility trace site");
      found = &value;
    }
  if (found == nullptr || found->data_type != DataType::BF16 ||
      found->first_row != 0 || found->row_count != 5 ||
      found->channels != width || found->bytes.size() != 5 * width * 2 ||
      found->values.size() != 5 * width)
    return absl::DataLossError("missing/malformed compatibility activation");
  return found;
}
absl::StatusOr<const TokenTraceAttention*> Probabilities(
    const TokenTraceResult& result) {
  const TokenTraceAttention* found = nullptr;
  for (const auto& value : result.attention)
    if (SameSite(value.site, AttentionSite("AttentionLayer"))) {
      if (found != nullptr)
        return absl::DataLossError("duplicate compatibility probabilities");
      found = &value;
    }
  if (found == nullptr || found->heads != 1 || found->prefix_length != 5 ||
      found->probabilities.size() != 25)
    return absl::DataLossError("missing/malformed compatibility probabilities");
  for (size_t q = 0; q < 5; ++q) {
    double sum = 0;
    for (size_t k = 0; k < 5; ++k) {
      const float p = found->probabilities[q * 5 + k];
      if (!std::isfinite(p) || p < 0 || p > 1 || (k > q && p != 0))
        return absl::DataLossError("invalid causal attention probability");
      sum += p;
    }
    if (std::abs(sum - 1) > 1e-5)
      return absl::DataLossError("attention row is not normalized");
  }
  return found;
}
bool EqualFloats(absl::Span<const float> a, absl::Span<const float> b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
bool EqualRow(const TokenTraceActivation& a, const TokenTraceActivation& b,
              size_t row, size_t first_channel, size_t channels) {
  if (a.channels != b.channels || row >= a.row_count || row >= b.row_count ||
      first_channel > a.channels || channels > a.channels - first_channel)
    return false;
  const size_t offset = (row * a.channels + first_channel) * 2;
  return std::memcmp(a.bytes.data() + offset, b.bytes.data() + offset,
                     channels * 2) == 0;
}
absl::StatusOr<std::vector<std::vector<uint8_t>>> Snapshot(
    cuda::Executor& executor, const Layer& model) {
  std::vector<std::vector<uint8_t>> result;
  for (const auto& weight : model.weights()) {
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<uint8_t>::Allocate(
                                    executor, weight.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), weight.data(), host.size_bytes(),
                        cudaMemcpyDeviceToHost, executor.stream()),
        "snapshot compatibility source weights"));
    RETURN_IF_ERROR(executor.Synchronize());
    result.emplace_back(host.begin(), host.end());
  }
  return result;
}

struct Capital {
  size_t line;                // Original one-based corpus index.
  int target;                 // Scoring only, never passed to forward.
  std::vector<int> prefix;    // Five input IDs; excludes the target.
  TokenTraceResult baseline;  // Owns immutable donor activation bytes.
};

// Reports every recorded forward's complete query logits. Captured forwards
// additionally retain all decoded coordinates, raw activation bytes and all
// causal attention probabilities; plain identity/post controls omit captures.
class Reports {
 public:
  explicit Reports(const std::filesystem::path& directory)
      : scores_(directory / "conditions.tsv"),
        controls_(directory / "controls.tsv"),
        probabilities_(directory / "probabilities.tsv"),
        captures_(directory / "captures.tsv"),
        values_(directory / "activation_values.tsv"),
        bytes_(directory / "activation_bytes.bin", std::ios::binary),
        logits_(directory / "logits.f32", std::ios::binary) {
    for (auto* stream : Streams())
      *stream << std::setprecision(17);
    scores_ << "run\tkind\trecipient_line\tdonor_line\tmask\twinner\t"
               "winner_text\tcategory\trecipient_target\tdonor_target\t"
               "recipient_probability\trecipient_margin\tdonor_probability\t"
               "donor_margin\n";
    controls_ << "control\trecipient_line\tdonor_line\tmask\tpassed\n";
    probabilities_ << "run\tsite\thead\tquery\tkey\tprobability\n";
    captures_ << "run\tsite\tdtype\tfirst_row\trows\tchannels\tbyte_offset\t"
                 "byte_count\n";
    values_ << "run\tsite\trow\tchannel\tvalue\n";
  }
  absl::Status Good() {
    for (auto* stream : Streams())
      if (!*stream)
        return absl::UnknownError("cannot write query/country report");
    return absl::OkStatus();
  }
  absl::Status Finish() {
    for (auto* stream : Streams())
      stream->flush();
    return Good();
  }
  absl::Status Check(bool passed, absl::string_view name, size_t r = 0,
                     size_t d = 0, int mask = -1) {
    controls_ << name << '\t' << r << '\t' << d << '\t' << mask << '\t'
              << passed << '\n';
    if (!passed)
      return absl::FailedPreconditionError(
          absl::StrCat("control failed: ", name));
    ++control_count_;
    return Good();
  }
  absl::Status Record(absl::string_view kind, const Capital& recipient,
                      const Capital& donor, int mask,
                      const TokenTraceResult& result,
                      const tokenizer::CompactVocabularyTokenizer& vocabulary,
                      const tokenizer::Gpt2Detokenizer& decoder) {
    if (result.query_row != kQuery || result.logits.size() != kVocabulary)
      return absl::DataLossError("invalid compatibility logits");
    double maximum = -std::numeric_limits<double>::infinity();
    for (float value : result.logits) {
      if (!std::isfinite(value))
        return absl::DataLossError("nonfinite compatibility logits");
      maximum = std::max(maximum, static_cast<double>(value));
    }
    const int winner = static_cast<int>(
        std::max_element(result.logits.begin(), result.logits.end()) -
        result.logits.begin());
    if (winner != result.predicted_token)
      return absl::DataLossError("compatibility winner disagrees with trace");
    double sum = 0;
    for (float value : result.logits)
      sum += std::exp(value - maximum);
    ASSIGN_OR_RETURN(int original, vocabulary.OriginalId(winner));
    ASSIGN_OR_RETURN(auto text, decoder.Decode({&original, 1}));
    const char* category = winner == recipient.target ? "recipient"
                           : winner == donor.target   ? "donor"
                                                      : "third";
    scores_ << run_count_ << '\t' << kind << '\t' << recipient.line << '\t'
            << donor.line << '\t' << mask << '\t' << winner << '\t'
            << Escape(text) << '\t' << category << '\t' << recipient.target
            << '\t' << donor.target;
    for (int target : {recipient.target, donor.target}) {
      double rival = -std::numeric_limits<double>::infinity();
      for (int token = 0; token < kVocabulary; ++token)
        if (token != target)
          rival = std::max(rival, static_cast<double>(result.logits[token]));
      scores_ << '\t' << std::exp(result.logits[target] - maximum) / sum << '\t'
              << result.logits[target] - rival;
    }
    scores_ << '\n';
    logits_.write(reinterpret_cast<const char*>(result.logits.data()),
                  result.logits.size() * sizeof(float));
    for (const auto& activation : result.activations) {
      const auto site = SiteName(activation.site);
      captures_ << run_count_ << '\t' << site << '\t'
                << static_cast<int>(activation.data_type) << '\t'
                << activation.first_row << '\t' << activation.row_count << '\t'
                << activation.channels << '\t' << byte_offset_ << '\t'
                << activation.bytes.size() << '\n';
      bytes_.write(reinterpret_cast<const char*>(activation.bytes.data()),
                   activation.bytes.size());
      byte_offset_ += activation.bytes.size();
      for (size_t row = 0; row < activation.row_count; ++row)
        for (size_t channel = 0; channel < activation.channels; ++channel)
          values_ << run_count_ << '\t' << site << '\t'
                  << activation.first_row + row << '\t' << channel << '\t'
                  << activation.values[row * activation.channels + channel]
                  << '\n';
    }
    for (const auto& attention : result.attention)
      for (size_t head = 0; head < attention.heads; ++head)
        for (size_t q = 0; q < attention.prefix_length; ++q)
          for (size_t k = 0; k < attention.prefix_length; ++k)
            probabilities_
                << run_count_ << '\t' << SiteName(attention.site) << '\t'
                << head << '\t' << q << '\t' << k << '\t'
                << attention
                       .probabilities[(head * attention.prefix_length + q) *
                                          attention.prefix_length +
                                      k]
                << '\n';
    if (kind == "factorial")
      std::cout << recipient.line << " <- " << donor.line << " mask " << mask
                << ": " << category << " \"" << Escape(text) << "\"\n";
    ++run_count_;
    return Good();
  }
  size_t run_count() const { return run_count_; }
  size_t control_count() const { return control_count_; }

 private:
  std::array<std::ofstream*, 7> Streams() {
    return {&scores_, &controls_, &probabilities_, &captures_,
            &values_, &bytes_,    &logits_};
  }
  std::ofstream scores_, controls_, probabilities_, captures_, values_, bytes_,
      logits_;
  size_t run_count_ = 0, control_count_ = 0, byte_offset_ = 0;
};

absl::StatusOr<std::vector<TokenTracePatch>> Patches(const Capital& donor,
                                                     int mask) {
  ASSIGN_OR_RETURN(auto query, Activation(donor.baseline, BoundarySite(), 16));
  ASSIGN_OR_RETURN(
      auto qkv,
      Activation(donor.baseline, AttentionSite("FullyConnectedLayer"), 48));
  std::vector<TokenTracePatch> result{
      {.site = BoundarySite(),
       .rows = TokenTraceRows::kOne,
       .row = kQuery,
       .replacement = TokenTraceReplacement::kDonor,
       .donor = query}};
  for (int bit = 0; bit < 2; ++bit)
    if (mask & (1 << bit))
      result.push_back({.site = AttentionSite("FullyConnectedLayer"),
                        .rows = TokenTraceRows::kOne,
                        .row = kCountry,
                        .first_channel = static_cast<size_t>((bit + 1) * 16),
                        .channel_count = 16,
                        .replacement = TokenTraceReplacement::kDonor,
                        .donor = qkv});
  return result;
}

absl::Status Run() {
  namespace fs = std::filesystem;
  const fs::path checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const fs::path directory = absl::GetFlag(FLAGS_output_dir);
  if (checkpoint.empty() || directory.empty() ||
      absl::GetFlag(FLAGS_tokenizer).empty())
    return absl::InvalidArgumentError(
        "checkpoint/tokenizer/fresh output_dir required");
  std::error_code error;
  if (fs::exists(directory, error))
    return absl::AlreadyExistsError("output_dir must be fresh");
  if (error)
    return absl::UnknownError(error.message());
  ASSIGN_OR_RETURN(
      auto base, tokenizer::Gpt2Tokenizer::Load(absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto vocabulary,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *base, checkpoint / "compact_vocabulary.tsv"));
  if (vocabulary->vocab_size() != kVocabulary ||
      vocabulary->original_eos_token_id() != base->eos_token_id())
    return absl::InvalidArgumentError(
        "compatibility probe requires 4,475 tokens and the original EOS");
  ASSIGN_OR_RETURN(auto decoder, tokenizer::Gpt2Detokenizer::Load(
                                     absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  std::vector<absl::string_view> lines = absl::StrSplit(corpus.text(), '\n');
  if (!lines.empty() && lines.back().empty())
    lines.pop_back();
  if (lines.size() != 1024)
    return absl::InvalidArgumentError("expected 1,024 corpus lines");
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto model, CreateGpt2(*executor, DataType::BF16, 0,
                                          {.transformer_block_count = 8,
                                           .model_width = 16,
                                           .attention_heads = 1,
                                           .feed_forward_width = 64,
                                           .vocabulary_size = kVocabulary,
                                           .pad_vocabulary = false}));
  RETURN_IF_ERROR(
      ReadFromDirectory(*executor, *model, checkpoint.string(), false));
  ASSIGN_OR_RETURN(auto before, Snapshot(*executor, *model));
  const TokenTraceOptions plain{.vocabulary_size = kVocabulary,
                                .padding_token = vocabulary->eos_token_id()};
  auto capture = plain;
  capture.capture_activations = capture.capture_attention = true;
  const std::array<size_t, 3> selected{80, 406, 411};
  const std::array<absl::string_view, 3> prompts{"The capital of France is",
                                                 "The capital of Greece is",
                                                 "The capital of Peru is"};
  std::vector<Capital> capitals;
  capitals.reserve(3);
  if (!fs::create_directory(directory, error) || error)
    return absl::UnknownError("cannot create compatibility output directory");
  Reports reports(directory);
  RETURN_IF_ERROR(reports.Good());
  for (size_t i = 0; i < selected.size(); ++i) {
    ASSIGN_OR_RETURN(auto tokens,
                     vocabulary->Encode(*executor, lines[selected[i] - 1]));
    ASSIGN_OR_RETURN(auto expected_prefix,
                     vocabulary->Encode(*executor, prompts[i]));
    if (tokens.size() <= 5 || expected_prefix.size() != 5 ||
        !std::equal(expected_prefix.begin(), expected_prefix.end(),
                    tokens.begin()))
      return absl::InvalidArgumentError(
          "fixed capital template does not match corpus");
    Capital item{
        selected[i], tokens[5], {tokens.begin(), tokens.begin() + 5}, {}};
    ASSIGN_OR_RETURN(auto ordinary,
                     TraceNextToken(*executor, *model, item.prefix, plain));
    ASSIGN_OR_RETURN(item.baseline,
                     TraceNextToken(*executor, *model, item.prefix, capture));
    RETURN_IF_ERROR(
        reports.Check(item.baseline.predicted_token == item.target &&
                          EqualFloats(ordinary.logits, item.baseline.logits),
                      "baseline_capture_plain_correct", item.line));
    RETURN_IF_ERROR(reports.Record("baseline_plain", item, item, -1, ordinary,
                                   *vocabulary, *decoder));
    RETURN_IF_ERROR(reports.Record("baseline", item, item, -1, item.baseline,
                                   *vocabulary, *decoder));
    capitals.push_back(std::move(item));
  }
  for (const auto& recipient : capitals) {
    for (int mask = 0; mask < 4; ++mask) {
      ASSIGN_OR_RETURN(auto patches, Patches(recipient, mask));
      auto options = plain;
      options.patches = patches;
      ASSIGN_OR_RETURN(auto identity, TraceNextToken(*executor, *model,
                                                     recipient.prefix, options));
      RETURN_IF_ERROR(reports.Check(
          EqualFloats(identity.logits, recipient.baseline.logits),
          "same_source_identity", recipient.line, recipient.line, mask));
      RETURN_IF_ERROR(reports.Record("identity", recipient, recipient, mask,
                                     identity, *vocabulary, *decoder));
    }
    ASSIGN_OR_RETURN(auto recipient_boundary,
                     Activation(recipient.baseline, BoundarySite(), 16));
    ASSIGN_OR_RETURN(auto recipient_qkv,
                     Activation(recipient.baseline,
                                AttentionSite("FullyConnectedLayer"), 48));
    for (const auto& donor : capitals) {
      if (recipient.line == donor.line)
        continue;
      ASSIGN_OR_RETURN(auto donor_boundary,
                       Activation(donor.baseline, BoundarySite(), 16));
      ASSIGN_OR_RETURN(
          auto donor_qkv,
          Activation(donor.baseline, AttentionSite("FullyConnectedLayer"), 48));
      ASSIGN_OR_RETURN(auto donor_probabilities, Probabilities(donor.baseline));
      for (const auto& pair :
           {std::make_pair(recipient_boundary, donor_boundary),
            std::make_pair(recipient_qkv, donor_qkv)}) {
        bool same = true;
        for (size_t row = 0; row < 3; ++row)
          same &=
              EqualRow(*pair.first, *pair.second, row, 0, pair.first->channels);
        RETURN_IF_ERROR(reports.Check(same, "shared_anchor_rows",
                                      recipient.line, donor.line,
                                      static_cast<int>(pair.first->channels)));
      }
      std::array<TokenTraceResult, 4> corners;
      for (int mask = 0; mask < 4; ++mask) {
        ASSIGN_OR_RETURN(auto patches, Patches(donor, mask));
        auto options = capture;
        options.patches = patches;
        ASSIGN_OR_RETURN(
            corners[mask],
            TraceNextToken(*executor, *model, recipient.prefix, options));
        const auto& result = corners[mask];
        ASSIGN_OR_RETURN(auto boundary, Activation(result, BoundarySite(), 16));
        bool boundary_ok = EqualRow(*boundary, *donor_boundary, 4, 0, 16);
        for (size_t row = 0; row < 4; ++row)
          boundary_ok &= EqualRow(*boundary, *recipient_boundary, row, 0, 16);
        RETURN_IF_ERROR(reports.Check(boundary_ok,
                                      "only_query_boundary_changed",
                                      recipient.line, donor.line, mask));
        ASSIGN_OR_RETURN(
            auto qkv,
            Activation(result, AttentionSite("FullyConnectedLayer"), 48));
        bool qkv_ok = EqualRow(*qkv, *donor_qkv, 4, 0, 48);
        for (size_t row = 0; row < 3; ++row)
          qkv_ok &= EqualRow(*qkv, *recipient_qkv, row, 0, 48);
        for (size_t part = 0; part < 3; ++part) {
          const bool use_donor = part != 0 && (mask & (1 << (part - 1)));
          qkv_ok &= EqualRow(*qkv, use_donor ? *donor_qkv : *recipient_qkv, 3,
                             part * 16, 16);
        }
        RETURN_IF_ERROR(reports.Check(qkv_ok, "exact_selected_qkv_segments",
                                      recipient.line, donor.line, mask));
        ASSIGN_OR_RETURN(auto probabilities, Probabilities(result));
        if (mask & 1)
          RETURN_IF_ERROR(reports.Check(
              EqualFloats(
                  absl::MakeConstSpan(probabilities->probabilities)
                      .subspan(20, 5),
                  absl::MakeConstSpan(donor_probabilities->probabilities)
                      .subspan(20, 5)),
              "donor_key_restores_query_routing", recipient.line, donor.line,
              mask));
        if (mask >= 2) {
          ASSIGN_OR_RETURN(auto same_keys, Probabilities(corners[mask - 2]));
          RETURN_IF_ERROR(
              reports.Check(EqualFloats(probabilities->probabilities,
                                        same_keys->probabilities),
                            "value_does_not_change_probabilities",
                            recipient.line, donor.line, mask));
        }
        if (mask == 3) {
          for (const auto& site :
               {AttentionSite("AttentionLayer"), MlpSite("LayerNormLayer"),
                MlpSite("GeluLayer")}) {
            const size_t width = site.layer_name == "GeluLayer" ? 64 : 16;
            ASSIGN_OR_RETURN(auto actual, Activation(result, site, width));
            ASSIGN_OR_RETURN(auto expected,
                             Activation(donor.baseline, site, width));
            RETURN_IF_ERROR(
                reports.Check(EqualRow(*actual, *expected, 4, 0, width),
                              absl::StrCat("both_restore_", site.layer_name),
                              recipient.line, donor.line, mask));
          }
          RETURN_IF_ERROR(
              reports.Check(EqualFloats(result.logits, donor.baseline.logits),
                            "both_restore_all_donor_logits", recipient.line,
                            donor.line, mask));
        }
        RETURN_IF_ERROR(reports.Record("factorial", recipient, donor, mask,
                                       result, *vocabulary, *decoder));
        ASSIGN_OR_RETURN(auto post, TraceNextToken(*executor, *model,
                                                   recipient.prefix, plain));
        RETURN_IF_ERROR(reports.Check(
            EqualFloats(post.logits, recipient.baseline.logits),
            "ordinary_post_logits", recipient.line, donor.line, mask));
        RETURN_IF_ERROR(reports.Record("post", recipient, donor, mask, post,
                                       *vocabulary, *decoder));
      }
    }
  }
  ASSIGN_OR_RETURN(auto after, Snapshot(*executor, *model));
  RETURN_IF_ERROR(
      reports.Check(before == after, "source_master_bytes_unchanged"));
  RETURN_IF_ERROR(reports.Finish());
  std::ofstream manifest(directory / "manifest.tsv");
  manifest
      << "key\tvalue\ncheckpoint\t" << checkpoint.string() << "\ntokenizer\t"
      << absl::GetFlag(FLAGS_tokenizer) << "\ncorpus\t"
      << absl::GetFlag(FLAGS_corpus)
      << "\nlines\t80,406,411\nprefix_tokens\t5\nfuture_input\tEOS only"
         "\nquery_donor_site\tblock 6 post-MLP residual row 4"
         "\ncountry_donor_site\tblock 7 packed QKV row 3"
         "\nmask\t0=query only;1=+country K;2=+country V;3=+country KV"
         "\nprimary_forwards\t24\nprotocol\tfixed six ordered pairs; no search"
         "\nlabels\tscoring only, never passed to forward"
         "\ninterpretation\toff-manifold compatibility intervention, not "
         "exclusive storage"
         "\nlogits_format\tnative little-endian FP32; 4475 per conditions.tsv "
         "run"
         "\nactivation_bytes_format\tnative physical bytes, indexed by "
         "captures.tsv"
         "\nrecorded_forwards\t"
      << reports.run_count() << "\npassed_controls\t" << reports.control_count()
      << "\ncompleted\ttrue\n";
  manifest.close();
  if (!manifest)
    return absl::UnknownError("cannot finish compatibility manifest");
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
