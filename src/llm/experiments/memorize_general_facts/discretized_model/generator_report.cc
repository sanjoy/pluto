#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_report.h"

#include <algorithm>
#include <iomanip>
#include <locale>
#include <sstream>
#include <type_traits>

#include "absl/strings/escaping.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_io.h"

namespace pluto::llm::discretized::generator {
namespace {

// A text sink, not an intermediate representation: each value is formatted as
// it arrives. Field selection remains explicit in the typed functions below.
class ReportWriter {
 public:
  ReportWriter() {
    stream_.imbue(std::locale::classic());
    stream_ << std::boolalpha << std::setprecision(17);
  }

  template <typename T>
  void Field(const char* name, const T& value) {
    stream_ << name << ": " << value << '\n';
  }
  void Sequence(const char* name, const std::vector<int>& values) {
    stream_ << name << ": [";
    for (size_t i = 0; i < values.size(); ++i)
      stream_ << (i == 0 ? "" : ", ") << values[i];
    stream_ << "]\n";
  }
  void Section(const char* name, const std::string& value) {
    stream_ << name << ":\n";
    std::istringstream lines(value);
    for (std::string line; std::getline(lines, line);)
      stream_ << "  " << line << '\n';
  }
  std::string Finish() const { return stream_.str(); }

 private:
  std::ostringstream stream_;
};

const char* PhaseName(ReductionPhase phase) {
  switch (phase) {
    case ReductionPhase::kNearest:
      return "nearest";
    case ReductionPhase::kNearestPassComplete:
      return "nearest_pass_complete";
    case ReductionPhase::kExhaustive:
      return "exhaustive";
    case ReductionPhase::kExhaustivePassComplete:
      return "exhaustive_pass_complete";
  }
  return "unknown";
}

const char* StoppingReasonName(SearchStoppingReason reason) {
  switch (reason) {
    case SearchStoppingReason::kPassLimit:
      return "pass_limit";
    case SearchStoppingReason::kAttemptLimit:
      return "attempt_limit";
    case SearchStoppingReason::kNearestCandidatesExhausted:
      return "nearest_candidates_exhausted";
    case SearchStoppingReason::kNoCompatiblePair:
      return "no_compatible_pair";
  }
  return "unknown";
}

std::string FormatReductionProgress(const ReductionProgress& progress) {
  ReportWriter report;
  report.Field("phase", PhaseName(progress.phase));
  report.Field("pass", progress.pass);
  report.Field("states", progress.states);
  report.Sequence("states_per_stage", progress.states_per_stage);
  report.Field("attempted", progress.attempted);
  report.Field("accepted", progress.accepted);
  report.Field("unions", progress.unions);
  report.Field("seconds", progress.seconds);
  return report.Finish();
}

std::string FormatSearch(const SearchStatistics& stats) {
  ReportWriter report;
  report.Field("stopping_reason", StoppingReasonName(stats.stopping_reason));
  report.Field("pairwise_irreducible", stats.pairwise_irreducible);
  report.Field("global_minimum_proven", stats.global_minimum_proven);
  report.Field("nearest_neighbors", stats.nearest_neighbors);
  report.Field("exhaustive_pair_limit", stats.exhaustive_pair_limit);
  report.Field("remaining_pairs_before_sweep",
               stats.remaining_pairs_before_sweep);
  for (const auto& event : stats.history)
    report.Section("pass", FormatReductionProgress(event));
  report.Field("seconds", stats.seconds);
  return report.Finish();
}

std::string FormatRelabeling(const RelabelStatistics& stats) {
  ReportWriter report;
  report.Sequence("eligible_layers", stats.eligible_layers);
  report.Sequence("skipped_layers", stats.skipped_layers);
  report.Field("changed_states", stats.changed_states);
  report.Field("layer_boundaries_preserved", stats.layer_boundaries_preserved);
  report.Field("vocabulary_aligned_final_boundaries",
               stats.vocabulary_aligned_final_boundaries);
  if (stats.last_attention_base)
    report.Field("last_attention_base", *stats.last_attention_base);
  if (stats.final_state_base)
    report.Field("final_state_base", *stats.final_state_base);
  if (stats.reserved_range_size)
    report.Field("reserved_range_size", *stats.reserved_range_size);
  return report.Finish();
}

}  // namespace

std::string FormatVerification(const VerificationResult& result) {
  ReportWriter report;
  report.Field("samples", result.samples);
  report.Field("targets", result.targets);
  report.Field("errors", result.errors);
  report.Field("explicit_eos", result.explicit_eos);
  return report.Finish();
}

std::string FormatProgress(const ProgressEvent& event) {
  return std::visit(
      [](const auto& value) -> std::string {
        using Event = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Event, ReductionProgress>) {
          return FormatReductionProgress(value);
        } else {
          ReportWriter report;
          if constexpr (std::is_same_v<Event, CaptureProgress>) {
            report.Field("phase", "capture");
            report.Field("samples", value.samples);
            report.Field("total_samples", value.total_samples);
            report.Field("rows", value.rows);
            report.Field("greedy_verified", value.greedy_verified);
          } else {
            report.Field("phase", value.phase == GenerationPhase::kBaseline
                                      ? "baseline"
                                      : "generated");
            report.Field("states", value.states);
            report.Section("verification",
                           FormatVerification(value.verification));
            if (value.phase == GenerationPhase::kGenerated)
              report.Field("output", absl::CEscape(value.output));
          }
          return report.Finish();
        }
      },
      event);
}

std::string FormatStatistics(const ModelStatistics& stats) {
  ReportWriter report;
  report.Field("captured_samples", stats.captured_samples);
  report.Field("scored_targets", stats.scored_targets);
  report.Field("exact_states", stats.exact_states);
  report.Field("states", stats.states);
  report.Field("membership_complete", stats.membership_complete);
  if (stats.membership_original_states)
    report.Field("membership_original_states",
                 *stats.membership_original_states);
  report.Sequence("states_per_stage", stats.states_per_stage);
  report.Field("attempted_seeds", stats.attempted_seeds);
  report.Field("accepted_seeds", stats.accepted_seeds);
  report.Field("state_unions", stats.state_unions);
  report.Field("cached_rejections", stats.cached_rejections);
  // Hash a canonical typed record encoding, not the human-readable summary.
  // The encoding is fixed-order decimal fields, with round-trippable doubles.
  if (!stats.accepted_merges.empty()) {
    std::ostringstream records;
    records.imbue(std::locale::classic());
    records << std::setprecision(17);
    double minimum = stats.accepted_merges.front().euclidean_distance;
    double maximum = minimum;
    int64_t induced = 0;
    for (const auto& merge : stats.accepted_merges) {
      records << merge.boundary << '\t' << merge.seed_ids[0] << '\t'
              << merge.seed_ids[1] << '\t' << merge.euclidean_distance << '\t'
              << merge.induced_unions << '\n';
      minimum = std::min(minimum, merge.euclidean_distance);
      maximum = std::max(maximum, merge.euclidean_distance);
      induced += merge.induced_unions;
    }
    report.Field("accepted_merges_sha256", Sha256(records.str()));
    report.Field("accepted_merge_records", stats.accepted_merges.size());
    ReportWriter summary;
    summary.Field("minimum", minimum);
    summary.Field("maximum", maximum);
    summary.Field("induced_unions", induced);
    report.Section("merge_distance_summary", summary.Finish());
  }
  if (stats.verification)
    report.Section("verification", FormatVerification(*stats.verification));
  if (stats.search)
    report.Section("search", FormatSearch(*stats.search));
  if (stats.pointwise_relabeling)
    report.Section("pointwise_relabeling",
                   FormatRelabeling(*stats.pointwise_relabeling));
  return report.Finish();
}

std::string FormatCertificate(const CertificateResult& certificate) {
  ReportWriter report;
  report.Field("status", certificate.status == CertificateStatus::kProven
                             ? "proven"
                             : "inconclusive");
  report.Field("method", "backward transition collision proof");
  report.Field("states", certificate.states);
  report.Field("boundaries", certificate.boundaries);
  report.Field("same_boundary_pairs", certificate.same_boundary_pairs);
  report.Field("proven_pairs", certificate.proven_pairs);
  report.Field("attention_pairs_checked", certificate.attention_pairs_checked);
  report.Field("pairwise_irreducible_proven",
               certificate.pairwise_irreducible_proven);
  report.Field("global_minimum_proven", certificate.global_minimum_proven);
  for (const auto& stage : certificate.stages) {
    ReportWriter details;
    details.Field("stage", stage.stage);
    details.Field("states", stage.states);
    details.Field("pairs", stage.pairs);
    details.Field("argument", stage.argument);
    report.Section("stage", details.Finish());
  }
  if (!certificate.reason.empty())
    report.Field("reason", certificate.reason);
  if (certificate.unresolved_stage)
    report.Field("unresolved_stage", *certificate.unresolved_stage);
  if (certificate.unresolved_pair)
    report.Sequence("unresolved_pair", {(*certificate.unresolved_pair)[0],
                                        (*certificate.unresolved_pair)[1]});
  return report.Finish();
}

}  // namespace pluto::llm::discretized::generator
