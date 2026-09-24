#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/generator_report.h"

#include <locale>
#include <string>

#include "gtest/gtest.h"

namespace pluto::llm::discretized::generator {
namespace {

TEST(GeneratorReportTest, EachProgressAlternativeHasTypedFields) {
  EXPECT_EQ(FormatProgress(CaptureProgress{.samples = 2,
                                           .total_samples = 4,
                                           .rows = 17,
                                           .greedy_verified = true}),
            "phase: capture\nsamples: 2\ntotal_samples: 4\nrows: 17\n"
            "greedy_verified: true\n");
  auto compaction = FormatProgress(
      CompactionProgress{.phase = CompactionPhase::kExhaustivePassComplete,
                         .pass = 3,
                         .states = 6,
                         .states_per_stage = {2, 2, 2},
                         .attempted = 5,
                         .accepted = 1,
                         .compactions = 2,
                         .seconds = 0.5});
  EXPECT_NE(compaction.find("phase: compaction_exhaustive_pass_complete\n"),
            std::string::npos);
  EXPECT_NE(compaction.find("states_per_stage: [2, 2, 2]\n"),
            std::string::npos);
  EXPECT_NE(compaction.find("seconds: 0.5\n"), std::string::npos);
  auto generated = FormatProgress(GenerationProgress{
      .phase = GenerationPhase::kGenerated,
      .states = 6,
      .verification = {.samples = 1, .targets = 2, .explicit_eos = 1},
      .output = "generated\n/path"});
  EXPECT_NE(generated.find("phase: generated\n"), std::string::npos);
  EXPECT_NE(generated.find("  errors: 0\n"), std::string::npos);
  EXPECT_NE(generated.find("output: generated\\n/path\n"), std::string::npos);
  EXPECT_EQ(FormatProgress(GenerationProgress{}).find("output:"),
            std::string::npos);
}

TEST(GeneratorReportTest,
     StatisticsSummarizeTypedCompactionsDeterministically) {
  ModelStatistics stats;
  stats.states = 2;
  stats.accepted_compactions = {{.boundary = 0,
                                 .seed_ids = {10, 11},
                                 .euclidean_distance = 0.5,
                                 .induced_compactions = 3},
                                {.boundary = 1,
                                 .seed_ids = {20, 21},
                                 .euclidean_distance = 0.25,
                                 .induced_compactions = 1}};
  stats.compaction_search = CompactionSearchStatistics{
      .stopping_reason = CompactionStoppingReason::kNoCompatiblePair,
      .pairwise_compaction_complete = true};
  const auto report = FormatStatistics(stats);
  EXPECT_EQ(FormatStatistics(stats), report);
  EXPECT_NE(report.find("accepted_compaction_records: 2\n"), std::string::npos);
  EXPECT_NE(report.find("  minimum: 0.25\n"), std::string::npos);
  EXPECT_NE(report.find("  maximum: 0.5\n"), std::string::npos);
  EXPECT_NE(report.find("  induced_compactions: 4\n"), std::string::npos);
  EXPECT_NE(report.find("global_minimum_proven: false\n"), std::string::npos);
  EXPECT_EQ(report.find("membership_original_states"), std::string::npos);
  stats.accepted_compactions[0].seed_ids[0] = 9;
  EXPECT_NE(FormatStatistics(stats), report);
}

TEST(GeneratorReportTest, PointwiseProgressIsDistinctFromPairwiseSearch) {
  EXPECT_NE(
      FormatProgress(CompactionProgress{.phase = CompactionPhase::kPointwise})
          .find("phase: compaction_pointwise\n"),
      std::string::npos);
  EXPECT_NE(
      FormatProgress(
          CompactionProgress{.phase = CompactionPhase::kPointwisePassComplete})
          .find("phase: compaction_pointwise_pass_complete\n"),
      std::string::npos);
}

TEST(GeneratorReportTest, InconclusiveCertificateDoesNotClaimMinimality) {
  CertificateResult certificate;
  certificate.reason = "no sufficient collision proof";
  certificate.unresolved_stage = 0;
  certificate.unresolved_pair = {10, 11};
  auto report = FormatCertificate(certificate);
  EXPECT_NE(report.find("status: inconclusive\n"), std::string::npos);
  EXPECT_NE(report.find("pairwise_compaction_complete: false\n"),
            std::string::npos);
  EXPECT_NE(report.find("global_minimum_proven: false\n"), std::string::npos);
  EXPECT_NE(report.find("unresolved_stage: 0\n"), std::string::npos);
  EXPECT_NE(report.find("unresolved_pair: [10, 11]\n"), std::string::npos);
}

TEST(GeneratorReportTest, SymbolicCompactionDoesNotInventDistances) {
  ModelStatistics stats;
  stats.accepted_compactions = {
      {.boundary = 0, .seed_ids = {10, 11}, .induced_compactions = 2}};
  const auto symbolic = FormatStatistics(stats);
  EXPECT_EQ(symbolic.find("minimum:"), std::string::npos);
  EXPECT_EQ(symbolic.find("maximum:"), std::string::npos);
  EXPECT_NE(symbolic.find("induced_compactions: 2"), std::string::npos);
  stats.accepted_compactions.front().euclidean_distance = 0;
  const auto hinted = FormatStatistics(stats);
  EXPECT_NE(hinted.find("minimum: 0"), std::string::npos);
  EXPECT_NE(hinted, symbolic);
}

TEST(GeneratorReportTest, StatisticsAndCompactionHashesIgnoreProcessLocale) {
  ModelStatistics stats;
  stats.states = 1024;
  stats.accepted_compactions = {{.boundary = 0,
                                 .seed_ids = {10000, 10001},
                                 .euclidean_distance = 0.25,
                                 .induced_compactions = 2}};
  const std::string expected = FormatStatistics(stats);
  class CommaNumbers : public std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
    char do_thousands_sep() const override { return '.'; }
    std::string do_grouping() const override { return "\3"; }
  };
  // std::locale owns the facet; restore process state even if assertions fail.
  const auto original = std::locale();
  struct RestoreLocale {
    std::locale locale;
    ~RestoreLocale() { std::locale::global(locale); }
  } restore{original};
  std::locale::global(std::locale(original, new CommaNumbers));
  EXPECT_EQ(FormatStatistics(stats), expected);
}

}  // namespace
}  // namespace pluto::llm::discretized::generator
