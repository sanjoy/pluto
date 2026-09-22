#include "src/llm/experiments/memorize_general_facts/discretized_model/generator_report.h"

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
  auto reduction = FormatProgress(
      ReductionProgress{.phase = ReductionPhase::kExhaustivePassComplete,
                        .pass = 3,
                        .states = 6,
                        .states_per_stage = {2, 2, 2},
                        .attempted = 5,
                        .accepted = 1,
                        .unions = 2,
                        .seconds = 0.5});
  EXPECT_NE(reduction.find("phase: exhaustive_pass_complete\n"),
            std::string::npos);
  EXPECT_NE(reduction.find("states_per_stage: [2, 2, 2]\n"), std::string::npos);
  EXPECT_NE(reduction.find("seconds: 0.5\n"), std::string::npos);
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

TEST(GeneratorReportTest, StatisticsSummarizeTypedMergesDeterministically) {
  ModelStatistics stats;
  stats.states = 2;
  stats.accepted_merges = {{.boundary = 0,
                            .seed_ids = {10, 11},
                            .euclidean_distance = 0.5,
                            .induced_unions = 3},
                           {.boundary = 1,
                            .seed_ids = {20, 21},
                            .euclidean_distance = 0.25,
                            .induced_unions = 1}};
  stats.search = SearchStatistics{
      .stopping_reason = SearchStoppingReason::kNoCompatiblePair,
      .pairwise_irreducible = true};
  const auto report = FormatStatistics(stats);
  EXPECT_EQ(FormatStatistics(stats), report);
  EXPECT_NE(report.find("accepted_merge_records: 2\n"), std::string::npos);
  EXPECT_NE(report.find("  minimum: 0.25\n"), std::string::npos);
  EXPECT_NE(report.find("  maximum: 0.5\n"), std::string::npos);
  EXPECT_NE(report.find("  induced_unions: 4\n"), std::string::npos);
  EXPECT_NE(report.find("global_minimum_proven: false\n"), std::string::npos);
  EXPECT_EQ(report.find("membership_original_states"), std::string::npos);
  stats.accepted_merges[0].seed_ids[0] = 9;
  EXPECT_NE(FormatStatistics(stats), report);
}

TEST(GeneratorReportTest, InconclusiveCertificateDoesNotClaimMinimality) {
  CertificateResult certificate;
  certificate.reason = "no sufficient collision proof";
  certificate.unresolved_stage = 0;
  certificate.unresolved_pair = {10, 11};
  auto report = FormatCertificate(certificate);
  EXPECT_NE(report.find("status: inconclusive\n"), std::string::npos);
  EXPECT_NE(report.find("pairwise_irreducible_proven: false\n"),
            std::string::npos);
  EXPECT_NE(report.find("global_minimum_proven: false\n"), std::string::npos);
  EXPECT_NE(report.find("unresolved_stage: 0\n"), std::string::npos);
  EXPECT_NE(report.find("unresolved_pair: [10, 11]\n"), std::string::npos);
}

TEST(GeneratorReportTest, StatisticsAndMergeHashesIgnoreProcessLocale) {
  ModelStatistics stats;
  stats.states = 1024;
  stats.accepted_merges = {{.boundary = 0,
                            .seed_ids = {10000, 10001},
                            .euclidean_distance = 0.25,
                            .induced_unions = 2}};
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
