#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_certificate.h"

#include <algorithm>

#include "gtest/gtest.h"

namespace pluto::llm::discretized::generator {
namespace {
SymbolicModel DistinctOutputs(int size = 2) {
  int vocabulary = std::max(size, 3);
  SymbolicModel model;
  // A certificate needs only the typed transition system; it never accesses
  // corpus samples, vocabulary spellings, or reduction/search metadata.
  model.metadata = {.width = 1, .layers = 1, .vocab_size = vocabulary};
  model.transformers.resize(1);
  for (int stage = 0; stage < 3; ++stage)
    for (int i = 0; i < size; ++i)
      model.states.push_back({.id = vocabulary + stage * size + i,
                              .boundary = stage,
                              .bits = {0}});
  for (int i = 0; i < size; ++i) {
    model.entry.push_back({i, 0, vocabulary + i});
    model.transformers[0].attention.push_back(
        {{vocabulary + i}, vocabulary + size + i});
    model.transformers[0].mlp.push_back(
        {vocabulary + size + i, vocabulary + 2 * size + i});
    model.language_modeling_head.push_back({vocabulary + 2 * size + i, i});
  }
  return model;
}
TEST(DiscretizeCertificateTest,
     ProvesWithoutTrustingSearchMetadataOrMutatingInput) {
  auto model = DistinctOutputs();
  model.stats.search = SearchStatistics{.pairwise_irreducible = false};
  const auto before = model;
  auto result = CertifyModel(model);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->status, CertificateStatus::kProven);
  EXPECT_EQ(result->proven_pairs, 3);
  EXPECT_EQ(result->same_boundary_pairs, 3);
  EXPECT_EQ(result->attention_pairs_checked, 1);
  EXPECT_EQ(result->global_minimum_proven, false);
  EXPECT_EQ(model, before);
}
TEST(DiscretizeCertificateTest, DuplicateOrMissingReadoutIsInconclusive) {
  auto model = DistinctOutputs();
  model.language_modeling_head[1].output = 0;
  model.stats.search = SearchStatistics{.pairwise_irreducible = true};
  auto result = CertifyModel(model);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->status, CertificateStatus::kInconclusive);
  EXPECT_EQ(result->unresolved_stage, 2);
  EXPECT_EQ(result->proven_pairs, 0);
  model.language_modeling_head.erase(model.language_modeling_head.begin() + 1);
  result = CertifyModel(model);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->status, CertificateStatus::kInconclusive);
}
TEST(DiscretizeCertificateTest, ShapeAndBoundaryErrorsReturnStatus) {
  auto model = DistinctOutputs();
  model.states[0].bits = {0, 1};
  EXPECT_FALSE(CertifyModel(model).ok());
  model = DistinctOutputs();
  model.transformers[0].attention[0].output = 7;
  EXPECT_FALSE(CertifyModel(model).ok());
}
TEST(DiscretizeCertificateTest, ComparesTwoRewrittenHistories) {
  auto model = DistinctOutputs();
  model.transformers[0].attention = {{{3, 4}, 5}, {{4, 3}, 6}};
  auto result = CertifyModel(model);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->status, CertificateStatus::kProven);
}
TEST(DiscretizeCertificateTest, DifferentHistoryLengthsDoNotCollide) {
  auto model = DistinctOutputs();
  model.transformers[0].attention = {{{3}, 5}, {{4, 3}, 6}};
  auto result = CertifyModel(model);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->status, CertificateStatus::kInconclusive);
  EXPECT_EQ(result->unresolved_stage, 0);
  EXPECT_EQ(result->unresolved_pair, (std::array<int, 2>{3, 4}));
}
TEST(DiscretizeCertificateTest, NonInjectiveOrMissingMlpIsInconclusive) {
  auto model = DistinctOutputs();
  model.transformers[0].mlp[1].output = model.transformers[0].mlp[0].output;
  auto result = CertifyModel(model);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->status, CertificateStatus::kInconclusive);
  EXPECT_EQ(result->unresolved_stage, 1);
  model.transformers[0].mlp.erase(model.transformers[0].mlp.begin() + 1);
  result = CertifyModel(model);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->status, CertificateStatus::kInconclusive);
}
TEST(DiscretizeCertificateTest, LargeAlphabetHasNoByteEncodingRestriction) {
  auto result = CertifyModel(DistinctOutputs(257));
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->status, CertificateStatus::kProven);
  EXPECT_EQ(result->attention_pairs_checked, 257 * 256 / 2);
  EXPECT_EQ(result->proven_pairs, 3 * 257 * 256 / 2);
}
}  // namespace
}  // namespace pluto::llm::discretized::generator
