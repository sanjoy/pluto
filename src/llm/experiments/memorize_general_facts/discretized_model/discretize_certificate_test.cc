#include "src/llm/experiments/memorize_general_facts/discretized_model/discretize_certificate.h"

#include <algorithm>

#include "gtest/gtest.h"

namespace pluto::llm::discretized::generator {
namespace {
Json DistinctOutputs(int size = 2) {
  int vocabulary = std::max(size, 3);
  Json model = {{"schema", 1},
                {"width", 1},
                {"layers", 1},
                {"vocab_size", vocabulary},
                {"states", Json::array()},
                {"entry", Json::array()},
                {"attention", Json::array({Json::array()})},
                {"mlp", Json::array({Json::array()})},
                {"language_modeling_head", Json::array()}};
  for (int stage = 0; stage < 3; ++stage)
    for (int i = 0; i < size; ++i)
      model["states"].push_back({{"id", vocabulary + stage * size + i},
                                 {"stage", stage},
                                 {"bits", Json::array({0})}});
  for (int i = 0; i < size; ++i) {
    model["entry"].push_back({i, 0, vocabulary + i});
    model["attention"][0].push_back(
        {Json::array({vocabulary + i}), vocabulary + size + i});
    model["mlp"][0].push_back(
        {vocabulary + size + i, vocabulary + 2 * size + i});
    model["language_modeling_head"].push_back({vocabulary + 2 * size + i, i});
  }
  return model;
}
TEST(DiscretizeCertificateTest,
     ProvesWithoutTrustingSearchMetadataOrMutatingInput) {
  auto model = DistinctOutputs();
  model["stats"] = {{"search", {{"pairwise_irreducible", false}}}};
  const auto before = model;
  auto result = CertifyModel(model);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ((*result)["status"], "proven");
  EXPECT_EQ((*result)["proven_pairs"], 3);
  EXPECT_EQ((*result)["same_boundary_pairs"], 3);
  EXPECT_EQ((*result)["attention_pairs_checked"], 1);
  EXPECT_EQ((*result)["global_minimum_proven"], false);
  EXPECT_EQ(model, before);
}
TEST(DiscretizeCertificateTest, DuplicateOrMissingReadoutIsInconclusive) {
  auto model = DistinctOutputs();
  model["language_modeling_head"][1][1] = 0;
  model["stats"] = {{"search", {{"pairwise_irreducible", true}}}};
  auto result = CertifyModel(model);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ((*result)["status"], "inconclusive");
  EXPECT_EQ((*result)["unresolved_stage"], 2);
  EXPECT_EQ((*result)["proven_pairs"], 0);
  model["language_modeling_head"].erase(1);
  result = CertifyModel(model);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ((*result)["status"], "inconclusive");
}
TEST(DiscretizeCertificateTest, ShapeAndBoundaryErrorsReturnStatus) {
  auto model = DistinctOutputs();
  model["states"][0]["bits"] = {0, 1};
  EXPECT_FALSE(CertifyModel(model).ok());
  model = DistinctOutputs();
  model["attention"][0][0][1] = 7;
  EXPECT_FALSE(CertifyModel(model).ok());
}
TEST(DiscretizeCertificateTest, ComparesTwoRewrittenHistories) {
  auto model = DistinctOutputs();
  model["attention"] =
      Json::array({Json::array({Json::array({Json::array({3, 4}), 5}),
                                Json::array({Json::array({4, 3}), 6})})});
  auto result = CertifyModel(model);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ((*result)["status"], "proven");
}
TEST(DiscretizeCertificateTest, DifferentHistoryLengthsDoNotCollide) {
  auto model = DistinctOutputs();
  model["attention"] =
      Json::array({Json::array({Json::array({Json::array({3}), 5}),
                                Json::array({Json::array({4, 3}), 6})})});
  auto result = CertifyModel(model);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ((*result)["status"], "inconclusive");
  EXPECT_EQ((*result)["unresolved_stage"], 0);
  EXPECT_EQ((*result)["unresolved_pair"], Json::array({3, 4}));
}
TEST(DiscretizeCertificateTest, NonInjectiveOrMissingMlpIsInconclusive) {
  auto model = DistinctOutputs();
  model["mlp"][0][1][1] = model["mlp"][0][0][1];
  auto result = CertifyModel(model);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ((*result)["status"], "inconclusive");
  EXPECT_EQ((*result)["unresolved_stage"], 1);
  model["mlp"][0].erase(1);
  result = CertifyModel(model);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ((*result)["status"], "inconclusive");
}
TEST(DiscretizeCertificateTest, LargeAlphabetHasNoByteEncodingRestriction) {
  auto result = CertifyModel(DistinctOutputs(257));
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ((*result)["status"], "proven");
  EXPECT_EQ((*result)["attention_pairs_checked"], 257 * 256 / 2);
  EXPECT_EQ((*result)["proven_pairs"], 3 * 257 * 256 / 2);
}
}  // namespace
}  // namespace pluto::llm::discretized::generator
