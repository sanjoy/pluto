#include <sstream>

#include "gtest/gtest.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/runtime.h"
namespace pluto::llm::discretized {
absl::Status VerifyGeneratedModel(const Model&, std::ostream&);
TEST(GeneratedIntegerModel, IndependentAutoregressiveCorpusVerification) {
  const auto& model = GeneratedModel();
  ASSERT_TRUE(ValidateModel(model).ok());
  std::ostringstream report;
  auto result = VerifyGeneratedModel(model, report);
  EXPECT_TRUE(result.ok()) << result << "\n" << report.str();
}
}  // namespace pluto::llm::discretized
