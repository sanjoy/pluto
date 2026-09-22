#include <type_traits>

#include "gtest/gtest.h"
#include "pluto/discretized/gen/model.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/runtime.h"

namespace pluto::llm::discretized {
namespace {

// An external consumer needs no generated boundary factories, vocabulary
// constants, or CLI support headers to construct and use the model.
static_assert(
    std::is_same_v<decltype(gen::GeneratedModel()), const DiscreteModel&>);

TEST(GeneratedPublicApi, FactoryProvidesStableUsableModel) {
  const DiscreteModel& model = gen::GeneratedModel();
  EXPECT_EQ(&model, &gen::GeneratedModel());
  ASSERT_TRUE(ValidateModel(model).ok());
  ASSERT_FALSE(model.vocabulary.empty());
  ASSERT_FALSE(model.transformers.empty());

  // Even corpus-independent operations work with just the factory result.
  const DiscreteToken eos[] = {model.eos_token};
  const auto decoded = Decode(model, eos);
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  EXPECT_FALSE(decoded->empty());
}

}  // namespace
}  // namespace pluto::llm::discretized
