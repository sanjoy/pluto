#include "src/llm/recipes/gpt2_shakespeare_cli.h"

#include <initializer_list>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

namespace pluto::llm {
namespace {

absl::Status Validate(Gpt2ShakespeareMode mode,
                      std::initializer_list<absl::string_view> flags,
                      absl::string_view inference_from = "/model/step_1",
                      absl::string_view sparse_from = "/model/step_1") {
  return ValidateGpt2ShakespeareModeFlags(
      mode, absl::MakeConstSpan(flags.begin(), flags.size()), inference_from,
      sparse_from);
}

TEST(Gpt2ShakespeareCliTest, ParsesOnlyNamedModes) {
  EXPECT_EQ(*ParseGpt2ShakespeareMode("train_model"),
            Gpt2ShakespeareMode::kTrainModel);
  EXPECT_EQ(*ParseGpt2ShakespeareMode("infer_model"),
            Gpt2ShakespeareMode::kInferModel);
  EXPECT_EQ(*ParseGpt2ShakespeareMode("train_sae"),
            Gpt2ShakespeareMode::kTrainSparseAutoEncoder);
  EXPECT_FALSE(ParseGpt2ShakespeareMode("").ok());
  EXPECT_FALSE(ParseGpt2ShakespeareMode("train").ok());
}

TEST(Gpt2ShakespeareCliTest, ModelTrainingAcceptsOnlyTrainingFlags) {
  EXPECT_TRUE(Validate(Gpt2ShakespeareMode::kTrainModel,
                       {"corpus", "resume_from", "checkpoint_dir", "steps",
                        "test_fraction", "batch_size"})
                  .ok());
  EXPECT_EQ(
      Validate(Gpt2ShakespeareMode::kTrainModel, {"inference_from"}).code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      Validate(Gpt2ShakespeareMode::kTrainModel, {"sparse_autoencoder_from"})
          .code(),
      absl::StatusCode::kInvalidArgument);
}

TEST(Gpt2ShakespeareCliTest, InferenceAcceptsOnlyInferenceFlags) {
  EXPECT_TRUE(
      Validate(Gpt2ShakespeareMode::kInferModel,
               {"inference_from", "prompt", "generation_tokens", "temperature"})
          .ok());
  EXPECT_EQ(Validate(Gpt2ShakespeareMode::kInferModel, {"resume_from"}).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(Validate(Gpt2ShakespeareMode::kInferModel, {"steps"}).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(Validate(Gpt2ShakespeareMode::kInferModel, {}, "", "/model").code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(Gpt2ShakespeareCliTest, SparseTrainingAllowsResumeButNotModelEvalFlags) {
  EXPECT_TRUE(Validate(Gpt2ShakespeareMode::kTrainSparseAutoEncoder,
                       {"sparse_autoencoder_from", "resume_from", "steps",
                        "checkpoint_dir", "batch_size"})
                  .ok());
  EXPECT_EQ(
      Validate(Gpt2ShakespeareMode::kTrainSparseAutoEncoder, {"test_fraction"})
          .code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      Validate(Gpt2ShakespeareMode::kTrainSparseAutoEncoder, {}, "/model", "")
          .code(),
      absl::StatusCode::kInvalidArgument);
}

TEST(Gpt2ShakespeareCliTest, RejectsFlagsMissingFromThePolicy) {
  EXPECT_EQ(Validate(Gpt2ShakespeareMode::kTrainModel, {"new_flag"}).code(),
            absl::StatusCode::kInternal);
}

}  // namespace
}  // namespace pluto::llm
