#include "src/llm/recipes/gpt2_shakespeare_cli.h"

#include <initializer_list>
#include <limits>

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
  EXPECT_EQ(*ParseGpt2ShakespeareMode("infer_SAE"),
            Gpt2ShakespeareMode::kInferSparseAutoEncoder);
  EXPECT_EQ(*ParseGpt2ShakespeareMode("infer_sae"),
            Gpt2ShakespeareMode::kInferSparseAutoEncoder);
  EXPECT_EQ(
      Gpt2ShakespeareModeName(Gpt2ShakespeareMode::kInferSparseAutoEncoder),
      "infer_SAE");
  EXPECT_FALSE(ParseGpt2ShakespeareMode("").ok());
  EXPECT_FALSE(ParseGpt2ShakespeareMode("train").ok());
}

TEST(Gpt2ShakespeareCliTest, ModelTrainingAcceptsOnlyTrainingFlags) {
  EXPECT_TRUE(Validate(Gpt2ShakespeareMode::kTrainModel,
                       {"corpus", "resume_from", "checkpoint_dir", "steps",
                        "test_fraction", "batch_size", "training_seconds",
                        "checkpoint_initial"})
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
  EXPECT_EQ(
      Validate(Gpt2ShakespeareMode::kInferModel, {"training_seconds"}).code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      Validate(Gpt2ShakespeareMode::kInferModel, {"checkpoint_initial"}).code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(Validate(Gpt2ShakespeareMode::kInferModel, {}, "", "/model").code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(Gpt2ShakespeareCliTest, SparseTrainingAllowsResumeButNotModelEvalFlags) {
  EXPECT_TRUE(Validate(Gpt2ShakespeareMode::kTrainSparseAutoEncoder,
                       {"sparse_autoencoder_from", "resume_from", "steps",
                        "checkpoint_dir", "batch_size", "training_seconds",
                        "checkpoint_initial"})
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

TEST(Gpt2ShakespeareCliTest, SparseInferenceRequiresBothCheckpoints) {
  constexpr auto mode = Gpt2ShakespeareMode::kInferSparseAutoEncoder;
  EXPECT_TRUE(
      Validate(mode, {"sparse_autoencoder_from", "inference_from", "prompt"})
          .ok());
  EXPECT_EQ(Validate(mode, {}, "", "/gpt2/step_1").code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(Validate(mode, {}, "/sae/step_1", "").code(),
            absl::StatusCode::kInvalidArgument);
  for (absl::string_view flag :
       {"resume_from", "steps", "batch_size", "corpus", "checkpoint_every",
        "checkpoint_dir", "learning_rate", "eval_batches", "test_fraction",
        "train_until_loss", "training_eval_interval", "log_file",
        "generation_tokens", "temperature", "training_seconds",
        "checkpoint_initial"}) {
    EXPECT_EQ(Validate(mode, {flag}).code(), absl::StatusCode::kInvalidArgument)
        << flag;
  }
  EXPECT_FALSE(
      Validate(Gpt2ShakespeareMode::kInferModel, {"sparse_autoencoder_from"})
          .ok());
  EXPECT_FALSE(
      Validate(Gpt2ShakespeareMode::kTrainSparseAutoEncoder, {"inference_from"})
          .ok());
}

TEST(Gpt2ShakespeareCliTest, RejectsFlagsMissingFromThePolicy) {
  EXPECT_EQ(Validate(Gpt2ShakespeareMode::kTrainModel, {"new_flag"}).code(),
            absl::StatusCode::kInternal);
}

TEST(Gpt2ShakespeareCliTest, TrainingSecondsMustBePositiveAndFinite) {
  EXPECT_TRUE(ValidateGpt2ShakespeareTrainingSeconds(14'400.0).ok());
  EXPECT_TRUE(ValidateGpt2ShakespeareTrainingSeconds(1e-9).ok());
  for (double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_EQ(ValidateGpt2ShakespeareTrainingSeconds(invalid).code(),
              absl::StatusCode::kInvalidArgument);
  }
}

}  // namespace
}  // namespace pluto::llm
