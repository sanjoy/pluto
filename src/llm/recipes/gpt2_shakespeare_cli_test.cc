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

TEST(Gpt2ShakespeareCliTest, ParsesActivationInspectionOptions) {
  struct Case {
    absl::string_view text;
    ActivationInspectionMode mode;
    double min_prob;
  };
  const Case cases[] = {
      {"", ActivationInspectionMode::kDisabled, 0.01},
      {" \t\r\n", ActivationInspectionMode::kDisabled, 0.01},
      {"neighboring_vocab", ActivationInspectionMode::kNeighboringVocab, 0.01},
      {"neighboring_vocab()", ActivationInspectionMode::kNeighboringVocab,
       0.01},
      {" neighboring_vocab ( ) ", ActivationInspectionMode::kNeighboringVocab,
       0.01},
      {"neighboring_vocab(min_prob=0.05)",
       ActivationInspectionMode::kNeighboringVocab, 0.05},
      {"\tneighboring_vocab ( min_prob = .05 )\r\n",
       ActivationInspectionMode::kNeighboringVocab, 0.05},
      {"neighboring_vocab(min_prob=5e-2)",
       ActivationInspectionMode::kNeighboringVocab, 0.05},
      {"neighboring_vocab(min_prob=+0.5)",
       ActivationInspectionMode::kNeighboringVocab, 0.5},
      {"neighboring_vocab(min_prob=0)",
       ActivationInspectionMode::kNeighboringVocab, 0.0},
      {"neighboring_vocab(min_prob=-0.0)",
       ActivationInspectionMode::kNeighboringVocab, 0.0},
      {"neighboring_vocab(min_prob=1)",
       ActivationInspectionMode::kNeighboringVocab, 1.0},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(test.text);
    auto parsed = ParseActivationInspectionMode(test.text);
    ASSERT_TRUE(parsed.ok()) << parsed.status();
    EXPECT_EQ(parsed->mode, test.mode);
    EXPECT_DOUBLE_EQ(parsed->min_prob, test.min_prob);
  }
}

TEST(Gpt2ShakespeareCliTest, RejectsUnknownModesMalformedSettingsAndModeLists) {
  for (absl::string_view invalid :
       {"disabled",
        "unknown",
        "NEIGHBORING_VOCAB",
        "neighboring _vocab",
        "neighboring_vocab|unknown",
        "neighboring_vocab|neighboring_vocab",
        "neighboring_vocab,neighboring_vocab",
        "neighboring_vocab()|neighboring_vocab()",
        "neighboring_vocab(min_prob=0.05)|neighboring_vocab",
        "neighboring_vocab(",
        "neighboring_vocab)",
        "neighboring_vocab(()",
        "neighboring_vocab())",
        "neighboring_vocab((min_prob=0.05))",
        "neighboring_vocab(min_prob=(0.05))",
        "neighboring_vocab()trailing",
        "neighboring_vocab()()",
        "(min_prob=0.05)",
        "neighboring_vocab(,)",
        "neighboring_vocab(,min_prob=0.05)",
        "neighboring_vocab(min_prob=0.05,)",
        "neighboring_vocab(min_prob=0.05,   )",
        "neighboring_vocab(min_prob=0.05,,min_prob=0.1)",
        "neighboring_vocab(min_prob)",
        "neighboring_vocab(min_prob=)",
        "neighboring_vocab(=0.05)",
        "neighboring_vocab( = )",
        "neighboring_vocab(min_prob==0.05)",
        "neighboring_vocab(min_prob=0.05=0.1)"}) {
    SCOPED_TRACE(invalid);
    auto parsed = ParseActivationInspectionMode(invalid);
    EXPECT_EQ(parsed.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(parsed.status().message().find("--inspect_activations"),
              absl::string_view::npos);
    EXPECT_NE(
        parsed.status().message().find("neighboring_vocab(min_prob=0.05)"),
        absl::string_view::npos);
  }
}

TEST(Gpt2ShakespeareCliTest, RejectsUnknownAndRepeatedInspectionKeys) {
  struct Case {
    absl::string_view text;
    absl::string_view reason;
  };
  const Case cases[] = {
      {"neighboring_vocab(threshold=0.05)", "unknown setting"},
      {"neighboring_vocab(MIN_PROB=0.05)", "unknown setting"},
      {"neighboring_vocab(min prob=0.05)", "unknown setting"},
      {"neighboring_vocab(min_prob=0.05,other=1)", "unknown setting"},
      {"neighboring_vocab(other=1,min_prob=0.05)", "unknown setting"},
      {"neighboring_vocab(min_prob=0.05,min_prob=0.05)", "duplicate min_prob"},
      {"neighboring_vocab(min_prob=0.05, min_prob = 0.1)",
       "duplicate min_prob"},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(test.text);
    auto parsed = ParseActivationInspectionMode(test.text);
    EXPECT_EQ(parsed.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(parsed.status().message().find(test.reason),
              absl::string_view::npos);
  }
}

TEST(Gpt2ShakespeareCliTest,
     InspectionProbabilityMustBeFiniteAndInUnitInterval) {
  for (absl::string_view invalid :
       {"neighboring_vocab(min_prob=-0.0001)",
        "neighboring_vocab(min_prob=1.0001)",
        "neighboring_vocab(min_prob=5)",  // Unit probability, not percent.
        "neighboring_vocab(min_prob=5%)", "neighboring_vocab(min_prob=nan)",
        "neighboring_vocab(min_prob=NaN)", "neighboring_vocab(min_prob=inf)",
        "neighboring_vocab(min_prob=+inf)", "neighboring_vocab(min_prob=-inf)",
        "neighboring_vocab(min_prob=infinity)",
        "neighboring_vocab(min_prob=1e999)",
        "neighboring_vocab(min_prob=-1e999)",
        "neighboring_vocab(min_prob=garbage)",
        "neighboring_vocab(min_prob=0.05garbage)",
        "neighboring_vocab(min_prob=0. 05)",
        "neighboring_vocab(min_prob=0.05|0.1)",
        "neighboring_vocab(min_prob=\"0.05\")"}) {
    SCOPED_TRACE(invalid);
    auto parsed = ParseActivationInspectionMode(invalid);
    EXPECT_EQ(parsed.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_NE(parsed.status().message().find("min_prob must be finite"),
              absl::string_view::npos);
  }
}

TEST(Gpt2ShakespeareCliTest,
     ActivationInspectionIsOnlyAllowedInModelInference) {
  for (Gpt2ShakespeareMode mode :
       {Gpt2ShakespeareMode::kTrainModel, Gpt2ShakespeareMode::kInferModel,
        Gpt2ShakespeareMode::kTrainSparseAutoEncoder,
        Gpt2ShakespeareMode::kInferSparseAutoEncoder}) {
    SCOPED_TRACE(Gpt2ShakespeareModeName(mode));
    EXPECT_TRUE(
        Validate(mode, {}).ok());  // Omitting inspection is always fine.
    const auto status = Validate(mode, {"inspect_activations"});
    if (mode == Gpt2ShakespeareMode::kInferModel)
      EXPECT_TRUE(status.ok()) << status;
    else {
      EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
      EXPECT_NE(status.message().find("--inspect_activations"),
                absl::string_view::npos);
    }
  }
}

TEST(Gpt2ShakespeareCliTest,
     ExplicitEmptyInspectionStillRejectedOutsideModelInference) {
  // Value parsing disables inspection, but explicit flag presence is still
  // mode-specific. --inspect_activations= must not bypass that policy.
  auto disabled = ParseActivationInspectionMode("");
  ASSERT_TRUE(disabled.ok()) << disabled.status();
  EXPECT_EQ(disabled->mode, ActivationInspectionMode::kDisabled);
  for (Gpt2ShakespeareMode mode :
       {Gpt2ShakespeareMode::kTrainModel,
        Gpt2ShakespeareMode::kTrainSparseAutoEncoder,
        Gpt2ShakespeareMode::kInferSparseAutoEncoder})
    EXPECT_EQ(Validate(mode, {"inspect_activations"}).code(),
              absl::StatusCode::kInvalidArgument);
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
  EXPECT_TRUE(Validate(Gpt2ShakespeareMode::kInferModel,
                       {"inference_from", "prompt", "generation_tokens",
                        "temperature", "inspect_activations"})
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
        "checkpoint_initial", "inspect_activations"}) {
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
