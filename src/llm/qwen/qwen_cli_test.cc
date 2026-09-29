#include "src/llm/qwen/qwen_cli.h"

#include <initializer_list>
#include <limits>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

namespace pluto::llm::qwen {
namespace {

CommandLineOptions InferenceOptions() {
  CommandLineOptions options;
  options.mode = "infer_model";
  options.checkpoint = "/checkpoint/need-not-exist";
  options.max_new_tokens = 32;
  options.context_length = 512;
  return options;
}

CommandLineOptions TrainingOptions() {
  CommandLineOptions options;
  options.mode = "train_model";
  options.checkpoint = "/checkpoint/need-not-exist";
  options.sequence_length = 8;
  options.batch_size = 1;
  options.steps = 2;
  options.switch_every = 50;
  options.start_block = -1;
  options.learning_rate = 1e-5;
  return options;
}

CommandLineOptions EmbeddingAlgebraOptions() {
  CommandLineOptions options;
  options.mode = "embedding_algebra";
  options.checkpoint = "/checkpoint/need-not-exist";
  options.expression = "king - queen + boy";
  return options;
}

absl::Status Validate(const CommandLineOptions& options,
                      std::initializer_list<absl::string_view> flags = {}) {
  return ParseAndValidateRunMode(
             options, absl::MakeConstSpan(flags.begin(), flags.size()))
      .status();
}

void ExpectInvalid(const absl::Status& status, absl::string_view diagnostic) {
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  EXPECT_NE(status.message().find(diagnostic), absl::string_view::npos)
      << status;
}

TEST(QwenCliTest, ParsesOnlyExactNamedModes) {
  const auto inference = ParseAndValidateRunMode(InferenceOptions(), {});
  ASSERT_TRUE(inference.ok()) << inference.status();
  EXPECT_EQ(*inference, Mode::kInferModel);
  const auto training = ParseAndValidateRunMode(TrainingOptions(), {});
  ASSERT_TRUE(training.ok()) << training.status();
  EXPECT_EQ(*training, Mode::kTrainModel);
  const auto algebra = ParseAndValidateRunMode(EmbeddingAlgebraOptions(), {});
  ASSERT_TRUE(algebra.ok()) << algebra.status();
  EXPECT_EQ(*algebra, Mode::kEmbeddingAlgebra);
  for (absl::string_view mode :
       {"", " ", "train", "infer", "TRAIN_MODEL", "Infer_model", " train_model",
        "infer_model ", "infer_model\n", "train_sae", "train_model,infer_model",
        "algebra", "Embedding_Algebra", "embedding_algebra "}) {
    SCOPED_TRACE(mode);
    auto options = InferenceOptions();
    options.mode = std::string(mode);
    ExpectInvalid(Validate(options), "--mode");
  }
}

TEST(QwenCliTest, RequiresCheckpointInEveryMode) {
  for (auto options :
       {InferenceOptions(), TrainingOptions(), EmbeddingAlgebraOptions()}) {
    SCOPED_TRACE(options.mode);
    options.checkpoint.clear();
    ExpectInvalid(Validate(options), "--checkpoint");
  }
}

TEST(QwenCliTest, FiltersEveryFlagEvenIfItsValueIsEmptyOrDefault) {
  struct Case {
    absl::string_view name;
    bool inference;
    bool training;
    bool algebra;
  };
  const Case cases[] = {
      {"mode", true, true, true},
      {"checkpoint", true, true, true},
      {"prompt", true, false, false},
      {"max_new_tokens", true, false, false},
      {"context_length", true, false, false},
      {"raw_prompt", true, false, false},
      {"thinking", true, false, false},
      {"text", false, true, false},
      {"sequence_length", false, true, false},
      {"batch_size", false, true, false},
      {"steps", false, true, false},
      {"switch_every", false, true, false},
      {"start_block", false, true, false},
      {"learning_rate", false, true, false},
      {"max_active_gib", false, true, false},
      {"resume_weights", false, true, false},
      {"save_weights", false, true, false},
      {"expression", false, false, true},
  };
  for (const Case& test : cases) {
    SCOPED_TRACE(test.name);
    const auto inference = Validate(InferenceOptions(), {test.name});
    const auto training = Validate(TrainingOptions(), {test.name});
    const auto algebra = Validate(EmbeddingAlgebraOptions(), {test.name});
    if (test.inference)
      EXPECT_TRUE(inference.ok()) << inference;
    else
      ExpectInvalid(inference, test.name);
    if (test.training)
      EXPECT_TRUE(training.ok()) << training;
    else
      ExpectInvalid(training, test.name);
    if (test.algebra)
      EXPECT_TRUE(algebra.ok()) << algebra;
    else
      ExpectInvalid(algebra, test.name);
  }
}

TEST(QwenCliTest, AcceptsCompleteFlagSets) {
  const auto inference = Validate(
      InferenceOptions(), {"mode", "checkpoint", "prompt", "max_new_tokens",
                           "context_length", "raw_prompt", "thinking"});
  EXPECT_TRUE(inference.ok()) << inference;
  const auto training =
      Validate(TrainingOptions(),
               {"mode", "checkpoint", "text", "sequence_length", "batch_size",
                "steps", "switch_every", "start_block", "learning_rate",
                "max_active_gib", "resume_weights", "save_weights"});
  EXPECT_TRUE(training.ok()) << training;
  const auto algebra =
      Validate(EmbeddingAlgebraOptions(), {"mode", "checkpoint", "expression"});
  EXPECT_TRUE(algebra.ok()) << algebra;
}

TEST(QwenCliTest, UnknownPolicyNamesAreInternalErrors) {
  for (const auto& options :
       {InferenceOptions(), TrainingOptions(), EmbeddingAlgebraOptions()})
    for (absl::string_view name : {"unregistered", "--prompt", ""}) {
      SCOPED_TRACE(options.mode);
      SCOPED_TRACE(name);
      const auto status = Validate(options, {name});
      EXPECT_EQ(status.code(), absl::StatusCode::kInternal) << status;
      EXPECT_NE(status.message().find("missing mode policy"),
                absl::string_view::npos)
          << status;
    }
}

TEST(QwenCliTest, IgnoresUnselectedModesValuesWhenNotExplicit) {
  auto inference = InferenceOptions();
  inference.learning_rate = std::numeric_limits<double>::quiet_NaN();
  inference.max_active_gib = std::numeric_limits<double>::infinity();
  inference.start_block = -200;
  EXPECT_TRUE(Validate(inference).ok());
  auto training = TrainingOptions();
  training.max_new_tokens = -1;
  training.context_length = -1;
  training.raw_prompt = true;
  training.thinking = true;
  EXPECT_TRUE(Validate(training).ok());
  auto algebra = EmbeddingAlgebraOptions();
  algebra.max_new_tokens = -1;
  algebra.context_length = -1;
  algebra.raw_prompt = true;
  algebra.thinking = true;
  algebra.learning_rate = std::numeric_limits<double>::quiet_NaN();
  algebra.max_active_gib = std::numeric_limits<double>::infinity();
  algebra.start_block = -200;
  EXPECT_TRUE(Validate(algebra).ok());
}

TEST(QwenCliTest, EmbeddingAlgebraAllowsInteractiveOrNonemptyExpression) {
  auto options = EmbeddingAlgebraOptions();
  options.expression.clear();
  EXPECT_TRUE(Validate(options, {"mode", "checkpoint"}).ok());
  for (absl::string_view expression : {"", " ", "\t\r\n"}) {
    options.expression = std::string(expression);
    ExpectInvalid(Validate(options, {"expression"}), "--expression");
  }
  for (absl::string_view expression :
       {"king", "king - queen + boy", "raw king - queen + boy"}) {
    options.expression = std::string(expression);
    EXPECT_TRUE(Validate(options, {"expression"}).ok());
  }
}

TEST(QwenCliTest, RejectsNonpositiveInferenceLimits) {
  struct Case {
    absl::string_view name;
    int CommandLineOptions::*member;
  };
  for (const Case& test :
       {Case{"max_new_tokens", &CommandLineOptions::max_new_tokens},
        Case{"context_length", &CommandLineOptions::context_length}})
    for (int value : {std::numeric_limits<int>::min(), -1, 0}) {
      SCOPED_TRACE(test.name);
      SCOPED_TRACE(value);
      auto options = InferenceOptions();
      options.*test.member = value;
      ExpectInvalid(Validate(options), test.name);
    }
}

TEST(QwenCliTest, AcceptsPositiveInferenceLimits) {
  for (int value : {1, std::numeric_limits<int>::max()}) {
    auto options = InferenceOptions();
    options.max_new_tokens = value;
    options.context_length = value;
    EXPECT_TRUE(Validate(options).ok());
  }
}

TEST(QwenCliTest, ThinkingRequiresChatTemplate) {
  for (bool raw : {false, true})
    for (bool thinking : {false, true}) {
      auto options = InferenceOptions();
      options.raw_prompt = raw;
      options.thinking = thinking;
      if (raw && thinking)
        ExpectInvalid(Validate(options), "--thinking");
      else
        EXPECT_TRUE(Validate(options).ok());
    }
}

TEST(QwenCliTest, RestrictsSequenceLength) {
  for (int value : {-1, 0, 129, std::numeric_limits<int>::max()}) {
    auto options = TrainingOptions();
    options.sequence_length = value;
    ExpectInvalid(Validate(options), "--sequence_length");
  }
  for (int value : {1, 128}) {
    auto options = TrainingOptions();
    options.sequence_length = value;
    EXPECT_TRUE(Validate(options).ok());
  }
}

TEST(QwenCliTest, RequiresSingleSampleBatches) {
  for (int value : {-1, 0, 2, std::numeric_limits<int>::max()}) {
    auto options = TrainingOptions();
    options.batch_size = value;
    ExpectInvalid(Validate(options), "--batch_size");
  }
}

TEST(QwenCliTest, RequiresPositiveUpdateCounts) {
  struct Case {
    absl::string_view name;
    int CommandLineOptions::*member;
  };
  for (const Case& test :
       {Case{"steps", &CommandLineOptions::steps},
        Case{"switch_every", &CommandLineOptions::switch_every}})
    for (int value : {std::numeric_limits<int>::min(), -1, 0, 1,
                      std::numeric_limits<int>::max()}) {
      SCOPED_TRACE(test.name);
      SCOPED_TRACE(value);
      auto options = TrainingOptions();
      options.*test.member = value;
      if (value <= 0)
        ExpectInvalid(Validate(options), test.name);
      else
        EXPECT_TRUE(Validate(options).ok());
    }
}

TEST(QwenCliTest, ValidatesStartBlockSentinel) {
  for (int value : {std::numeric_limits<int>::min(), -2, -1, 0, 65}) {
    auto options = TrainingOptions();
    options.start_block = value;
    if (value < -1)
      ExpectInvalid(Validate(options), "--start_block");
    else
      EXPECT_TRUE(Validate(options).ok());
  }
  // The upper bound depends on the checkpoint, which this CPU helper does not
  // open. The model-loading path checks it against the decoder block count.
  auto options = TrainingOptions();
  options.start_block = std::numeric_limits<int>::max();
  EXPECT_TRUE(Validate(options).ok());
}

TEST(QwenCliTest, RequiresFiniteRepresentableLearningRate) {
  for (double value :
       {-1., 0., 1.00001, std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::min(),
        std::numeric_limits<double>::denorm_min()}) {
    SCOPED_TRACE(value);
    auto options = TrainingOptions();
    options.learning_rate = value;
    ExpectInvalid(Validate(options), "--learning_rate");
  }
  for (double value : {double{std::numeric_limits<float>::min()}, 1e-5, 1.}) {
    auto options = TrainingOptions();
    options.learning_rate = value;
    EXPECT_TRUE(Validate(options).ok());
  }
}

TEST(QwenCliTest, RequiresBoundedFiniteMemoryCap) {
  for (double value :
       {-1., 1024.00001, std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity()}) {
    SCOPED_TRACE(value);
    auto options = TrainingOptions();
    options.max_active_gib = value;
    ExpectInvalid(Validate(options), "--max_active_gib");
  }
  for (double value : {0., 1., 1024.}) {
    auto options = TrainingOptions();
    options.max_active_gib = value;
    EXPECT_TRUE(Validate(options).ok());
  }
}

}  // namespace
}  // namespace pluto::llm::qwen
