#include "src/llm/experiments/memorize_general_facts/memorize_general_facts_cli.h"

#include <initializer_list>
#include <limits>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

namespace pluto::llm::memorize_general_facts {
namespace {

absl::Status Validate(Mode mode, std::initializer_list<absl::string_view> flags,
                      absl::string_view infer_checkpoint = "",
                      absl::string_view verify_checkpoint = "",
                      absl::string_view tokenizer = "/tokenizer",
                      absl::string_view checkpoint_dir = "/checkpoints",
                      absl::string_view puzzle_checkpoint = "/puzzle",
                      bool train_mlp = false, bool train_stacked_mlp = false,
                      bool train_mlp_transformer = false) {
  return ValidateModeFlags(
      mode, absl::MakeConstSpan(flags.begin(), flags.size()), tokenizer,
      checkpoint_dir, infer_checkpoint, verify_checkpoint, puzzle_checkpoint,
      train_mlp, train_stacked_mlp, train_mlp_transformer);
}

void ExpectInvalid(const absl::Status& status, absl::string_view diagnostic) {
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  EXPECT_NE(status.message().find(diagnostic), absl::string_view::npos)
      << status;
}

CommandLineOptions TrainingOptions() {
  CommandLineOptions options;
  options.mode = "train_model";
  options.tokenizer = "/tokenizer";
  options.checkpoint_dir = "/checkpoints";
  options.corpus = "/corpus";
  options.output_dir = "/output";
  options.context_length = 27;
  options.batch_size = 1;
  options.eval_every = 1;
  options.checkpoint_every = 1;
  options.learning_rate = 0.001;
  return options;
}

CommandLineOptions GenerationOptions() {
  CommandLineOptions options;
  options.mode = "infer_model";
  options.tokenizer = "/tokenizer";
  options.infer_checkpoint = "/model";
  options.context_length = 27;
  return options;
}

CommandLineOptions VerificationOptions() {
  CommandLineOptions options;
  options.mode = "infer_model";
  options.tokenizer = "/tokenizer";
  options.verify_checkpoint = "/model";
  options.corpus = "/corpus";
  options.output_dir = "/output";
  options.context_length = 27;
  options.batch_size = 1;
  return options;
}

CommandLineOptions PuzzleCommandLineOptions(
    bool train_mlp = false, bool train_stacked_mlp = false,
    bool train_mlp_transformer = false) {
  auto options = TrainingOptions();
  options.mode = "puzzle";
  options.checkpoint_dir.clear();
  options.puzzle_checkpoint = "/puzzle";
  options.layers = 4;
  options.model_width = 10;
  options.mlp_width = 150;
  options.train_mlp = train_mlp;
  options.train_stacked_mlp = train_stacked_mlp;
  options.train_mlp_transformer = train_mlp_transformer;
  return ResolveModeDefaults(options, {});
}

absl::Status ValidateOptions(
    const CommandLineOptions& options,
    std::initializer_list<absl::string_view> flags = {}) {
  return ParseAndValidateRunMode(
             options, absl::MakeConstSpan(flags.begin(), flags.size()))
      .status();
}

TEST(MemorizeGeneralFactsCliTest, ParsesOnlyExactNamedModes) {
  struct Case {
    absl::string_view name;
    Mode mode;
  };
  for (const auto& test : {Case{"train_model", Mode::kTrainModel},
                           Case{"infer_model", Mode::kInferModel},
                           Case{"puzzle", Mode::kPuzzle}}) {
    SCOPED_TRACE(test.name);
    const auto parsed = ParseMode(test.name);
    ASSERT_TRUE(parsed.ok()) << parsed.status();
    EXPECT_EQ(*parsed, test.mode);
    EXPECT_EQ(ModeName(test.mode), test.name);
  }
  for (absl::string_view invalid :
       {"", " ", "train", "infer", "train_sae", "infer_sae", "TRAIN_MODEL",
        "Infer_model", " train_model", "train_model ", "infer_model\n",
        "train_model,infer_model"}) {
    SCOPED_TRACE(invalid);
    ExpectInvalid(ParseMode(invalid).status(), "--mode");
  }
}

TEST(MemorizeGeneralFactsCliTest, RejectsInvalidEnumValues) {
  for (int invalid : {-1, 3, 127}) {
    SCOPED_TRACE(invalid);
    const Mode mode = static_cast<Mode>(invalid);
    EXPECT_EQ(ModeName(mode), "unknown");
    ExpectInvalid(Validate(mode, {}), "--mode");
    ExpectInvalid(Validate(mode, {"mode"}), "--mode");
  }
}

TEST(MemorizeGeneralFactsCliTest, FiltersEveryFlagByExecutionPath) {
  struct Case {
    absl::string_view flag;
    bool train;
    bool generate;
    bool verify;
    bool capture;
    bool train_mlp;
  };
  const Case cases[] = {
      {"mode", true, true, true, true, true},
      {"tokenizer", true, true, true, true, true},
      {"layers", true, true, true, true, true},
      {"model_width", true, true, true, true, true},
      {"attention_heads", true, true, true, true, true},
      {"feed_forward_width", true, true, true, true, true},
      {"context_length", true, true, true, true, true},
      {"compact_vocabulary", true, true, true, true, true},
      {"seed", true, true, true, false, true},
      {"checkpoint_dir", true, false, false, false, false},
      {"search", true, false, false, false, false},
      {"steps", true, false, false, false, true},
      {"eval_every", true, false, false, false, true},
      {"checkpoint_every", true, false, false, false, false},
      {"learning_rate", true, false, false, false, true},
      {"warmup_steps", true, false, false, false, false},
      {"training_seconds", true, false, false, false, false},
      {"corpus", true, false, true, true, true},
      {"output_dir", true, false, true, true, true},
      {"batch_size", true, false, true, true, true},
      {"infer_checkpoint", false, true, false, false, false},
      {"prompt", false, true, false, false, false},
      {"generation_tokens", false, true, false, false, false},
      {"print_attention_probs", false, true, false, false, false},
      {"verify_checkpoint", false, false, true, false, false},
      {"puzzle_checkpoint", false, false, false, true, true},
      {"train_mlp", false, false, false, true, true},
      {"train_stacked_mlp", false, false, false, true, true},
      {"train_mlp_transformer", false, false, false, true, true},
      {"mlp_width", false, false, false, false, true},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(test.flag);
    const absl::Status statuses[] = {
        Validate(Mode::kTrainModel, {test.flag}),
        Validate(Mode::kInferModel, {test.flag}, "/model"),
        Validate(Mode::kInferModel, {test.flag}, "", "/model"),
        Validate(Mode::kPuzzle, {test.flag}),
        Validate(Mode::kPuzzle, {test.flag}, "", "", "/tokenizer", "",
                 "/puzzle", true),
        Validate(Mode::kPuzzle, {test.flag}, "", "", "/tokenizer", "",
                 "/puzzle", false, true),
        Validate(Mode::kPuzzle, {test.flag}, "", "", "/tokenizer", "",
                 "/puzzle", false, false, true),
    };
    const bool allowed[] = {test.train,
                            test.generate,
                            test.verify,
                            test.capture,
                            test.train_mlp,
                            test.train_mlp && test.flag != "mlp_width",
                            test.train_mlp && test.flag != "mlp_width"};
    const absl::string_view paths[] = {"train",
                                       "generate",
                                       "verify",
                                       "capture",
                                       "train_mlp",
                                       "train_stacked_mlp",
                                       "train_mlp_transformer"};
    for (int i = 0; i < 7; ++i) {
      SCOPED_TRACE(paths[i]);
      if (allowed[i])
        EXPECT_TRUE(statuses[i].ok()) << statuses[i];
      else
        ExpectInvalid(statuses[i], test.flag);
    }
  }
}

TEST(MemorizeGeneralFactsCliTest, AcceptsCompleteFlagSetsForEachPath) {
  EXPECT_TRUE(Validate(Mode::kTrainModel, {"mode",
                                           "tokenizer",
                                           "layers",
                                           "model_width",
                                           "attention_heads",
                                           "feed_forward_width",
                                           "context_length",
                                           "compact_vocabulary",
                                           "seed",
                                           "checkpoint_dir",
                                           "search",
                                           "steps",
                                           "eval_every",
                                           "checkpoint_every",
                                           "learning_rate",
                                           "warmup_steps",
                                           "training_seconds",
                                           "corpus",
                                           "output_dir",
                                           "batch_size"})
                  .ok());
  EXPECT_TRUE(
      Validate(Mode::kInferModel,
               {"mode", "tokenizer", "layers", "model_width", "attention_heads",
                "feed_forward_width", "context_length", "compact_vocabulary",
                "seed", "infer_checkpoint", "prompt", "generation_tokens",
                "print_attention_probs"},
               "/model", "", "/tokenizer", "")
          .ok());
  EXPECT_TRUE(
      Validate(
          Mode::kInferModel,
          {"mode", "tokenizer", "layers", "model_width", "attention_heads",
           "feed_forward_width", "context_length", "compact_vocabulary", "seed",
           "verify_checkpoint", "corpus", "output_dir", "batch_size"},
          "", "/model", "/tokenizer", "")
          .ok());
}

TEST(MemorizeGeneralFactsCliTest, RequiresTokenizerInEveryPath) {
  ExpectInvalid(Validate(Mode::kPuzzle, {}, "", "", ""), "--tokenizer");
  ExpectInvalid(Validate(Mode::kPuzzle, {"tokenizer"}, "", "", ""),
                "--tokenizer");
  ExpectInvalid(Validate(Mode::kTrainModel, {}, "", "", ""), "--tokenizer");
  ExpectInvalid(Validate(Mode::kInferModel, {}, "/model", "", ""),
                "--tokenizer");
  ExpectInvalid(Validate(Mode::kInferModel, {}, "", "/model", ""),
                "--tokenizer");
  ExpectInvalid(Validate(Mode::kTrainModel, {"tokenizer"}, "", "", ""),
                "--tokenizer");
  ExpectInvalid(Validate(Mode::kInferModel, {"tokenizer"}, "/model", "", ""),
                "--tokenizer");
  ExpectInvalid(Validate(Mode::kInferModel, {"tokenizer"}, "", "/model", ""),
                "--tokenizer");
}

TEST(MemorizeGeneralFactsCliTest, RequiresCheckpointDirectoryOnlyForTraining) {
  ExpectInvalid(Validate(Mode::kTrainModel, {}, "", "", "/tokenizer", ""),
                "--checkpoint_dir");
  ExpectInvalid(
      Validate(Mode::kTrainModel, {"checkpoint_dir"}, "", "", "/tokenizer", ""),
      "--checkpoint_dir");
  EXPECT_TRUE(
      Validate(Mode::kInferModel, {}, "/model", "", "/tokenizer", "").ok());
  EXPECT_TRUE(
      Validate(Mode::kInferModel, {}, "", "/model", "/tokenizer", "").ok());
  // Explicitly supplying even an empty training-only directory is rejected.
  ExpectInvalid(Validate(Mode::kInferModel, {"checkpoint_dir"}, "/model", "",
                         "/tokenizer", ""),
                "--checkpoint_dir");
  ExpectInvalid(Validate(Mode::kInferModel, {"checkpoint_dir"}, "", "/model",
                         "/tokenizer", ""),
                "--checkpoint_dir");
}

TEST(MemorizeGeneralFactsCliTest, InferenceRequiresANonemptyCheckpoint) {
  ExpectInvalid(Validate(Mode::kInferModel, {}), "exactly one nonempty");
  ExpectInvalid(Validate(Mode::kInferModel, {"infer_checkpoint"}),
                "exactly one nonempty");
  ExpectInvalid(Validate(Mode::kInferModel, {"verify_checkpoint"}),
                "exactly one nonempty");
  EXPECT_TRUE(Validate(Mode::kTrainModel, {}).ok());
  EXPECT_TRUE(Validate(Mode::kInferModel, {}, "/model").ok());
  EXPECT_TRUE(Validate(Mode::kInferModel, {}, "", "/model").ok());
}

TEST(MemorizeGeneralFactsCliTest, CheckpointSelectorsAreMutuallyExclusive) {
  for (absl::string_view infer : {"", "/generation"}) {
    for (absl::string_view verify : {"", "/verification"}) {
      SCOPED_TRACE(infer);
      SCOPED_TRACE(verify);
      for (bool reverse : {false, true}) {
        const auto status =
            reverse ? Validate(Mode::kInferModel,
                               {"verify_checkpoint", "infer_checkpoint"}, infer,
                               verify)
                    : Validate(Mode::kInferModel,
                               {"infer_checkpoint", "verify_checkpoint"}, infer,
                               verify);
        ExpectInvalid(status, "mutually exclusive");
      }
    }
  }
  ExpectInvalid(Validate(Mode::kInferModel, {}, "/generation", "/verification"),
                "mutually exclusive");
  ExpectInvalid(Validate(Mode::kInferModel, {"verify_checkpoint"}, "/model"),
                "mutually exclusive");
  ExpectInvalid(Validate(Mode::kInferModel, {"infer_checkpoint"}, "", "/model"),
                "mutually exclusive");
}

TEST(MemorizeGeneralFactsCliTest, ExplicitPresenceControlsFiltering) {
  // The caller records --search=false and --prompt= as explicitly supplied.
  // Filtering cannot rely on their values differing from flag defaults.
  ExpectInvalid(Validate(Mode::kInferModel, {"search"}, "/model"), "--search");
  ExpectInvalid(Validate(Mode::kInferModel, {"search"}, "", "/model"),
                "--search");
  ExpectInvalid(Validate(Mode::kTrainModel, {"prompt"}), "--prompt");
  ExpectInvalid(Validate(Mode::kInferModel, {"prompt"}, "", "/model"),
                "--prompt");
  EXPECT_TRUE(Validate(Mode::kInferModel, {"prompt"}, "/model").ok());
  ExpectInvalid(Validate(Mode::kTrainModel, {"infer_checkpoint"}),
                "--infer_checkpoint");
  ExpectInvalid(Validate(Mode::kTrainModel, {"verify_checkpoint"}),
                "--verify_checkpoint");
}

TEST(MemorizeGeneralFactsCliTest, RejectsFlagsMissingFromThePolicy) {
  for (absl::string_view unknown : {"new_flag", "", "--steps"}) {
    SCOPED_TRACE(unknown);
    const absl::Status statuses[] = {
        Validate(Mode::kTrainModel, {unknown}),
        Validate(Mode::kInferModel, {unknown}, "/model"),
        Validate(Mode::kInferModel, {unknown}, "", "/model"),
        Validate(Mode::kPuzzle, {unknown}),
        Validate(Mode::kPuzzle, {unknown}, "", "", "/tokenizer", "", "/puzzle",
                 true),
        Validate(Mode::kPuzzle, {unknown}, "", "", "/tokenizer", "", "/puzzle",
                 false, true),
        Validate(Mode::kPuzzle, {unknown}, "", "", "/tokenizer", "", "/puzzle",
                 false, false, true),
    };
    for (const auto& status : statuses) {
      EXPECT_EQ(status.code(), absl::StatusCode::kInternal) << status;
      EXPECT_NE(status.message().find("missing mode policy for --"),
                absl::string_view::npos);
    }
  }
}

TEST(MemorizeGeneralFactsCliTest, ValidatesAndReturnsEachExecutionMode) {
  for (const auto& options :
       {TrainingOptions(), GenerationOptions(), VerificationOptions(),
        PuzzleCommandLineOptions(), PuzzleCommandLineOptions(true),
        PuzzleCommandLineOptions(false, true),
        PuzzleCommandLineOptions(false, false, true)}) {
    SCOPED_TRACE(options.mode);
    SCOPED_TRACE(options.verify_checkpoint);
    const auto mode = ParseAndValidateRunMode(options, {});
    ASSERT_TRUE(mode.ok()) << mode.status();
    EXPECT_EQ(*mode, options.mode == "train_model" ? Mode::kTrainModel
                     : options.mode == "puzzle"    ? Mode::kPuzzle
                                                   : Mode::kInferModel);
  }
}

TEST(MemorizeGeneralFactsCliTest, GenerationRequiresNonnegativeTokenCount) {
  auto options = GenerationOptions();
  for (int value : {std::numeric_limits<int>::min(), -1, 0, 1,
                    std::numeric_limits<int>::max()}) {
    SCOPED_TRACE(value);
    options.generation_tokens = value;
    const auto status = ValidateOptions(options);
    if (value < 0)
      ExpectInvalid(status, "--generation_tokens must be nonnegative");
    else
      EXPECT_TRUE(status.ok()) << status;
  }
}

TEST(MemorizeGeneralFactsCliTest, EmptyPromptIsAllowedOnlyWhenOmitted) {
  auto options = GenerationOptions();
  EXPECT_TRUE(ValidateOptions(options).ok());
  ExpectInvalid(ValidateOptions(options, {"prompt"}),
                "--prompt must be nonempty when supplied");
  options.prompt = "A fact about Earth";
  EXPECT_TRUE(ValidateOptions(options, {"prompt"}).ok());
}

TEST(MemorizeGeneralFactsCliTest,
     CorpusPathsAreRequiredForTrainingAndVerification) {
  struct Case {
    std::string CommandLineOptions::*field;
    absl::string_view diagnostic;
  };
  const Case cases[] = {
      {&CommandLineOptions::corpus, "--corpus must be nonempty"},
      {&CommandLineOptions::output_dir, "--output_dir must be nonempty"},
  };
  for (const auto& valid :
       {TrainingOptions(), VerificationOptions(), PuzzleCommandLineOptions()}) {
    SCOPED_TRACE(valid.mode);
    for (const auto& test : cases) {
      auto options = valid;
      (options.*test.field).clear();
      ExpectInvalid(ValidateOptions(options), test.diagnostic);
    }
  }
}

TEST(MemorizeGeneralFactsCliTest, CorpusPathsRequirePositiveBatchSize) {
  for (auto options :
       {TrainingOptions(), VerificationOptions(), PuzzleCommandLineOptions()}) {
    SCOPED_TRACE(options.mode);
    for (int value :
         {std::numeric_limits<int>::min(), -1, 0, 1,
          std::numeric_limits<int>::max() / options.context_length}) {
      SCOPED_TRACE(value);
      options.batch_size = value;
      const auto status = ValidateOptions(options);
      if (value <= 0)
        ExpectInvalid(status, "--batch_size must be positive");
      else
        EXPECT_TRUE(status.ok()) << status;
    }
  }
}

TEST(MemorizeGeneralFactsCliTest, ContextLengthMustBePositiveInEveryPath) {
  for (auto options : {TrainingOptions(), GenerationOptions(),
                       VerificationOptions(), PuzzleCommandLineOptions()}) {
    SCOPED_TRACE(options.mode);
    SCOPED_TRACE(options.verify_checkpoint);
    for (int context_length : {std::numeric_limits<int>::min(), -1, 0}) {
      SCOPED_TRACE(context_length);
      options.context_length = context_length;
      ExpectInvalid(ValidateOptions(options, {"context_length"}),
                    "--context_length must be positive");
    }
    options.context_length = 27;
    EXPECT_TRUE(ValidateOptions(options, {"context_length"}).ok());
  }
}

TEST(MemorizeGeneralFactsCliTest, CorpusContextMustFitTheFiveTokenPrompt) {
  for (auto options :
       {TrainingOptions(), VerificationOptions(), PuzzleCommandLineOptions()}) {
    options.context_length = 4;
    ExpectInvalid(ValidateOptions(options),
                  "--context_length must be at least 5");
    options.context_length = 5;
    EXPECT_TRUE(ValidateOptions(options).ok());
  }
  auto options = GenerationOptions();
  options.context_length = 1;
  EXPECT_TRUE(ValidateOptions(options).ok());
}

TEST(MemorizeGeneralFactsCliTest, CorpusBatchTokenCountCannotOverflow) {
  for (auto options :
       {TrainingOptions(), VerificationOptions(), PuzzleCommandLineOptions()}) {
    for (int context_length : {27, std::numeric_limits<int>::max()}) {
      SCOPED_TRACE(context_length);
      options.context_length = context_length;
      options.batch_size = std::numeric_limits<int>::max() / context_length;
      EXPECT_TRUE(ValidateOptions(options).ok());
      ++options.batch_size;
      ExpectInvalid(ValidateOptions(options),
                    "--batch_size * --context_length exceeds");
    }
  }
  auto options = GenerationOptions();
  options.batch_size = std::numeric_limits<int>::max();
  EXPECT_TRUE(ValidateOptions(options).ok());
}

TEST(MemorizeGeneralFactsCliTest, ValidatesTrainingScheduleBoundaries) {
  struct Case {
    int CommandLineOptions::*field;
    absl::string_view diagnostic;
    bool allows_zero;
  };
  const Case cases[] = {
      {&CommandLineOptions::steps, "--steps must be nonnegative", true},
      {&CommandLineOptions::eval_every, "--eval_every must be positive", false},
      {&CommandLineOptions::checkpoint_every,
       "--checkpoint_every must be positive", false},
      {&CommandLineOptions::warmup_steps, "--warmup_steps must be nonnegative",
       true},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(test.diagnostic);
    auto options = TrainingOptions();
    for (int value : {std::numeric_limits<int>::min(), -1, 0, 1,
                      std::numeric_limits<int>::max()}) {
      SCOPED_TRACE(value);
      options.*test.field = value;
      const auto status = ValidateOptions(options);
      if (value < 0 || (value == 0 && !test.allows_zero))
        ExpectInvalid(status, test.diagnostic);
      else
        EXPECT_TRUE(status.ok()) << status;
    }
  }
}

TEST(MemorizeGeneralFactsCliTest, LearningRateMustBeFiniteAndPositive) {
  auto options = TrainingOptions();
  for (double value : {-1.0, 0.0, std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
    SCOPED_TRACE(value);
    options.learning_rate = value;
    ExpectInvalid(ValidateOptions(options),
                  "--learning_rate must be finite and positive");
  }
  for (double value : {std::numeric_limits<double>::denorm_min(),
                       std::numeric_limits<double>::min(), 1.0,
                       std::numeric_limits<double>::max()}) {
    SCOPED_TRACE(value);
    options.learning_rate = value;
    const auto status = ValidateOptions(options);
    EXPECT_TRUE(status.ok()) << status;
  }
}

TEST(MemorizeGeneralFactsCliTest, TrainingSecondsMustBeFiniteAndNonnegative) {
  auto options = TrainingOptions();
  for (double value : {-1.0, std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
    SCOPED_TRACE(value);
    options.training_seconds = value;
    ExpectInvalid(ValidateOptions(options),
                  "--training_seconds must be finite and nonnegative");
  }
  for (double value : {0.0, std::numeric_limits<double>::denorm_min(), 1.0,
                       std::numeric_limits<double>::max()}) {
    SCOPED_TRACE(value);
    options.training_seconds = value;
    const auto status = ValidateOptions(options);
    EXPECT_TRUE(status.ok()) << status;
  }
}

TEST(MemorizeGeneralFactsCliTest,
     IgnoresIrrelevantValuesUnlessExplicitlySupplied) {
  for (auto options : {GenerationOptions(), VerificationOptions()}) {
    SCOPED_TRACE(options.verify_checkpoint);
    options.steps = -1;
    options.eval_every = -1;
    options.checkpoint_every = -1;
    options.warmup_steps = -1;
    options.learning_rate = std::numeric_limits<double>::quiet_NaN();
    options.training_seconds = std::numeric_limits<double>::infinity();
    if (options.verify_checkpoint.empty())
      options.batch_size = -1;
    else
      options.generation_tokens = -1;
    const auto status = ValidateOptions(options);
    EXPECT_TRUE(status.ok()) << status;
    ExpectInvalid(ValidateOptions(options, {"steps"}), "--steps is not valid");
  }
  auto options = TrainingOptions();
  options.generation_tokens = -1;
  EXPECT_TRUE(ValidateOptions(options).ok());
  ExpectInvalid(ValidateOptions(options, {"generation_tokens"}),
                "--generation_tokens is not valid");
}

TEST(MemorizeGeneralFactsCliTest, DelegatesPolicyChecksBeforeValueValidation) {
  auto options = TrainingOptions();
  options.batch_size = 0;
  ExpectInvalid(ValidateOptions(options, {"prompt"}), "--prompt is not valid");
  const auto unknown = ValidateOptions(options, {"unknown_flag"});
  EXPECT_EQ(unknown.code(), absl::StatusCode::kInternal) << unknown;

  options.tokenizer.clear();
  ExpectInvalid(ValidateOptions(options), "--tokenizer");
  options.tokenizer = "/tokenizer";
  options.checkpoint_dir.clear();
  ExpectInvalid(ValidateOptions(options), "--checkpoint_dir");

  options = GenerationOptions();
  options.generation_tokens = -1;
  options.verify_checkpoint = "/verification";
  ExpectInvalid(ValidateOptions(options), "mutually exclusive");
  options.infer_checkpoint.clear();
  options.verify_checkpoint.clear();
  ExpectInvalid(ValidateOptions(options), "exactly one nonempty");
}

TEST(MemorizeGeneralFactsCliTest, ParsesModeBeforePolicyOrValues) {
  CommandLineOptions options;
  for (const char* mode : {"", "train", "INFER_MODEL"}) {
    SCOPED_TRACE(mode);
    options.mode = mode;
    ExpectInvalid(ValidateOptions(options, {"unknown_flag"}), "--mode");
  }
}

TEST(MemorizeGeneralFactsCliTest, PuzzleRequiresItsCheckpoint) {
  auto options = PuzzleCommandLineOptions();
  options.puzzle_checkpoint.clear();
  ExpectInvalid(ValidateOptions(options), "--puzzle_checkpoint is required");
  ExpectInvalid(ValidateOptions(options, {"puzzle_checkpoint"}),
                "--puzzle_checkpoint is required");
  options.puzzle_checkpoint = "/source";
  EXPECT_TRUE(ValidateOptions(options, {"puzzle_checkpoint"}).ok());
}

TEST(MemorizeGeneralFactsCliTest, PuzzleTrainingSelectorsAreExclusiveWhenTrue) {
  for (bool train_mlp : {false, true}) {
    for (bool train_stacked_mlp : {false, true}) {
      for (bool train_mlp_transformer : {false, true}) {
        SCOPED_TRACE(train_mlp);
        SCOPED_TRACE(train_stacked_mlp);
        SCOPED_TRACE(train_mlp_transformer);
        const auto options = PuzzleCommandLineOptions(
            train_mlp, train_stacked_mlp, train_mlp_transformer);
        const absl::Status statuses[] = {
            ValidateOptions(options),
            ValidateOptions(options, {"train_mlp", "train_stacked_mlp",
                                      "train_mlp_transformer"}),
        };
        for (const auto& status : statuses)
          if (train_mlp + train_stacked_mlp + train_mlp_transformer > 1)
            ExpectInvalid(status, "mutually exclusive");
          else
            // Explicit false selectors are allowed with the enabled selector.
            EXPECT_TRUE(status.ok()) << status;
      }
    }
  }
}

TEST(MemorizeGeneralFactsCliTest, MlpTransformerSelectorIsPuzzleOnly) {
  for (auto options :
       {TrainingOptions(), GenerationOptions(), VerificationOptions()}) {
    for (bool enabled : {false, true}) {
      options.train_mlp_transformer = enabled;
      ExpectInvalid(ValidateOptions(options, {"train_mlp_transformer"}),
                    "--train_mlp_transformer is not valid");
    }
  }
}

TEST(MemorizeGeneralFactsCliTest, MlpTransformerRequiresExactlyFourBlocks) {
  auto options = PuzzleCommandLineOptions(false, false, true);
  for (int layers : {std::numeric_limits<int>::min(), -1, 0, 1, 2, 3, 5,
                     std::numeric_limits<int>::max()}) {
    SCOPED_TRACE(layers);
    options.layers = layers;
    ExpectInvalid(ValidateOptions(options),
                  "--layers must be 4 with --train_mlp_transformer");
  }
  options.layers = 4;
  EXPECT_TRUE(ValidateOptions(options, {"layers"}).ok());
}

TEST(MemorizeGeneralFactsCliTest, MlpTransformerUsesSourceWidths) {
  auto options = PuzzleCommandLineOptions(false, false, true);
  for (int width : {-1, 0, 1, 10, 20}) {
    SCOPED_TRACE(width);
    options.model_width = width;
    const auto status = ValidateOptions(
        options, {"model_width", "feed_forward_width", "attention_heads"});
    if (width <= 0)
      ExpectInvalid(status, "--model_width must be positive");
    else
      EXPECT_TRUE(status.ok()) << status;
  }
  for (int width : {-1, 0, 150, 300}) {
    SCOPED_TRACE(width);
    options.mlp_width = width;
    // Its MLPs use the source feed-forward width, not the single-MLP setting.
    EXPECT_TRUE(ValidateOptions(options).ok());
    ExpectInvalid(ValidateOptions(options, {"mlp_width"}),
                  "--mlp_width is not valid");
  }
}

TEST(MemorizeGeneralFactsCliTest, MlpTransformerAloneConsumesTrainingSettings) {
  const auto options = PuzzleCommandLineOptions(false, false, true);
  EXPECT_TRUE(
      ValidateOptions(options, {"train_mlp_transformer", "steps", "eval_every",
                                "learning_rate", "seed", "batch_size"})
          .ok());
}

TEST(MemorizeGeneralFactsCliTest, StackedPuzzleTrainingHasFixedWidths) {
  auto options = PuzzleCommandLineOptions(false, true);
  for (int width : {-1, 0, 1, 9, 11, std::numeric_limits<int>::max()}) {
    SCOPED_TRACE(width);
    options.model_width = width;
    ExpectInvalid(ValidateOptions(options), "--model_width must");
  }
  options.model_width = 10;
  EXPECT_TRUE(ValidateOptions(options).ok());
  for (int width : {-1, 0, 150, 300}) {
    SCOPED_TRACE(width);
    options.mlp_width = width;
    // The stacked path uses its fixed hidden width, ignoring omitted values.
    EXPECT_TRUE(ValidateOptions(options).ok());
    ExpectInvalid(ValidateOptions(options, {"mlp_width"}),
                  "--mlp_width is not valid");
  }
  options.train_stacked_mlp = false;
  options.train_mlp = true;
  options.model_width = 20;
  options.mlp_width = 300;
  EXPECT_TRUE(ValidateOptions(options, {"model_width", "mlp_width"}).ok());
}

TEST(MemorizeGeneralFactsCliTest, StackedPuzzleAloneConsumesTrainingSettings) {
  const auto options = PuzzleCommandLineOptions(false, true);
  EXPECT_TRUE(
      ValidateOptions(options, {"train_stacked_mlp", "steps", "eval_every",
                                "learning_rate", "seed", "batch_size"})
          .ok());
}

TEST(MemorizeGeneralFactsCliTest, PuzzleDefaultsOnlyReplaceOmittedFlags) {
  for (auto options :
       {PuzzleCommandLineOptions(true), PuzzleCommandLineOptions(false, true),
        PuzzleCommandLineOptions(false, false, true)}) {
    options.steps = 120000;
    options.eval_every = 256;
    options.learning_rate = 0.0012;
    options.seed = 1337;
    const auto defaults = ResolveModeDefaults(options, {});
    EXPECT_EQ(defaults.steps, 300000);
    EXPECT_EQ(defaults.eval_every, 1000);
    EXPECT_DOUBLE_EQ(defaults.learning_rate, 0.01);
    EXPECT_EQ(defaults.seed, 3);
    EXPECT_EQ(defaults.mlp_width, 150);
    EXPECT_EQ(defaults.batch_size, options.batch_size);

    // Explicit values equal to the old training defaults must still win.
    const absl::string_view flags[] = {"steps", "eval_every", "learning_rate",
                                       "seed"};
    const auto explicit_values = ResolveModeDefaults(options, flags);
    EXPECT_EQ(explicit_values.steps, options.steps);
    EXPECT_EQ(explicit_values.eval_every, options.eval_every);
    EXPECT_DOUBLE_EQ(explicit_values.learning_rate, options.learning_rate);
    EXPECT_EQ(explicit_values.seed, options.seed);
    for (absl::string_view flag : flags) {
      SCOPED_TRACE(flag);
      const absl::string_view explicit_flag[] = {flag};
      const auto resolved = ResolveModeDefaults(options, explicit_flag);
      EXPECT_EQ(resolved.steps, flag == "steps" ? options.steps : 300000);
      EXPECT_EQ(resolved.eval_every,
                flag == "eval_every" ? options.eval_every : 1000);
      EXPECT_DOUBLE_EQ(resolved.learning_rate,
                       flag == "learning_rate" ? options.learning_rate : 0.01);
      EXPECT_EQ(resolved.seed, flag == "seed" ? options.seed : 3);
    }
  }
}

TEST(MemorizeGeneralFactsCliTest, ResolvingPuzzleDefaultsPreservesOtherModes) {
  for (auto options :
       {TrainingOptions(), GenerationOptions(), VerificationOptions()}) {
    options.steps = 120000;
    options.eval_every = 256;
    options.learning_rate = 0.0012;
    options.seed = 1337;
    const auto resolved = ResolveModeDefaults(options, {});
    EXPECT_EQ(resolved.steps, options.steps);
    EXPECT_EQ(resolved.eval_every, options.eval_every);
    EXPECT_DOUBLE_EQ(resolved.learning_rate, options.learning_rate);
    EXPECT_EQ(resolved.seed, options.seed);
    EXPECT_EQ(resolved.mode, options.mode);
  }
}

TEST(MemorizeGeneralFactsCliTest,
     PuzzleDefaultsDoNotHideInvalidExplicitValues) {
  for (auto options :
       {PuzzleCommandLineOptions(true), PuzzleCommandLineOptions(false, true),
        PuzzleCommandLineOptions(false, false, true)}) {
    options.steps = -1;
    const absl::string_view flags[] = {"steps"};
    ExpectInvalid(
        ValidateOptions(ResolveModeDefaults(options, flags), {"steps"}),
        "--steps must be nonnegative");
  }
}

TEST(MemorizeGeneralFactsCliTest, PuzzleRequiresThreeBlocksAndPositiveWidths) {
  for (bool train_mlp : {false, true}) {
    auto options = PuzzleCommandLineOptions(train_mlp);
    for (int layers : {-1, 0, 1, 2}) {
      options.layers = layers;
      ExpectInvalid(ValidateOptions(options), "--layers must be at least 3");
    }
    options.layers = 3;
    EXPECT_TRUE(ValidateOptions(options).ok());
    for (int width : {-1, 0}) {
      options.model_width = width;
      ExpectInvalid(ValidateOptions(options), "--model_width must be positive");
    }
    options.model_width = 1;
    for (int width : {-1, 0, 1}) {
      options.mlp_width = width;
      if (train_mlp && width <= 0)
        ExpectInvalid(ValidateOptions(options), "--mlp_width must be positive");
      else
        EXPECT_TRUE(ValidateOptions(options).ok());
    }
  }
}

TEST(MemorizeGeneralFactsCliTest,
     PuzzleCaptureRejectsExplicitTrainingSettings) {
  auto options = PuzzleCommandLineOptions();
  options.steps = -1;
  options.eval_every = 0;
  options.mlp_width = 0;
  options.learning_rate = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(ValidateOptions(options).ok());
  // Presence, rather than truthiness or equality to defaults, controls policy.
  for (absl::string_view flag :
       {"steps", "eval_every", "learning_rate", "seed", "mlp_width"}) {
    SCOPED_TRACE(flag);
    ExpectInvalid(ValidateOptions(options, {flag}), flag);
  }
  EXPECT_TRUE(ValidateOptions(options, {"train_mlp", "train_stacked_mlp",
                                        "train_mlp_transformer", "batch_size"})
                  .ok());
}

TEST(MemorizeGeneralFactsCliTest, PuzzleTrainingValidatesItsSchedule) {
  for (auto options :
       {PuzzleCommandLineOptions(true), PuzzleCommandLineOptions(false, true),
        PuzzleCommandLineOptions(false, false, true)}) {
    options.steps = -1;
    ExpectInvalid(ValidateOptions(options), "--steps must be nonnegative");
    options.steps = 0;
    EXPECT_TRUE(ValidateOptions(options).ok());
    for (int value : {-1, 0}) {
      options.eval_every = value;
      ExpectInvalid(ValidateOptions(options), "--eval_every must be positive");
    }
    options.eval_every = 1;
    // These unrelated training settings are ignored unless supplied explicitly.
    options.checkpoint_every = -1;
    options.warmup_steps = -1;
    options.training_seconds = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(ValidateOptions(options).ok());
    for (absl::string_view flag :
         {"checkpoint_every", "warmup_steps", "training_seconds"})
      ExpectInvalid(ValidateOptions(options, {flag}), flag);
  }
}

TEST(MemorizeGeneralFactsCliTest, OnlyPuzzleTrainingRequiresNonnegativeSeed) {
  for (auto puzzle :
       {PuzzleCommandLineOptions(true), PuzzleCommandLineOptions(false, true),
        PuzzleCommandLineOptions(false, false, true)}) {
    for (int seed : {std::numeric_limits<int>::min(), -1, 0, 3,
                     std::numeric_limits<int>::max()}) {
      SCOPED_TRACE(seed);
      puzzle.seed = seed;
      if (seed < 0)
        ExpectInvalid(ValidateOptions(puzzle, {"seed"}),
                      "--seed must be nonnegative");
      else
        EXPECT_TRUE(ValidateOptions(puzzle, {"seed"}).ok());
    }
  }
  for (auto options :
       {TrainingOptions(), GenerationOptions(), VerificationOptions()}) {
    options.seed = -1;
    EXPECT_TRUE(ValidateOptions(options, {"seed"}).ok());
  }
}

TEST(MemorizeGeneralFactsCliTest, PuzzleLearningRateMustBeUsableAsFloat) {
  for (auto options :
       {PuzzleCommandLineOptions(true), PuzzleCommandLineOptions(false, true),
        PuzzleCommandLineOptions(false, false, true)}) {
    for (double value : {-1.0, 0.0, std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::max(),
                         std::numeric_limits<double>::denorm_min()}) {
      SCOPED_TRACE(value);
      options.learning_rate = value;
      ExpectInvalid(ValidateOptions(options), "--learning_rate must");
    }
    for (double value :
         {static_cast<double>(std::numeric_limits<float>::min()), 0.01, 1.0}) {
      options.learning_rate = value;
      EXPECT_TRUE(ValidateOptions(options).ok());
    }
  }
}

}  // namespace
}  // namespace pluto::llm::memorize_general_facts
