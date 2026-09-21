#include "src/llm/experiments/memorize_general_facts/memorize_general_facts_cli.h"

#include <initializer_list>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

namespace pluto::llm::memorize_general_facts {
namespace {

absl::Status Validate(Mode mode, std::initializer_list<absl::string_view> flags,
                      absl::string_view infer_checkpoint = "",
                      absl::string_view verify_checkpoint = "",
                      absl::string_view tokenizer = "/tokenizer",
                      absl::string_view checkpoint_dir = "/checkpoints") {
  return ValidateModeFlags(
      mode, absl::MakeConstSpan(flags.begin(), flags.size()), tokenizer,
      checkpoint_dir, infer_checkpoint, verify_checkpoint);
}

void ExpectInvalid(const absl::Status& status, absl::string_view diagnostic) {
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  EXPECT_NE(status.message().find(diagnostic), absl::string_view::npos)
      << status;
}

TEST(MemorizeGeneralFactsCliTest, ParsesOnlyExactNamedModes) {
  struct Case {
    absl::string_view name;
    Mode mode;
  };
  for (const auto& test : {Case{"train_model", Mode::kTrainModel},
                           Case{"infer_model", Mode::kInferModel}}) {
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
  for (int invalid : {-1, 2, 127}) {
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
  };
  const Case cases[] = {
      {"mode", true, true, true},
      {"tokenizer", true, true, true},
      {"layers", true, true, true},
      {"model_width", true, true, true},
      {"attention_heads", true, true, true},
      {"feed_forward_width", true, true, true},
      {"compact_vocabulary", true, true, true},
      {"seed", true, true, true},
      {"checkpoint_dir", true, false, false},
      {"search", true, false, false},
      {"steps", true, false, false},
      {"eval_every", true, false, false},
      {"checkpoint_every", true, false, false},
      {"learning_rate", true, false, false},
      {"warmup_steps", true, false, false},
      {"training_seconds", true, false, false},
      {"corpus", true, false, true},
      {"output_dir", true, false, true},
      {"batch_size", true, false, true},
      {"infer_checkpoint", false, true, false},
      {"prompt", false, true, false},
      {"generation_tokens", false, true, false},
      {"verify_checkpoint", false, false, true},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(test.flag);
    const absl::Status statuses[] = {
        Validate(Mode::kTrainModel, {test.flag}),
        Validate(Mode::kInferModel, {test.flag}, "/model"),
        Validate(Mode::kInferModel, {test.flag}, "", "/model"),
    };
    const bool allowed[] = {test.train, test.generate, test.verify};
    const absl::string_view paths[] = {"train", "generate", "verify"};
    for (int i = 0; i < 3; ++i) {
      SCOPED_TRACE(paths[i]);
      if (allowed[i])
        EXPECT_TRUE(statuses[i].ok()) << statuses[i];
      else
        ExpectInvalid(statuses[i], test.flag);
    }
  }
}

TEST(MemorizeGeneralFactsCliTest, AcceptsCompleteFlagSetsForEachPath) {
  EXPECT_TRUE(
      Validate(Mode::kTrainModel,
               {"mode", "tokenizer", "layers", "model_width", "attention_heads",
                "feed_forward_width", "compact_vocabulary", "seed",
                "checkpoint_dir", "search", "steps", "eval_every",
                "checkpoint_every", "learning_rate", "warmup_steps",
                "training_seconds", "corpus", "output_dir", "batch_size"})
          .ok());
  EXPECT_TRUE(
      Validate(Mode::kInferModel,
               {"mode", "tokenizer", "layers", "model_width", "attention_heads",
                "feed_forward_width", "compact_vocabulary", "seed",
                "infer_checkpoint", "prompt", "generation_tokens"},
               "/model", "", "/tokenizer", "")
          .ok());
  EXPECT_TRUE(
      Validate(Mode::kInferModel,
               {"mode", "tokenizer", "layers", "model_width", "attention_heads",
                "feed_forward_width", "compact_vocabulary", "seed",
                "verify_checkpoint", "corpus", "output_dir", "batch_size"},
               "", "/model", "/tokenizer", "")
          .ok());
}

TEST(MemorizeGeneralFactsCliTest, RequiresTokenizerInEveryPath) {
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
    };
    for (const auto& status : statuses) {
      EXPECT_EQ(status.code(), absl::StatusCode::kInternal) << status;
      EXPECT_NE(status.message().find("missing mode policy for --"),
                absl::string_view::npos);
    }
  }
}

}  // namespace
}  // namespace pluto::llm::memorize_general_facts
