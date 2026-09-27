#include "src/llm/experiments/memorize_general_facts/memorize_general_facts_cli.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/util/status_macros.h"

namespace pluto::llm::memorize_general_facts {
namespace {

constexpr uint8_t kTrain = 1 << 0;
constexpr uint8_t kGenerate = 1 << 1;
constexpr uint8_t kVerify = 1 << 2;
constexpr uint8_t kPuzzleCapture = 1 << 3;
constexpr uint8_t kPuzzleTrain = 1 << 4;
constexpr uint8_t kPuzzle = kPuzzleCapture | kPuzzleTrain;
constexpr uint8_t kAll = kTrain | kGenerate | kVerify | kPuzzle;

struct FlagRule {
  absl::string_view name;
  uint8_t allowed_paths;
};

constexpr FlagRule kFlagRules[] = {
    {"mode", kAll},
    {"tokenizer", kAll},
    {"layers", kAll},
    {"model_width", kAll},
    {"attention_heads", kAll},
    {"feed_forward_width", kAll},
    {"context_length", kAll},
    {"compact_vocabulary", kAll},
    {"seed", kTrain | kGenerate | kVerify | kPuzzleTrain},
    {"checkpoint_dir", kTrain},
    {"search", kTrain},
    {"steps", kTrain | kPuzzleTrain},
    {"eval_every", kTrain | kPuzzleTrain},
    {"checkpoint_every", kTrain},
    {"learning_rate", kTrain | kPuzzleTrain},
    {"warmup_steps", kTrain},
    {"training_seconds", kTrain},
    {"corpus", kTrain | kVerify | kPuzzle},
    {"output_dir", kTrain | kVerify | kPuzzle},
    {"batch_size", kTrain | kVerify | kPuzzle},
    {"infer_checkpoint", kGenerate},
    {"prompt", kGenerate},
    {"generation_tokens", kGenerate},
    {"print_attention_probs", kGenerate},
    {"verify_checkpoint", kVerify},
    {"puzzle_checkpoint", kPuzzle},
    {"train_mlp", kPuzzle},
    {"mlp_width", kPuzzleTrain},
    {"mlp_depth", kPuzzleTrain},
    {"match_mlp_parameter_budget", kPuzzleTrain},
};

const FlagRule* FindRule(absl::string_view name) {
  for (const FlagRule& rule : kFlagRules)
    if (rule.name == name)
      return &rule;
  return nullptr;
}

}  // namespace

CommandLineOptions ResolveModeDefaults(
    CommandLineOptions options,
    absl::Span<const absl::string_view> explicitly_set_flags) {
  if (options.mode != "puzzle")
    return options;
  const auto omitted = [&](absl::string_view flag) {
    return std::find(explicitly_set_flags.begin(), explicitly_set_flags.end(),
                     flag) == explicitly_set_flags.end();
  };
  if (omitted("steps"))
    options.steps = 300000;
  if (omitted("eval_every"))
    options.eval_every = 1000;
  if (omitted("learning_rate"))
    options.learning_rate = 0.01;
  if (omitted("seed"))
    options.seed = 3;
  return options;
}

absl::StatusOr<Mode> ParseAndValidateRunMode(
    const CommandLineOptions& options,
    absl::Span<const absl::string_view> explicitly_set_flags) {
  ASSIGN_OR_RETURN(auto mode, ParseMode(options.mode));
  RETURN_IF_ERROR(ValidateModeFlags(
      mode, explicitly_set_flags, options.tokenizer, options.checkpoint_dir,
      options.infer_checkpoint, options.verify_checkpoint,
      options.puzzle_checkpoint, options.train_mlp));

  if (mode == Mode::kPuzzle) {
    if (options.layers < 3)
      return absl::InvalidArgumentError(
          "--layers must be at least 3 in --mode=puzzle");
    if (options.model_width <= 0)
      return absl::InvalidArgumentError("--model_width must be positive");
    if (options.train_mlp && options.mlp_width <= 0)
      return absl::InvalidArgumentError("--mlp_width must be positive");
    if (options.train_mlp && options.mlp_depth <= 0)
      return absl::InvalidArgumentError("--mlp_depth must be positive");
    if (options.train_mlp && options.seed < 0)
      return absl::InvalidArgumentError("--seed must be nonnegative");
  }

  if (options.context_length <= 0)
    return absl::InvalidArgumentError("--context_length must be positive");

  if (mode == Mode::kInferModel && !options.infer_checkpoint.empty()) {
    if (options.generation_tokens < 0)
      return absl::InvalidArgumentError(
          "--generation_tokens must be nonnegative");
    if (std::find(explicitly_set_flags.begin(), explicitly_set_flags.end(),
                  "prompt") != explicitly_set_flags.end() &&
        options.prompt.empty())
      return absl::InvalidArgumentError(
          "--prompt must be nonempty when supplied");
    return mode;
  }

  // Training, checkpoint verification, and puzzle capture consume corpus
  // batches and write artifacts.
  if (options.corpus.empty())
    return absl::InvalidArgumentError("--corpus must be nonempty");
  if (options.output_dir.empty())
    return absl::InvalidArgumentError("--output_dir must be nonempty");
  if (options.batch_size <= 0)
    return absl::InvalidArgumentError("--batch_size must be positive");
  if (options.context_length < 5)
    return absl::InvalidArgumentError(
        "--context_length must be at least 5 for the corpus prompt");
  if (int64_t{options.batch_size} * options.context_length >
      std::numeric_limits<int>::max())
    return absl::InvalidArgumentError(
        "--batch_size * --context_length exceeds the 32-bit token-count limit");
  if (mode == Mode::kTrainModel ||
      (mode == Mode::kPuzzle && options.train_mlp)) {
    if (options.steps < 0)
      return absl::InvalidArgumentError("--steps must be nonnegative");
    if (options.eval_every <= 0)
      return absl::InvalidArgumentError("--eval_every must be positive");
    if (!std::isfinite(options.learning_rate) || options.learning_rate <= 0)
      return absl::InvalidArgumentError(
          "--learning_rate must be finite and positive");
    if (mode == Mode::kPuzzle &&
        (options.learning_rate > std::numeric_limits<float>::max() ||
         static_cast<float>(options.learning_rate) <= 0))
      return absl::InvalidArgumentError(
          "--learning_rate must remain finite and positive as a float");
  }
  if (mode == Mode::kTrainModel) {
    if (options.checkpoint_every <= 0)
      return absl::InvalidArgumentError("--checkpoint_every must be positive");
    if (options.warmup_steps < 0)
      return absl::InvalidArgumentError("--warmup_steps must be nonnegative");
    if (!std::isfinite(options.training_seconds) ||
        options.training_seconds < 0)
      return absl::InvalidArgumentError(
          "--training_seconds must be finite and nonnegative");
  }
  return mode;
}

absl::StatusOr<Mode> ParseMode(absl::string_view mode) {
  if (mode == "train_model")
    return Mode::kTrainModel;
  if (mode == "infer_model")
    return Mode::kInferModel;
  if (mode == "puzzle")
    return Mode::kPuzzle;
  return absl::InvalidArgumentError(
      "--mode must be one of: train_model, infer_model, puzzle");
}

absl::string_view ModeName(Mode mode) {
  switch (mode) {
    case Mode::kTrainModel:
      return "train_model";
    case Mode::kInferModel:
      return "infer_model";
    case Mode::kPuzzle:
      return "puzzle";
  }
  return "unknown";
}

absl::Status ValidateModeFlags(
    Mode mode, absl::Span<const absl::string_view> explicitly_set_flags,
    absl::string_view tokenizer, absl::string_view checkpoint_dir,
    absl::string_view infer_checkpoint, absl::string_view verify_checkpoint,
    absl::string_view puzzle_checkpoint, bool train_mlp) {
  if (mode != Mode::kTrainModel && mode != Mode::kInferModel &&
      mode != Mode::kPuzzle)
    return absl::InvalidArgumentError("invalid --mode enum value");

  bool has_infer_checkpoint = !infer_checkpoint.empty();
  bool has_verify_checkpoint = !verify_checkpoint.empty();
  for (absl::string_view name : explicitly_set_flags) {
    if (FindRule(name) == nullptr)
      return absl::InternalError(
          absl::StrCat("missing mode policy for --", name));
    // A default-valued or empty explicit flag must not evade mode policy.
    has_infer_checkpoint |= name == "infer_checkpoint";
    has_verify_checkpoint |= name == "verify_checkpoint";
  }

  uint8_t path = kTrain;
  if (mode == Mode::kPuzzle)
    path = train_mlp ? kPuzzleTrain : kPuzzleCapture;
  if (mode == Mode::kInferModel) {
    if (has_infer_checkpoint && has_verify_checkpoint)
      return absl::InvalidArgumentError(
          "--infer_checkpoint and --verify_checkpoint are mutually exclusive "
          "in --mode=infer_model, including explicitly empty values");
    if (infer_checkpoint.empty() && verify_checkpoint.empty())
      return absl::InvalidArgumentError(
          "--mode=infer_model requires exactly one nonempty "
          "--infer_checkpoint or --verify_checkpoint");
    path = has_verify_checkpoint ? kVerify : kGenerate;
  }

  for (absl::string_view name : explicitly_set_flags) {
    if ((FindRule(name)->allowed_paths & path) != 0)
      continue;
    return absl::InvalidArgumentError(
        absl::StrCat("--", name, " is not valid in --mode=", ModeName(mode),
                     path == kGenerate        ? " with --infer_checkpoint"
                     : path == kVerify        ? " with --verify_checkpoint"
                     : path == kPuzzleCapture ? " without --train_mlp"
                                              : ""));
  }
  if (tokenizer.empty())
    return absl::InvalidArgumentError(
        absl::StrCat("--tokenizer is required in --mode=", ModeName(mode)));
  if (mode == Mode::kTrainModel && checkpoint_dir.empty())
    return absl::InvalidArgumentError(
        "--checkpoint_dir is required in --mode=train_model");
  if (mode == Mode::kPuzzle && puzzle_checkpoint.empty())
    return absl::InvalidArgumentError(
        "--puzzle_checkpoint is required in --mode=puzzle");
  return absl::OkStatus();
}

}  // namespace pluto::llm::memorize_general_facts
