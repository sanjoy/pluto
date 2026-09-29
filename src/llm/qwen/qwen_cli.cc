#include "src/llm/qwen/qwen_cli.h"

#include <cmath>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"

namespace pluto::llm::qwen {
namespace {

enum class FlagMode { kAll, kInfer, kTrain, kEmbeddingAlgebra };

struct FlagRule {
  absl::string_view name;
  FlagMode mode;
};

constexpr FlagRule kFlagRules[] = {
    {"mode", FlagMode::kAll},
    {"checkpoint", FlagMode::kAll},
    {"prompt", FlagMode::kInfer},
    {"max_new_tokens", FlagMode::kInfer},
    {"context_length", FlagMode::kInfer},
    {"raw_prompt", FlagMode::kInfer},
    {"thinking", FlagMode::kInfer},
    {"text", FlagMode::kTrain},
    {"sequence_length", FlagMode::kTrain},
    {"batch_size", FlagMode::kTrain},
    {"steps", FlagMode::kTrain},
    {"switch_every", FlagMode::kTrain},
    {"start_block", FlagMode::kTrain},
    {"learning_rate", FlagMode::kTrain},
    {"max_active_gib", FlagMode::kTrain},
    {"resume_weights", FlagMode::kTrain},
    {"save_weights", FlagMode::kTrain},
    {"expression", FlagMode::kEmbeddingAlgebra},
    {"top_n", FlagMode::kEmbeddingAlgebra},
};

const FlagRule* FindRule(absl::string_view name) {
  for (const FlagRule& rule : kFlagRules)
    if (rule.name == name)
      return &rule;
  return nullptr;
}

}  // namespace

absl::StatusOr<Mode> ParseAndValidateRunMode(
    const CommandLineOptions& options,
    absl::Span<const absl::string_view> explicitly_set_flags) {
  Mode mode;
  if (options.mode == "infer_model")
    mode = Mode::kInferModel;
  else if (options.mode == "train_model")
    mode = Mode::kTrainModel;
  else if (options.mode == "embedding_algebra")
    mode = Mode::kEmbeddingAlgebra;
  else
    return absl::InvalidArgumentError(
        "--mode is required and must be one of: infer_model, train_model, "
        "embedding_algebra");

  FlagMode allowed = FlagMode::kEmbeddingAlgebra;
  if (mode == Mode::kInferModel)
    allowed = FlagMode::kInfer;
  else if (mode == Mode::kTrainModel)
    allowed = FlagMode::kTrain;
  bool expression_is_explicit = false;
  for (absl::string_view flag : explicitly_set_flags) {
    const FlagRule* rule = FindRule(flag);
    if (rule == nullptr)
      return absl::InternalError(
          absl::StrCat("missing mode policy for --", flag));
    if (rule->mode != FlagMode::kAll && rule->mode != allowed)
      return absl::InvalidArgumentError(
          absl::StrCat("--", flag, " is not valid in --mode=", options.mode));
    if (flag == "expression")
      expression_is_explicit = true;
  }
  if (options.checkpoint.empty())
    return absl::InvalidArgumentError("--checkpoint is required");

  if (mode == Mode::kEmbeddingAlgebra) {
    if (options.top_n <= 0)
      return absl::InvalidArgumentError("--top_n must be positive");
    if (expression_is_explicit &&
        absl::StripAsciiWhitespace(options.expression).empty())
      return absl::InvalidArgumentError("--expression must not be empty");
    return mode;
  }

  if (mode == Mode::kInferModel) {
    if (options.max_new_tokens <= 0)
      return absl::InvalidArgumentError("--max_new_tokens must be positive");
    if (options.context_length <= 0)
      return absl::InvalidArgumentError("--context_length must be positive");
    if (options.raw_prompt && options.thinking)
      return absl::InvalidArgumentError(
          "--thinking is only valid with the chat template, not --raw_prompt");
    return mode;
  }

  if (options.sequence_length <= 0 || options.sequence_length > 128)
    return absl::InvalidArgumentError("--sequence_length must be in [1,128]");
  if (options.batch_size != 1)
    return absl::InvalidArgumentError("--batch_size must be 1");
  if (options.steps <= 0)
    return absl::InvalidArgumentError("--steps must be positive");
  if (options.switch_every <= 0)
    return absl::InvalidArgumentError("--switch_every must be positive");
  if (options.start_block < -1)
    return absl::InvalidArgumentError("--start_block must be at least -1");
  if (!std::isfinite(options.learning_rate) || options.learning_rate <= 0 ||
      options.learning_rate > 1 ||
      static_cast<float>(options.learning_rate) <= 0)
    return absl::InvalidArgumentError(
        "--learning_rate must be finite, in (0,1], and positive as a float");
  if (!std::isfinite(options.max_active_gib) || options.max_active_gib < 0 ||
      options.max_active_gib > 1024)
    return absl::InvalidArgumentError(
        "--max_active_gib must be finite and in [0,1024]");
  return mode;
}

}  // namespace pluto::llm::qwen
