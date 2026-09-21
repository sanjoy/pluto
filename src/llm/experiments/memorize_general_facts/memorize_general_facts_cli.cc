#include "src/llm/experiments/memorize_general_facts/memorize_general_facts_cli.h"

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace pluto::llm::memorize_general_facts {
namespace {

constexpr uint8_t kTrain = 1 << 0;
constexpr uint8_t kGenerate = 1 << 1;
constexpr uint8_t kVerify = 1 << 2;
constexpr uint8_t kAll = kTrain | kGenerate | kVerify;

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
    {"compact_vocabulary", kAll},
    {"seed", kAll},
    {"checkpoint_dir", kTrain},
    {"search", kTrain},
    {"steps", kTrain},
    {"eval_every", kTrain},
    {"checkpoint_every", kTrain},
    {"learning_rate", kTrain},
    {"warmup_steps", kTrain},
    {"training_seconds", kTrain},
    {"corpus", kTrain | kVerify},
    {"output_dir", kTrain | kVerify},
    {"batch_size", kTrain | kVerify},
    {"infer_checkpoint", kGenerate},
    {"prompt", kGenerate},
    {"generation_tokens", kGenerate},
    {"verify_checkpoint", kVerify},
};

const FlagRule* FindRule(absl::string_view name) {
  for (const FlagRule& rule : kFlagRules)
    if (rule.name == name)
      return &rule;
  return nullptr;
}

}  // namespace

absl::StatusOr<Mode> ParseMode(absl::string_view mode) {
  if (mode == "train_model")
    return Mode::kTrainModel;
  if (mode == "infer_model")
    return Mode::kInferModel;
  return absl::InvalidArgumentError(
      "--mode must be one of: train_model, infer_model");
}

absl::string_view ModeName(Mode mode) {
  switch (mode) {
    case Mode::kTrainModel:
      return "train_model";
    case Mode::kInferModel:
      return "infer_model";
  }
  return "unknown";
}

absl::Status ValidateModeFlags(
    Mode mode, absl::Span<const absl::string_view> explicitly_set_flags,
    absl::string_view tokenizer, absl::string_view checkpoint_dir,
    absl::string_view infer_checkpoint, absl::string_view verify_checkpoint) {
  if (mode != Mode::kTrainModel && mode != Mode::kInferModel)
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
                     path == kGenerate ? " with --infer_checkpoint"
                     : path == kVerify ? " with --verify_checkpoint"
                                       : ""));
  }
  if (tokenizer.empty())
    return absl::InvalidArgumentError(
        absl::StrCat("--tokenizer is required in --mode=", ModeName(mode)));
  if (mode == Mode::kTrainModel && checkpoint_dir.empty())
    return absl::InvalidArgumentError(
        "--checkpoint_dir is required in --mode=train_model");
  return absl::OkStatus();
}

}  // namespace pluto::llm::memorize_general_facts
