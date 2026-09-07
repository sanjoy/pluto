#include "src/llm/recipes/gpt2_shakespeare_cli.h"

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace pluto::llm {
namespace {

constexpr uint8_t kTrainModel = 1 << 0;
constexpr uint8_t kInferModel = 1 << 1;
constexpr uint8_t kTrainSparseAutoEncoder = 1 << 2;
constexpr uint8_t kTrain = kTrainModel | kTrainSparseAutoEncoder;

struct FlagRule {
  absl::string_view name;
  uint8_t allowed_modes;
};

constexpr FlagRule kFlagRules[] = {
    {"corpus", kTrain},
    {"resume_from", kTrain},
    {"inference_from", kInferModel},
    {"sparse_autoencoder_from", kTrainSparseAutoEncoder},
    {"checkpoint_dir", kTrain},
    {"checkpoint_every", kTrain},
    {"steps", kTrain},
    {"learning_rate", kTrain},
    {"adam_beta1", kTrain},
    {"adam_beta2", kTrain},
    {"adam_epsilon", kTrain},
    {"weight_decay", kTrain},
    {"eval_batches", kTrain},
    {"test_fraction", kTrainModel},
    {"train_until_loss", kTrain},
    {"training_eval_interval", kTrain},
    {"prompt", kInferModel},
    {"generation_tokens", kInferModel},
    {"temperature", kInferModel},
    {"batch_size", kTrain},
    {"log_file", kTrain},
};

uint8_t ModeMask(Gpt2ShakespeareMode mode) {
  switch (mode) {
    case Gpt2ShakespeareMode::kTrainModel:
      return kTrainModel;
    case Gpt2ShakespeareMode::kInferModel:
      return kInferModel;
    case Gpt2ShakespeareMode::kTrainSparseAutoEncoder:
      return kTrainSparseAutoEncoder;
  }
  return 0;
}

const FlagRule* FindRule(absl::string_view name) {
  for (const FlagRule& rule : kFlagRules) {
    if (rule.name == name) return &rule;
  }
  return nullptr;
}

}  // namespace

absl::StatusOr<Gpt2ShakespeareMode> ParseGpt2ShakespeareMode(
    absl::string_view mode) {
  if (mode == "train_model") return Gpt2ShakespeareMode::kTrainModel;
  if (mode == "infer_model") return Gpt2ShakespeareMode::kInferModel;
  if (mode == "train_sae") {
    return Gpt2ShakespeareMode::kTrainSparseAutoEncoder;
  }
  return absl::InvalidArgumentError(
      "--mode must be one of: train_model, infer_model, train_sae");
}

absl::string_view Gpt2ShakespeareModeName(Gpt2ShakespeareMode mode) {
  switch (mode) {
    case Gpt2ShakespeareMode::kTrainModel:
      return "train_model";
    case Gpt2ShakespeareMode::kInferModel:
      return "infer_model";
    case Gpt2ShakespeareMode::kTrainSparseAutoEncoder:
      return "train_sae";
  }
  return "unknown";
}

absl::Status ValidateGpt2ShakespeareModeFlags(
    Gpt2ShakespeareMode mode,
    absl::Span<const absl::string_view> explicitly_set_flags,
    absl::string_view inference_from,
    absl::string_view sparse_autoencoder_from) {
  const uint8_t mode_mask = ModeMask(mode);
  for (absl::string_view name : explicitly_set_flags) {
    const FlagRule* rule = FindRule(name);
    if (rule == nullptr) {
      return absl::InternalError(
          absl::StrCat("missing mode policy for --", name));
    }
    if ((rule->allowed_modes & mode_mask) == 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "--", name,
          " is not valid in --mode=", Gpt2ShakespeareModeName(mode)));
    }
  }
  if (mode == Gpt2ShakespeareMode::kInferModel && inference_from.empty()) {
    return absl::InvalidArgumentError(
        "--inference_from is required in --mode=infer_model");
  }
  if (mode == Gpt2ShakespeareMode::kTrainSparseAutoEncoder &&
      sparse_autoencoder_from.empty()) {
    return absl::InvalidArgumentError(
        "--sparse_autoencoder_from is required in --mode=train_sae");
  }
  return absl::OkStatus();
}

}  // namespace pluto::llm
