#include "src/llm/experiments/gpt2_shakespeare/gpt2_shakespeare_cli.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace pluto::llm {
namespace {

constexpr uint8_t kTrainModel = 1 << 0;
constexpr uint8_t kInferModel = 1 << 1;
constexpr uint8_t kTrainSparseAutoEncoder = 1 << 2;
constexpr uint8_t kInferSparseAutoEncoder = 1 << 3;
constexpr uint8_t kTrain = kTrainModel | kTrainSparseAutoEncoder;

struct FlagRule {
  absl::string_view name;
  uint8_t allowed_modes;
};

constexpr FlagRule kFlagRules[] = {
    {"corpus", kTrain},
    {"resume_from", kTrain},
    {"inference_from", kInferModel | kInferSparseAutoEncoder},
    {"sparse_autoencoder_from",
     kTrainSparseAutoEncoder | kInferSparseAutoEncoder},
    {"checkpoint_dir", kTrain},
    {"checkpoint_every", kTrain},
    {"checkpoint_initial", kTrain},
    {"steps", kTrain},
    {"training_seconds", kTrain},
    {"learning_rate", kTrain},
    {"adam_beta1", kTrain},
    {"adam_beta2", kTrain},
    {"adam_epsilon", kTrain},
    {"weight_decay", kTrain},
    {"eval_batches", kTrain},
    {"test_fraction", kTrainModel},
    {"train_until_loss", kTrain},
    {"training_eval_interval", kTrain},
    {"prompt", kInferModel | kInferSparseAutoEncoder},
    {"generation_tokens", kInferModel},
    {"temperature", kInferModel},
    {"inspect_activations", kInferModel},
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
    case Gpt2ShakespeareMode::kInferSparseAutoEncoder:
      return kInferSparseAutoEncoder;
  }
  return 0;
}

const FlagRule* FindRule(absl::string_view name) {
  for (const FlagRule& rule : kFlagRules)
    if (rule.name == name)
      return &rule;
  return nullptr;
}

absl::Status InspectionSettingError(absl::string_view reason) {
  return absl::InvalidArgumentError(absl::StrCat(
      "--inspect_activations: ", reason,
      "; expected empty (disabled), neighboring_vocab, or "
      "neighboring_vocab(min_prob=0.05) with a probability in [0, 1]"));
}

}  // namespace

absl::Status ValidateGpt2ShakespeareTrainingSeconds(double training_seconds) {
  if (!std::isfinite(training_seconds) || training_seconds <= 0.0) {
    return absl::InvalidArgumentError(
        "--training_seconds must be finite and positive");
  }
  return absl::OkStatus();
}

absl::StatusOr<Gpt2ShakespeareMode> ParseGpt2ShakespeareMode(
    absl::string_view mode) {
  if (mode == "train_model")
    return Gpt2ShakespeareMode::kTrainModel;
  if (mode == "infer_model")
    return Gpt2ShakespeareMode::kInferModel;
  if (mode == "train_sae")
    return Gpt2ShakespeareMode::kTrainSparseAutoEncoder;
  if (mode == "infer_SAE" || mode == "infer_sae")
    return Gpt2ShakespeareMode::kInferSparseAutoEncoder;
  return absl::InvalidArgumentError(
      "--mode must be one of: train_model, infer_model, train_sae, infer_SAE");
}

absl::StatusOr<ActivationInspectionOptions> ParseActivationInspectionMode(
    absl::string_view mode) {
  ActivationInspectionOptions options;
  mode = absl::StripAsciiWhitespace(mode);
  if (mode.empty())
    return options;

  const size_t open = mode.find('(');
  const absl::string_view name =
      absl::StripAsciiWhitespace(mode.substr(0, open));
  if (name != "neighboring_vocab")
    return InspectionSettingError("unknown inspection mode; select one mode");
  options.mode = ActivationInspectionMode::kNeighboringVocab;
  if (open == absl::string_view::npos)
    return options;

  // A setting value cannot contain nested calls, and the matching close must
  // end the expression. This also rejects multiple parenthesized modes.
  if (mode.back() != ')' || mode.find(')', open + 1) != mode.size() - 1 ||
      mode.find('(', open + 1) != absl::string_view::npos)
    return InspectionSettingError("malformed or nested mode parentheses");
  absl::string_view settings =
      absl::StripAsciiWhitespace(mode.substr(open + 1, mode.size() - open - 2));
  if (settings.empty())
    return options;

  bool saw_min_prob = false;
  while (true) {
    const size_t comma = settings.find(',');
    const absl::string_view setting =
        absl::StripAsciiWhitespace(settings.substr(0, comma));
    const size_t equals = setting.find('=');
    if (setting.empty() || equals == absl::string_view::npos ||
        setting.find('=', equals + 1) != absl::string_view::npos)
      return InspectionSettingError(
          "each setting must be a nonempty key=value");
    const absl::string_view key =
        absl::StripAsciiWhitespace(setting.substr(0, equals));
    const absl::string_view value =
        absl::StripAsciiWhitespace(setting.substr(equals + 1));
    if (key.empty() || value.empty())
      return InspectionSettingError(
          "setting keys and values must not be empty");
    if (key != "min_prob")
      return InspectionSettingError(
          absl::StrCat("unknown setting '", key, "'; supported key: min_prob"));
    if (saw_min_prob)
      return InspectionSettingError("duplicate min_prob setting");
    saw_min_prob = true;
    if (!absl::SimpleAtod(value, &options.min_prob) ||
        !std::isfinite(options.min_prob) || options.min_prob < 0.0 ||
        options.min_prob > 1.0)
      return InspectionSettingError("min_prob must be finite and in [0, 1]");

    if (comma == absl::string_view::npos)
      return options;
    // Do not skip empty fields: leading, trailing, or repeated commas are
    // malformed settings rather than an opportunity to silently use defaults.
    settings.remove_prefix(comma + 1);
  }
}

absl::string_view Gpt2ShakespeareModeName(Gpt2ShakespeareMode mode) {
  switch (mode) {
    case Gpt2ShakespeareMode::kTrainModel:
      return "train_model";
    case Gpt2ShakespeareMode::kInferModel:
      return "infer_model";
    case Gpt2ShakespeareMode::kTrainSparseAutoEncoder:
      return "train_sae";
    case Gpt2ShakespeareMode::kInferSparseAutoEncoder:
      return "infer_SAE";
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
  if ((mode == Gpt2ShakespeareMode::kInferModel ||
       mode == Gpt2ShakespeareMode::kInferSparseAutoEncoder) &&
      inference_from.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("--inference_from is required in --mode=",
                     Gpt2ShakespeareModeName(mode)));
  }
  if ((mode == Gpt2ShakespeareMode::kTrainSparseAutoEncoder ||
       mode == Gpt2ShakespeareMode::kInferSparseAutoEncoder) &&
      sparse_autoencoder_from.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("--sparse_autoencoder_from is required in --mode=",
                     Gpt2ShakespeareModeName(mode)));
  }
  return absl::OkStatus();
}

}  // namespace pluto::llm
