#include "src/llm/experiments/one_shot_memorizer/context_analysis.h"

#include <algorithm>
#include <limits>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace pluto::llm::one_shot_memorizer {
namespace {

absl::Status ValidateCorpus(absl::Span<const std::vector<int>> sentences,
                            ContextAnalysisOptions options) {
  if (options.prompt_tokens <= 0 || options.eos_token < 0)
    return absl::InvalidArgumentError(
        "context analysis requires positive prompt_tokens and nonnegative EOS");
  if (sentences.empty())
    return absl::InvalidArgumentError(
        "context analysis corpus must not be empty");
  int64_t total_targets = 0;
  for (size_t index = 0; index < sentences.size(); ++index) {
    const auto& sentence = sentences[index];
    if (sentence.size() < static_cast<size_t>(options.prompt_tokens) ||
        sentence.size() >= static_cast<size_t>(std::numeric_limits<int>::max()))
      return absl::InvalidArgumentError(
          absl::StrCat("sentence ", index,
                       " must contain at least prompt_tokens and fewer "
                       "than INT_MAX text tokens"));
    for (int token : sentence)
      if (token < 0 || token == options.eos_token)
        return absl::InvalidArgumentError(absl::StrCat(
            "sentence ", index, " contains a negative or reserved EOS token"));
    const int64_t targets =
        static_cast<int64_t>(sentence.size()) - options.prompt_tokens + 1;
    if (targets > std::numeric_limits<int64_t>::max() - total_targets)
      return absl::InvalidArgumentError("supervised target count overflows");
    total_targets += targets;
  }
  return absl::OkStatus();
}

std::vector<int> ContextKey(absl::Span<const int> history, int window) {
  const size_t length = std::min(history.size(), static_cast<size_t>(window));
  if (length == 0)
    return {};
  return {history.end() - length, history.end()};
}

}  // namespace

absl::StatusOr<ContextModel> ContextModel::Build(
    absl::Span<const std::vector<int>> sentences, int window,
    ContextAnalysisOptions options) {
  if (window < 0)
    return absl::InvalidArgumentError("context window must be nonnegative");
  const auto status = ValidateCorpus(sentences, options);
  if (!status.ok())
    return status;
  ContextModel model(window);
  for (const auto& sentence : sentences) {
    for (int index = options.prompt_tokens;
         index <= static_cast<int>(sentence.size()); ++index) {
      const int token = index == static_cast<int>(sentence.size())
                            ? options.eos_token
                            : sentence[index];
      auto& counts = model.counts_[ContextKey(
          absl::MakeConstSpan(sentence).first(index), window)];
      ++counts.next_tokens[token];
      ++counts.prediction.occurrences;
      ++model.stats_.targets;
    }
  }
  model.stats_.contexts = static_cast<int64_t>(model.counts_.size());
  for (auto& [context, counts] : model.counts_) {
    int64_t largest_count = 0;
    for (const auto& [token, count] : counts.next_tokens) {
      if (count > largest_count ||
          (count == largest_count && token < counts.prediction.token)) {
        largest_count = count;
        counts.prediction.token = token;
      }
    }
    counts.prediction.distinct_next_tokens =
        static_cast<int64_t>(counts.next_tokens.size());
    counts.prediction.probability =
        static_cast<double>(largest_count) / counts.prediction.occurrences;
    model.stats_.conflicting_contexts += counts.next_tokens.size() > 1;
    model.stats_.irreducible_top1_errors +=
        counts.prediction.occurrences - largest_count;
  }
  return model;
}

absl::StatusOr<const ContextModel::ContextCounts*> ContextModel::Find(
    absl::Span<const int> history) const {
  if (std::any_of(history.begin(), history.end(),
                  [](int token) { return token < 0; }))
    return absl::InvalidArgumentError("history contains a negative token ID");
  const auto found = counts_.find(ContextKey(history, window()));
  if (found == counts_.end())
    return absl::NotFoundError(
        "history has no supervised context at this window");
  return &found->second;
}

absl::StatusOr<ContextPrediction> ContextModel::Predict(
    absl::Span<const int> history) const {
  const auto counts = Find(history);
  if (!counts.ok())
    return counts.status();
  return (*counts)->prediction;
}

absl::StatusOr<double> ContextModel::Probability(absl::Span<const int> history,
                                                 int token) const {
  if (token < 0)
    return absl::InvalidArgumentError("predicted token ID must be nonnegative");
  const auto counts = Find(history);
  if (!counts.ok())
    return counts.status();
  const auto found = (*counts)->next_tokens.find(token);
  if (found == (*counts)->next_tokens.end())
    return 0.0;
  return static_cast<double>(found->second) / (*counts)->prediction.occurrences;
}

absl::StatusOr<ContextAnalysis> AnalyzeContexts(
    absl::Span<const std::vector<int>> sentences,
    ContextAnalysisOptions options) {
  const auto status = ValidateCorpus(sentences, options);
  if (!status.ok())
    return status;
  ContextAnalysis analysis;
  for (size_t sentence_index = 0; sentence_index < sentences.size();
       ++sentence_index) {
    const auto& sentence = sentences[sentence_index];
    analysis.longest_prefix =
        std::max(analysis.longest_prefix, static_cast<int>(sentence.size()));
    for (int index = options.prompt_tokens;
         index <= static_cast<int>(sentence.size()); ++index) {
      analysis.targets.push_back(
          {.sentence_index = sentence_index,
           .target_index = index,
           .target_token = index == static_cast<int>(sentence.size())
                               ? options.eos_token
                               : sentence[index]});
    }
  }
  analysis.windows.reserve(static_cast<size_t>(analysis.longest_prefix) + 1);
  for (int window = 0; window <= analysis.longest_prefix; ++window) {
    const auto model = ContextModel::Build(sentences, window, options);
    if (!model.ok())
      return model.status();
    analysis.windows.push_back(model->stats());
    if (analysis.shortest_exact_window < 0 &&
        model->stats().irreducible_top1_errors == 0)
      analysis.shortest_exact_window = window;
    for (auto& target : analysis.targets) {
      if (target.shortest_sufficient_window >= 0 &&
          target.shortest_unique_window >= 0)
        continue;
      const auto prediction =
          model->Predict(absl::MakeConstSpan(sentences[target.sentence_index])
                             .first(target.target_index));
      if (!prediction.ok())
        return prediction.status();
      if (target.shortest_sufficient_window < 0 &&
          prediction->distinct_next_tokens == 1)
        target.shortest_sufficient_window = window;
      if (target.shortest_unique_window < 0 && prediction->occurrences == 1)
        target.shortest_unique_window = window;
    }
  }
  return analysis;
}

}  // namespace pluto::llm::one_shot_memorizer
