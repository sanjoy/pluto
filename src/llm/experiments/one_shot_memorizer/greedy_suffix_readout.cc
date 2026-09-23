#include "src/llm/experiments/one_shot_memorizer/greedy_suffix_readout.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "src/util/status_macros.h"

namespace pluto::llm::one_shot_memorizer {

absl::StatusOr<GreedySuffixReadout> ReadGreedySuffix(
    const GreedySuffixReadoutOptions& options,
    const GreedySuffixLogits& logits) {
  if (!logits || options.vocabulary_size <= 0 || options.eos_token < 0 ||
      options.eos_token >= options.vocabulary_size ||
      options.prompt_token_count <= 0 || options.max_non_eos_tokens < 0 ||
      options.selected_token_ids.size() < 2 ||
      options.selected_token_ids.size() >
          static_cast<size_t>(options.vocabulary_size))
    return absl::InvalidArgumentError(
        "invalid greedy suffix options or callback");
  std::vector<int> selected(options.selected_token_ids.begin(),
                            options.selected_token_ids.end());
  std::sort(selected.begin(), selected.end());
  if (selected.front() < 0 || selected.back() >= options.vocabulary_size ||
      std::adjacent_find(selected.begin(), selected.end()) != selected.end() ||
      !std::binary_search(selected.begin(), selected.end(), options.eos_token))
    return absl::InvalidArgumentError(
        "selected IDs must be distinct, valid, and include EOS");
  selected.erase(
      std::lower_bound(selected.begin(), selected.end(), options.eos_token));
  const size_t n = selected.size();
  if (options.max_non_eos_tokens == 0 &&
      n > std::numeric_limits<int>::max() / 2)
    return absl::OutOfRangeError("derived token limit overflows int");
  const int limit = options.max_non_eos_tokens == 0
                        ? 2 * static_cast<int>(n)
                        : options.max_non_eos_tokens;
  if (options.prompt_token_count > std::numeric_limits<int>::max() - limit)
    return absl::OutOfRangeError("greedy history length overflows int");
  GreedySuffixReadout result;
  result.max_non_eos_tokens = limit;
  for (int start : selected) {
    GreedySuffixCandidate candidate{
        .start_token = start, .token_ids = {start}, .steps = {}};
    std::vector<bool> covered(n, false);
    covered[std::lower_bound(selected.begin(), selected.end(), start) -
            selected.begin()] = true;
    std::vector<int> history(options.prompt_token_count, options.eos_token);
    history.push_back(start);
    while (true) {
      ASSIGN_OR_RETURN(auto values, logits(history));
      ++result.model_query_count;
      if (values.size() != static_cast<size_t>(options.vocabulary_size))
        return absl::InvalidArgumentError(
            "callback returned wrong vocabulary logit count");
      int winner = 0;
      for (size_t token = 0; token < values.size(); ++token) {
        if (!std::isfinite(values[token]))
          return absl::InvalidArgumentError(
              "callback logits must all be finite");
        if (values[token] > values[winner])
          winner = static_cast<int>(token);
      }
      const double maximum = values[winner];
      double denominator = 0;
      for (float value : values)
        denominator += std::exp(static_cast<double>(value) - maximum);
      const double log_probability = -std::log(denominator);
      const bool append =
          winner == options.eos_token ||
          candidate.token_ids.size() < static_cast<size_t>(limit);
      candidate.steps.push_back({winner, log_probability, append});
      if (!append) {
        candidate.hit_token_limit = true;
        break;
      }
      candidate.token_ids.push_back(winner);
      candidate.log_score += log_probability;
      if (winner == options.eos_token) {
        candidate.terminated_with_eos = true;
        break;
      }
      const auto found =
          std::lower_bound(selected.begin(), selected.end(), winner);
      if (found == selected.end() || *found != winner)
        candidate.all_tokens_selected = false;
      else
        covered[found - selected.begin()] = true;
      history.push_back(winner);
    }
    candidate.distinct_selected_covered =
        std::count(covered.begin(), covered.end(), true);
    candidate.covers_selected_tokens = candidate.distinct_selected_covered == n;
    if (candidate.accepted())
      result.accepted_order.push_back(result.candidates.size());
    result.candidates.push_back(std::move(candidate));
  }
  std::sort(result.accepted_order.begin(), result.accepted_order.end(),
            [&](size_t a, size_t b) {
              const auto& left = result.candidates[a];
              const auto& right = result.candidates[b];
              return left.log_score != right.log_score
                         ? left.log_score > right.log_score
                         : left.start_token < right.start_token;
            });
  return result;
}

}  // namespace pluto::llm::one_shot_memorizer
