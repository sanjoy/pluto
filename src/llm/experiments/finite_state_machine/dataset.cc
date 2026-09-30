#include "src/llm/experiments/finite_state_machine/dataset.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/util/status_macros.h"

namespace pluto::llm::fsm {
namespace {

bool Digits(absl::string_view text) {
  return text.size() == 3 && std::all_of(text.begin(), text.end(), [](char c) {
           return c >= '0' && c <= '9';
         });
}

int StateNumber(absl::string_view text) {
  return 100 * (text[0] - '0') + 10 * (text[1] - '0') + text[2] - '0';
}

struct Sentence {
  size_t prompt_bytes;
  int answer_token;
};

// Parse and execute the line independently of the generator. The prompt ends
// immediately after >; its next token is the single checked state or ERR.
absl::StatusOr<Sentence> ValidateSentence(absl::string_view line) {
  const size_t separator = line.find('>');
  if (separator == absl::string_view::npos ||
      line.find('>', separator + 1) != absl::string_view::npos)
    return absl::InvalidArgumentError(
        "expected exactly one > in transitions;input>output");
  const std::vector<absl::string_view> fields =
      absl::StrSplit(line.substr(0, separator), ';');
  if (fields.size() < 2)
    return absl::InvalidArgumentError("expected transitions;input>output");
  std::array<int, 1000 * 26> transitions;
  transitions.fill(-1);
  for (size_t i = 0; i + 1 < fields.size(); ++i) {
    const auto edge = fields[i];
    if (edge.size() != 7 || !Digits(edge.substr(0, 3)) ||
        !Digits(edge.substr(4, 3)) || edge[3] < 'A' || edge[3] > 'Z')
      return absl::InvalidArgumentError("invalid transition: " +
                                        std::string(edge));
    int& target =
        transitions[StateNumber(edge.substr(0, 3)) * 26 + edge[3] - 'A'];
    if (target != -1)
      return absl::InvalidArgumentError("duplicate state/symbol transition");
    target = StateNumber(edge.substr(4, 3));
  }
  const auto input = fields.back();
  const auto answer = line.substr(separator + 1);
  if (input.empty() || !std::all_of(input.begin(), input.end(), [](char c) {
        return c >= 'A' && c <= 'Z';
      }))
    return absl::InvalidArgumentError("input must be a nonempty A-Z string");
  if (answer != "ERR" && !Digits(answer))
    return absl::InvalidArgumentError(
        "output must be a three-digit state or ERR");
  int state = 0;
  for (char letter : input) {
    state = transitions[state * 26 + letter - 'A'];
    if (state < 0)
      break;
  }
  if ((state < 0 && answer != "ERR") ||
      (state >= 0 && (answer == "ERR" || StateNumber(answer) != state)))
    return absl::InvalidArgumentError("output does not match FSM execution");
  return Sentence{separator + 1, state < 0 ? kErrorToken : state};
}

}  // namespace

absl::StatusOr<std::unique_ptr<FsmDataSetIterator>> FsmDataSetIterator::Create(
    cuda::Executor& executor, absl::string_view corpus_text,
    const FsmTokenizer& tokenizer, DataSetOptions options) {
  // Tokenizer IDs use int while dataset device storage uses int32_t.
  static_assert(sizeof(int) == sizeof(int32_t));
  if (options.batch_size <= 0 || options.context_length <= 0 ||
      int64_t{options.batch_size} * options.context_length >
          std::numeric_limits<int>::max())
    return absl::InvalidArgumentError("invalid FSM batch/context dimensions");
  if (corpus_text.empty())
    return absl::InvalidArgumentError("FSM corpus must not be empty");
  if (corpus_text.back() == '\n')
    corpus_text.remove_suffix(1);
  const std::vector<absl::string_view> lines =
      absl::StrSplit(corpus_text, '\n');
  if (lines.size() >
      std::numeric_limits<size_t>::max() / options.context_length / sizeof(int))
    return absl::InvalidArgumentError("padded FSM corpus size overflows");
  const size_t rows = lines.size() * options.context_length;
  ASSIGN_OR_RETURN(auto host_inputs,
                   cuda::PageLockedHostArray<int>::Allocate(executor, rows));
  ASSIGN_OR_RETURN(auto host_targets,
                   cuda::PageLockedHostArray<int>::Allocate(executor, rows));
  std::fill(host_inputs.begin(), host_inputs.end(), 0);
  std::fill(host_targets.begin(), host_targets.end(), -1);
  std::vector<int> supervised_rows;
  supervised_rows.reserve(lines.size());
  int max_tokens = 0;
  for (size_t i = 0; i < lines.size(); ++i) {
    auto line = lines[i];
    if (!line.empty() && line.back() == '\r')
      line.remove_suffix(1);
    auto sentence = ValidateSentence(line);
    if (!sentence.ok())
      return absl::InvalidArgumentError(
          absl::StrCat("line ", i + 1, ": ", sentence.status().message()));
    ASSIGN_OR_RETURN(auto tokens, tokenizer.Encode(executor, line));
    ASSIGN_OR_RETURN(
        auto prompt,
        tokenizer.Encode(executor, line.substr(0, sentence->prompt_bytes)));
    if (tokens.size() > static_cast<size_t>(options.context_length))
      return absl::InvalidArgumentError(
          absl::StrCat("line ", i + 1, " has ", tokens.size(),
                       " tokens; context_length is ", options.context_length));
    // The answer occupies exactly one token after >. Verify the full-line
    // encoding preserves its prompt and matches the independently run FSM.
    if (prompt.empty() || prompt.size() + 1 != tokens.size() ||
        prompt[prompt.size() - 1] != kOutputSeparatorToken ||
        tokens[tokens.size() - 1] != sentence->answer_token ||
        !std::equal(prompt.begin(), prompt.end(), tokens.begin()))
      return absl::InvalidArgumentError(
          "FSM tokenizer must encode exactly one answer token after >");
    if (std::any_of(tokens.begin(), tokens.end(), [&](int token) {
          return token < 0 || token >= tokenizer.vocab_size();
        }))
      return absl::InvalidArgumentError(
          "tokenizer returned an invalid token ID");
    const size_t base = i * options.context_length;
    std::copy(tokens.begin(), tokens.end(), host_inputs.data() + base);
    const size_t first_target = options.answer_only ? prompt.size() - 1 : 0;
    for (size_t row = first_target; row + 1 < tokens.size(); ++row)
      host_targets[base + row] = tokens[row + 1];
    supervised_rows.push_back(
        static_cast<int>(tokens.size() - 1 - first_target));
    max_tokens = std::max(max_tokens, static_cast<int>(tokens.size()));
  }
  ASSIGN_OR_RETURN(auto inputs,
                   cuda::Buffer::Allocate(executor, host_inputs.size_bytes()));
  ASSIGN_OR_RETURN(auto targets,
                   cuda::Buffer::Allocate(executor, host_targets.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(inputs.data(), host_inputs.data(),
                      host_inputs.size_bytes(), cudaMemcpyHostToDevice,
                      executor.stream()),
      "upload FSM inputs"));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(targets.data(), host_targets.data(),
                      host_targets.size_bytes(), cudaMemcpyHostToDevice,
                      executor.stream()),
      "upload FSM targets"));
  // Pinned staging buffers can be released now; their frees are ordered after
  // the uploads on executor. No compute-stream synchronization is needed.
  const auto allocate_batch =
      [&](size_t samples) -> absl::StatusOr<BatchBuffers> {
    const size_t bytes = samples * options.context_length * sizeof(int);
    ASSIGN_OR_RETURN(auto batch_inputs, cuda::Buffer::Allocate(executor, bytes));
    ASSIGN_OR_RETURN(auto batch_targets,
                     cuda::Buffer::Allocate(executor, bytes));
    return BatchBuffers{std::move(batch_inputs), std::move(batch_targets)};
  };
  ASSIGN_OR_RETURN(auto full_batch, allocate_batch(std::min<size_t>(
                                        options.batch_size, lines.size())));
  std::optional<BatchBuffers> partial_batch;
  if (lines.size() > static_cast<size_t>(options.batch_size) &&
      lines.size() % options.batch_size != 0) {
    ASSIGN_OR_RETURN(partial_batch,
                     allocate_batch(lines.size() % options.batch_size));
  }
  return absl::WrapUnique(new FsmDataSetIterator(
      executor, options, std::move(inputs), std::move(targets),
      std::move(supervised_rows), max_tokens, std::move(full_batch),
      std::move(partial_batch)));
}

FsmDataSetIterator::FsmDataSetIterator(
    cuda::Executor& executor, DataSetOptions options, cuda::Buffer inputs,
    cuda::Buffer targets, std::vector<int> supervised_rows, int max_tokens,
    BatchBuffers full_batch, std::optional<BatchBuffers> partial_batch)
    : executor_(executor),
      options_(options),
      corpus_inputs_(std::move(inputs)),
      corpus_targets_(std::move(targets)),
      supervised_rows_(std::move(supervised_rows)),
      max_tokens_(max_tokens),
      full_batch_(std::move(full_batch)),
      partial_batch_(std::move(partial_batch)),
      order_(supervised_rows_.size()),
      random_(options.seed) {
  BeginEpoch();
}

void FsmDataSetIterator::BeginEpoch() {
  std::iota(order_.begin(), order_.end(), 0);
  if (options_.shuffle)
    std::shuffle(order_.begin(), order_.end(), random_);
  next_sample_ = 0;
}

absl::Status FsmDataSetIterator::Reset() {
  random_.seed(options_.seed);
  BeginEpoch();
  return absl::OkStatus();
}

size_t FsmDataSetIterator::batches_per_epoch() const {
  return (sample_count() - 1) / options_.batch_size + 1;
}

int64_t FsmDataSetIterator::supervised_row_count() const {
  return std::accumulate(supervised_rows_.begin(), supervised_rows_.end(),
                         int64_t{0});
}

absl::StatusOr<DataBatch> FsmDataSetIterator::Next() {
  if (next_sample_ == order_.size())
    BeginEpoch();
  const size_t samples =
      std::min<size_t>(options_.batch_size, order_.size() - next_sample_);
  const auto& batch =
      samples == std::min<size_t>(options_.batch_size, order_.size())
          ? full_batch_
          : *partial_batch_;
  const size_t sample_bytes = options_.context_length * sizeof(int);
  int supervised = 0;
  for (size_t i = 0; i < samples; ++i) {
    const size_t index = order_[next_sample_ + i];
    const size_t source = index * options_.context_length;
    const size_t destination = i * options_.context_length;
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(static_cast<int*>(batch.inputs.data()) + destination,
                        static_cast<const int*>(corpus_inputs_.data()) + source,
                        sample_bytes, cudaMemcpyDeviceToDevice,
                        executor_.stream()),
        "copy FSM batch inputs"));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(
            static_cast<int*>(batch.targets.data()) + destination,
            static_cast<const int*>(corpus_targets_.data()) + source,
            sample_bytes, cudaMemcpyDeviceToDevice, executor_.stream()),
        "copy FSM batch targets"));
    supervised += supervised_rows_[index];
  }
  next_sample_ += samples;
  return DataBatch{.inputs = batch.inputs,
                   .targets = batch.targets,
                   .batch_size = static_cast<int32_t>(samples),
                   .sequence_length = options_.context_length,
                   .supervised_row_count = supervised};
}

}  // namespace pluto::llm::fsm
