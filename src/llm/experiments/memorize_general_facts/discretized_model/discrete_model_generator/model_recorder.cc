#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/model_recorder.h"

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "src/cuda/executor.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/gpt2_detokenizer.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/padded_line_dataset.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/captured_model_util.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/snapshot_execution_trace.h"
#include "src/llm/experiments/memorize_general_facts/discretized_model/discrete_model_generator/utils.h"
#include "src/llm/gpt2.h"
#include "src/util/status_macros.h"

namespace pluto::llm::discretized::generator {
namespace capture = pluto::llm::memorize_general_facts::discretized_model;

absl::StatusOr<CapturedModel> ModelRecorder::Record(
    const ModelRecorderOptions& options, StateVectorHints* vector_hints) {
  if (options.checkpoint.empty() || options.tokenizer.empty() ||
      options.corpus.empty())
    return absl::InvalidArgumentError(
        "checkpoint, tokenizer, and corpus are required");
  if (options.layers <= 0 || options.layers > 1024 ||
      options.expected_samples <= 0 || options.prompt_tokens <= 0 ||
      options.prompt_tokens > capture::kCaptureContext)
    return absl::InvalidArgumentError("invalid layer/sample/prompt count");
  if (options.model_width != capture::kCaptureWidth ||
      options.context_length != capture::kCaptureContext)
    return absl::InvalidArgumentError(
        "recording requires model_width=16 and context_length=1024");
  ASSIGN_OR_RETURN(auto original,
                   tokenizer::Gpt2Tokenizer::Load(options.tokenizer.string()));
  ASSIGN_OR_RETURN(auto detokenizer,
                   tokenizer::Gpt2Detokenizer::Load(options.tokenizer.string()));
  ASSIGN_OR_RETURN(
      auto vocabulary,
      tokenizer::CompactVocabularyTokenizer::LoadFromFile(
          *original, options.checkpoint / "compact_vocabulary.tsv"));
  if (vocabulary->original_eos_token_id() != original->eos_token_id() ||
      original->vocab_size() != detokenizer->vocab_size() ||
      original->eos_token_id() != detokenizer->eos_token_id())
    return absl::InvalidArgumentError("checkpoint tokenizer identity mismatch");
  const capture::CaptureOptions capture_options{
      .layers = options.layers,
      .vocab_size = vocabulary->vocab_size(),
      .eos_token = vocabulary->eos_token_id(),
      .prompt_tokens = options.prompt_tokens};
  const Gpt2Config config{.transformer_block_count = options.layers,
                          .model_width = options.model_width,
                          .attention_heads = options.attention_heads,
                          .feed_forward_width = options.feed_forward_width,
                          .vocabulary_size = vocabulary->vocab_size(),
                          .pad_vocabulary = false};
  RETURN_IF_ERROR(config.Validate());
  ASSIGN_OR_RETURN(auto corpus, ReadFile(options.corpus));
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto data, PaddedLineDataSetIterator::Create(
                                  *executor, corpus, *vocabulary,
                                  {.batch_size = 1,
                                   .context_length = options.context_length,
                                   .prompt_tokens = options.prompt_tokens,
                                   .eos_token = vocabulary->eos_token_id(),
                                   .shuffle = false}));
  if (data->sample_count() != static_cast<size_t>(options.expected_samples))
    return absl::InvalidArgumentError(
        absl::StrCat("expected ", options.expected_samples,
                     " corpus samples, got ", data->sample_count()));
  ASSIGN_OR_RETURN(auto model, CreateGpt2(*executor, DataType::BF16, 0, config));
  RETURN_IF_ERROR(ReadFromDirectory(*executor, *model,
                                    options.checkpoint.string(),
                                    /*allow_prefix=*/false));
  ModelMetadata metadata{.width = options.model_width,
                         .layers = options.layers,
                         .vocab_size = vocabulary->vocab_size(),
                         .eos_token = vocabulary->eos_token_id(),
                         .prompt_tokens = options.prompt_tokens};
  for (int id : vocabulary->original_token_ids()) {
    ASSIGN_OR_RETURN(auto bytes, detokenizer->Decode({&id, 1}));
    metadata.vocabulary.push_back({id, std::move(bytes)});
  }
  std::vector<ExecutionSample> samples;
  samples.reserve(data->sample_count());
  size_t rows = 0;
  for (size_t index = 0; index < data->sample_count(); ++index) {
    ASSIGN_OR_RETURN(
        auto sample,
        capture::CaptureSample(*executor, *model, data->sample_tokens(index),
                               capture_options));
    auto status = capture::ValidateCapturedPredictions(sample, capture_options);
    if (status.ok() && options.verify_greedy)
      status = capture::VerifyGreedyCapture(*executor, *model, sample,
                                            capture_options);
    if (!status.ok())
      return absl::Status(status.code(), absl::StrCat("sample ", index, ": ",
                                                      status.message()));
    rows += sample.tokens.size();
    ExecutionSample observed{.tokens = std::move(sample.tokens),
                             .predictions = std::move(sample.predictions)};
    observed.boundaries.reserve(sample.boundaries.size());
    for (const auto& boundary : sample.boundaries) {
      auto& observed_rows = observed.boundaries.emplace_back();
      observed_rows.reserve(boundary.size());
      for (const auto& row : boundary)
        observed_rows.emplace_back(row.begin(), row.end());
    }
    samples.push_back(std::move(observed));
    if (options.progress &&
        ((index + 1) % 32 == 0 || index + 1 == data->sample_count()))
      options.progress(CaptureProgress{
          .samples = static_cast<int64_t>(index + 1),
          .total_samples = static_cast<int64_t>(data->sample_count()),
          .rows = static_cast<int64_t>(rows),
          .greedy_verified = options.verify_greedy});
  }
  RETURN_IF_ERROR(executor->Synchronize());
  // Hints are published only after the full symbolic model validates, leaving
  // an existing caller-owned hint map untouched when recording fails.
  StateVectorHints recorded_hints;
  ASSIGN_OR_RETURN(
      auto captured,
      BuildModel(metadata, samples, options.expected_samples,
                 vector_hints == nullptr ? nullptr : &recorded_hints));
  if (vector_hints != nullptr)
    *vector_hints = std::move(recorded_hints);
  return captured;
}
}  // namespace pluto::llm::discretized::generator
