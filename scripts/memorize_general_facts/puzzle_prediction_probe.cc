// Teacher-forced diagnostics for a saved puzzle readout; never trains a model.
// Build: bazel build -c opt
// //scripts/memorize_general_facts:puzzle_prediction_probe Run the binary with
// --source_checkpoint=... --readout_checkpoint=...
// --tokenizer=... --corpus=testdata/general_facts_dataset.txt
// --output=/tmp/predictions.tsv (the output must not already exist).
// Optional --attention_states_output=/tmp/states.tsv adds exact A3/A4 rows.
// Indices and compact vocabulary IDs are zero-based; target_rank is one-based.
// Text fields escape backslash, tab, LF, CR, and other non-ASCII/control bytes
// as \\, \t, \n, \r, and \xHH respectively, preserving split UTF-8 tokens.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_detokenizer.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/padded_line_dataset.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/memorize_general_facts/puzzle_readout.h"
#include "src/llm/gpt2.h"
#include "src/llm/layer.h"
#include "src/llm/layer_hooks.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, source_checkpoint, "", "Frozen source checkpoint");
ABSL_FLAG(std::string, readout_checkpoint, "",
          "Eight-tensor readout checkpoint");
ABSL_FLAG(std::string, tokenizer, "", "Base GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt",
          "Fact corpus");
ABSL_FLAG(std::string, output, "", "New output TSV file");
ABSL_FLAG(std::string, attention_states_output, "",
          "Optional new TSV of scored post-attention A3/A4 BF16 coordinates");
ABSL_FLAG(int, layers, 4, "Source transformer blocks");
ABSL_FLAG(int, model_width, 10, "Source residual width");
ABSL_FLAG(int, feed_forward_width, 20, "Source MLP expansion width");
ABSL_FLAG(int, attention_heads, 1, "Source attention heads");
ABSL_FLAG(int, context_length, 27, "Source context length");
ABSL_FLAG(int, mlp_width, 150, "Readout MLP expansion width");
ABSL_FLAG(int, seed, 3, "Fresh readout comparison initialization seed");
ABSL_FLAG(int, batch_size, 32, "Facts per diagnostic batch");

namespace pluto::llm::memorize_general_facts {
namespace {

std::string Escape(absl::string_view text) {
  constexpr char hex[] = "0123456789abcdef";
  std::string escaped;
  for (unsigned char byte : text) {
    switch (byte) {
      case '\\':
        escaped += "\\\\";
        break;
      case '\t':
        escaped += "\\t";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      default:
        if (byte < 32 || byte >= 127) {
          escaped += "\\x";
          escaped += hex[byte >> 4];
          escaped += hex[byte & 15];
        } else {
          escaped += static_cast<char>(byte);
        }
    }
  }
  return escaped;
}

// Transfers are queued into pinned storage and synchronized once per batch.
template <class T>
absl::StatusOr<cuda::PageLockedHostArray<T>> Download(cuda::Executor& executor,
                                                      const Buffer& buffer) {
  if (buffer.size_bytes() % sizeof(T) != 0)
    return absl::DataLossError("unexpected diagnostic buffer element size");
  ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<T>::Allocate(
                                  executor, buffer.size_bytes() / sizeof(T)));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemcpyAsync(host.data(), buffer.data(), buffer.size_bytes(),
                      cudaMemcpyDeviceToHost, executor.stream()),
      "download diagnostic buffer"));
  return host;
}

struct AttentionStates {
  Buffer a3;
  Buffer a4;
};

// The first residual in each block is after attention and before that block's
// MLP. Keep both buffers from one full forward, using identical source tokens.
absl::StatusOr<AttentionStates> CaptureAttentionStates(cuda::Executor& executor,
                                                       const Layer& source,
                                                       const Gpt2Config& config,
                                                       const Buffer& tokens) {
  if (config.transformer_block_count < 4 || source.name() != "gpt2" ||
      source.output_type() != DataType::BF16)
    return absl::InvalidArgumentError(
        "A3/A4 capture requires a four-block BF16 GPT-2");
  if (&tokens.executor() != &executor || tokens.size_bytes() == 0 ||
      tokens.size_bytes() % (config.context_length * sizeof(int)) != 0)
    return absl::InvalidArgumentError(
        "A3/A4 capture token shape/executor mismatch");
  const size_t rows = tokens.size_bytes() / sizeof(int);
  const size_t expected_bytes = rows * config.model_width * sizeof(uint16_t);
  const ActivationType expected_type(
      DataType::BF16, {-2, config.context_length, config.model_width});
  std::vector<std::string> scopes;
  std::optional<Buffer> a3;
  std::optional<Buffer> a4;
  LayerHooks hooks;
  hooks.enter_combinator = [&](auto&, auto name) {
    scopes.emplace_back(name);
    return absl::OkStatus();
  };
  hooks.exit_combinator = [&](auto&) {
    if (scopes.empty())
      return absl::DataLossError("unbalanced A3/A4 capture scope");
    scopes.pop_back();
    return absl::OkStatus();
  };
  hooks.activation_hook = [&](auto&, auto name, auto types, auto buffers) {
    if (name != "ResidualLayer" || scopes.size() != 2 || scopes[0] != "gpt2")
      return absl::OkStatus();
    std::optional<Buffer>* destination = nullptr;
    if (scopes[1] == "transformer_block_2")
      destination = &a3;
    if (scopes[1] == "transformer_block_3")
      destination = &a4;
    if (destination == nullptr || destination->has_value())
      return absl::OkStatus();
    if (types.size() != 1 || types[0] != expected_type || buffers.size() != 1 ||
        buffers[0].size_bytes() != expected_bytes ||
        &buffers[0].executor() != &executor)
      return absl::DataLossError("unexpected A3/A4 capture type/shape");
    *destination = buffers[0];
    return absl::OkStatus();
  };
  ASSIGN_OR_RETURN(auto forward, source.fwd(executor, {tokens}, &hooks));
  forward.state = BackwardState{};
  if (!a3 || !a4 || !scopes.empty())
    return absl::DataLossError("missing A3/A4 post-attention residual capture");
  return AttentionStates{std::move(*a3), std::move(*a4)};
}

struct RowMetrics {
  int prediction = -1;
  int target_rank = 1;
  double target_probability = 0;
  double top1_probability = 0;
  double margin = 0;
};

// Only the logical vocabulary participates. Lowest ID wins an exact tie,
// matching ExtractTop1Ids; rank uses the same tie ordering.
absl::StatusOr<RowMetrics> ScoreRow(absl::Span<const float> logits,
                                    int target) {
  if (target < 0 || static_cast<size_t>(target) >= logits.size())
    return absl::DataLossError("diagnostic target is outside the vocabulary");
  RowMetrics result;
  double maximum = -std::numeric_limits<double>::infinity();
  double largest_other = maximum;
  const double target_logit = logits[target];
  for (size_t token = 0; token < logits.size(); ++token) {
    const double logit = logits[token];
    if (!std::isfinite(logit))
      return absl::DataLossError("nonfinite diagnostic logit");
    if (logit > maximum) {
      maximum = logit;
      result.prediction = static_cast<int>(token);
    }
    if (static_cast<int>(token) != target)
      largest_other = std::max(largest_other, logit);
    if (logit > target_logit ||
        (logit == target_logit && static_cast<int>(token) < target))
      ++result.target_rank;
  }
  double denominator = 0;
  for (float logit : logits)
    denominator += std::exp(static_cast<double>(logit) - maximum);
  result.target_probability = std::exp(target_logit - maximum) / denominator;
  result.top1_probability = 1.0 / denominator;
  result.margin = target_logit - largest_other;
  return result;
}

absl::Status RunProbe() {
  const std::string source_path = absl::GetFlag(FLAGS_source_checkpoint);
  const std::string readout_path = absl::GetFlag(FLAGS_readout_checkpoint);
  const std::string tokenizer_path = absl::GetFlag(FLAGS_tokenizer);
  const std::string output_path = absl::GetFlag(FLAGS_output);
  const std::string states_path = absl::GetFlag(FLAGS_attention_states_output);
  if (source_path.empty() || readout_path.empty() || tokenizer_path.empty() ||
      output_path.empty() || absl::GetFlag(FLAGS_batch_size) <= 0)
    return absl::InvalidArgumentError(
        "source_checkpoint, readout_checkpoint, tokenizer, and output are "
        "required; batch_size must be positive");
  std::error_code error;
  if (std::filesystem::exists(output_path, error))
    return absl::AlreadyExistsError("diagnostic output file must be new");
  if (error)
    return absl::InternalError(error.message());
  if (!states_path.empty()) {
    if (absl::GetFlag(FLAGS_layers) < 4)
      return absl::InvalidArgumentError(
          "attention state output requires at least four blocks");
    if (std::filesystem::exists(states_path, error))
      return absl::AlreadyExistsError(
          "attention state output file must be new");
    if (error)
      return absl::InternalError(error.message());
    const auto absolute_states =
        std::filesystem::weakly_canonical(states_path, error);
    if (error)
      return absl::InternalError(error.message());
    const auto absolute_output =
        std::filesystem::weakly_canonical(output_path, error);
    if (error)
      return absl::InternalError(error.message());
    if (absolute_states == absolute_output)
      return absl::InvalidArgumentError(
          "prediction and attention outputs must be different files");
  }
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto base_tokenizer,
                   tokenizer::Gpt2Tokenizer::Load(tokenizer_path));
  ASSIGN_OR_RETURN(auto detokenizer,
                   tokenizer::Gpt2Detokenizer::Load(tokenizer_path));
  ASSIGN_OR_RETURN(auto compact,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *base_tokenizer, std::filesystem::path(source_path) /
                                            "compact_vocabulary.tsv"));
  Gpt2Config config{
      .transformer_block_count = absl::GetFlag(FLAGS_layers),
      .model_width = absl::GetFlag(FLAGS_model_width),
      .attention_heads = absl::GetFlag(FLAGS_attention_heads),
      .feed_forward_width = absl::GetFlag(FLAGS_feed_forward_width),
      .vocabulary_size = compact->vocab_size(),
      .pad_vocabulary = false,
      .context_length = absl::GetFlag(FLAGS_context_length)};
  RETURN_IF_ERROR(config.Validate());
  ASSIGN_OR_RETURN(auto source,
                   CreateGpt2(*executor, DataType::BF16, 0, config));
  RETURN_IF_ERROR(ReadFromDirectory(*executor, *source, source_path, false));
  ASSIGN_OR_RETURN(auto fresh,
                   CreatePuzzleReadout(*executor, *source, config,
                                       absl::GetFlag(FLAGS_mlp_width),
                                       absl::GetFlag(FLAGS_seed)));
  ASSIGN_OR_RETURN(auto trained,
                   CreatePuzzleReadout(*executor, *source, config,
                                       absl::GetFlag(FLAGS_mlp_width),
                                       absl::GetFlag(FLAGS_seed)));
  RETURN_IF_ERROR(
      ReadFromDirectory(*executor, *trained.trainable, readout_path, false));
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  ASSIGN_OR_RETURN(auto data,
                   PaddedLineDataSetIterator::Create(
                       *executor, corpus.text(), *compact,
                       {.batch_size = absl::GetFlag(FLAGS_batch_size),
                        .context_length = config.context_length,
                        .prompt_tokens = 5,
                        .eos_token = compact->eos_token_id()}));
  if (data->sample_count() != 1024)
    return absl::InvalidArgumentError(
        "diagnostic corpus must contain 1024 facts");
  std::vector<std::string> token_text(config.vocabulary_size);
  for (int token = 0; token < config.vocabulary_size; ++token) {
    ASSIGN_OR_RETURN(int original, compact->OriginalId(token));
    ASSIGN_OR_RETURN(auto decoded, detokenizer->Decode({&original, 1}));
    token_text[token] = Escape(decoded);
  }
  std::ofstream output(output_path);
  if (!output)
    return absl::InternalError("cannot create diagnostic TSV");
  output
      << "fact_index\tposition\ttarget\tprediction\tsource_prediction\t"
         "fresh_prediction\tinput_token\tprefix\ttarget_text\tprediction_text\t"
         "target_probability\ttop1_probability\ttarget_rank\tmargin\t"
         "source_target_probability\tsource_margin\thidden_norm\n"
      << std::setprecision(17);
  std::ofstream states_output;
  if (!states_path.empty()) {
    states_output.open(states_path);
    if (!states_output)
      return absl::InternalError("cannot create attention state TSV");
    states_output << "fact_index\tposition\ttarget\tinput_token";
    for (int dimension = 0; dimension < config.model_width; ++dimension)
      states_output << "\ta3_" << dimension;
    for (int dimension = 0; dimension < config.model_width; ++dimension)
      states_output << "\ta4_" << dimension;
    states_output << '\n' << std::setprecision(17);
  }
  const int stride = ((config.vocabulary_size + 15) / 16) * 16;
  size_t first_fact = 0;
  int64_t scored = 0;
  int64_t correct = 0;
  int64_t source_correct = 0;
  int64_t fresh_correct = 0;
  RETURN_IF_ERROR(data->Reset());
  for (size_t batch_index = 0; batch_index < data->batches_per_epoch();
       ++batch_index) {
    ASSIGN_OR_RETURN(auto batch, data->Next());
    ASSIGN_OR_RETURN(auto captured,
                     CaptureThirdAttention(*executor, *source, batch.inputs));
    ASSIGN_OR_RETURN(auto trained_forward,
                     trained.model->fwd(*executor, {captured.hidden}));
    trained_forward.state = BackwardState{};
    ASSIGN_OR_RETURN(auto fresh_forward,
                     fresh.model->fwd(*executor, {captured.hidden}));
    fresh_forward.state = BackwardState{};
    ASSIGN_OR_RETURN(auto targets, Download<int>(*executor, batch.targets));
    ASSIGN_OR_RETURN(auto trained_logits,
                     Download<float>(*executor, trained_forward.outputs[0]));
    ASSIGN_OR_RETURN(auto source_logits,
                     Download<float>(*executor, captured.logits));
    ASSIGN_OR_RETURN(auto fresh_logits,
                     Download<float>(*executor, fresh_forward.outputs[0]));
    ASSIGN_OR_RETURN(auto hidden,
                     Download<uint16_t>(*executor, captured.hidden));
    cuda::PageLockedHostArray<uint16_t> a3_check;
    cuda::PageLockedHostArray<uint16_t> a4_states;
    if (!states_path.empty()) {
      ASSIGN_OR_RETURN(
          auto attention,
          CaptureAttentionStates(*executor, *source, config, batch.inputs));
      ASSIGN_OR_RETURN(a3_check, Download<uint16_t>(*executor, attention.a3));
      ASSIGN_OR_RETURN(a4_states, Download<uint16_t>(*executor, attention.a4));
    }
    RETURN_IF_ERROR(executor->Synchronize());
    const size_t rows =
        static_cast<size_t>(batch.batch_size) * config.context_length;
    if (targets.size() != rows || trained_logits.size() != rows * stride ||
        source_logits.size() != rows * stride ||
        fresh_logits.size() != rows * stride ||
        hidden.size() != rows * config.model_width)
      return absl::DataLossError("unexpected probe buffer shape");
    if (!states_path.empty() &&
        (a3_check.size() != hidden.size() ||
         a4_states.size() != hidden.size() ||
         !std::equal(hidden.begin(), hidden.end(), a3_check.begin())))
      return absl::DataLossError(
          "A3 capture differs from CaptureThirdAttention");
    for (int sample = 0; sample < batch.batch_size; ++sample) {
      const size_t fact_index = first_fact + sample;
      const auto tokens = data->sample_tokens(fact_index);
      std::vector<int> original_tokens;
      original_tokens.reserve(tokens.size());
      for (int token : tokens) {
        ASSIGN_OR_RETURN(int original, compact->OriginalId(token));
        original_tokens.push_back(original);
      }
      for (int position = 0; position < config.context_length; ++position) {
        const size_t row =
            static_cast<size_t>(sample) * config.context_length + position;
        const int target = targets[row];
        if (target < 0)
          continue;
        ASSIGN_OR_RETURN(auto trained_score,
                         ScoreRow(trained_logits.span().subspan(
                                      row * stride, config.vocabulary_size),
                                  target));
        ASSIGN_OR_RETURN(auto source_score,
                         ScoreRow(source_logits.span().subspan(
                                      row * stride, config.vocabulary_size),
                                  target));
        ASSIGN_OR_RETURN(auto fresh_score,
                         ScoreRow(fresh_logits.span().subspan(
                                      row * stride, config.vocabulary_size),
                                  target));
        ASSIGN_OR_RETURN(
            auto prefix,
            detokenizer->Decode(
                absl::MakeConstSpan(original_tokens).first(position + 1)));
        double squared_norm = 0;
        for (int dimension = 0; dimension < config.model_width; ++dimension) {
          const uint32_t bits =
              static_cast<uint32_t>(
                  hidden[row * config.model_width + dimension])
              << 16;
          const double value = std::bit_cast<float>(bits);
          if (!std::isfinite(value))
            return absl::DataLossError("nonfinite captured hidden value");
          squared_norm += value * value;
        }
        output << fact_index << '\t' << position << '\t' << target << '\t'
               << trained_score.prediction << '\t' << source_score.prediction
               << '\t' << fresh_score.prediction << '\t' << tokens[position]
               << '\t' << Escape(prefix) << '\t' << token_text[target] << '\t'
               << token_text[trained_score.prediction] << '\t'
               << trained_score.target_probability << '\t'
               << trained_score.top1_probability << '\t'
               << trained_score.target_rank << '\t' << trained_score.margin
               << '\t' << source_score.target_probability << '\t'
               << source_score.margin << '\t' << std::sqrt(squared_norm)
               << '\n';
        if (!states_path.empty()) {
          states_output << fact_index << '\t' << position << '\t' << target
                        << '\t' << tokens[position];
          for (const auto* values : {&a3_check, &a4_states}) {
            for (int dimension = 0; dimension < config.model_width;
                 ++dimension) {
              const uint32_t bits =
                  static_cast<uint32_t>(
                      (*values)[row * config.model_width + dimension])
                  << 16;
              const float value = std::bit_cast<float>(bits);
              if (!std::isfinite(value))
                return absl::DataLossError("nonfinite A3/A4 coordinate");
              states_output << '\t' << value;
            }
          }
          states_output << '\n';
        }
        ++scored;
        correct += trained_score.prediction == target;
        source_correct += source_score.prediction == target;
        fresh_correct += fresh_score.prediction == target;
      }
    }
    first_fact += batch.batch_size;
  }
  output.close();
  if (!output)
    return absl::InternalError("writing diagnostic TSV failed");
  if (!states_path.empty()) {
    states_output.close();
    if (!states_output)
      return absl::InternalError("writing attention state TSV failed");
  }
  if (first_fact != data->sample_count() ||
      scored != data->supervised_row_count())
    return absl::InternalError(
        "diagnostic probe did not visit every scored row");
  std::cout << "facts=" << first_fact << " scored=" << scored
            << " correct=" << correct << " source_correct=" << source_correct
            << " fresh_correct=" << fresh_correct << " output=" << output_path
            << '\n';
  if (!states_path.empty())
    std::cout << "attention_states=" << states_path << " rows=" << scored
              << " width=" << config.model_width << " a3_byte_equal=true\n";
  return absl::OkStatus();
}

}  // namespace
}  // namespace pluto::llm::memorize_general_facts

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const auto status = pluto::llm::memorize_general_facts::RunProbe();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
