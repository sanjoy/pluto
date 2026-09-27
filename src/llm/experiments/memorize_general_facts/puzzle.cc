#include "src/llm/experiments/memorize_general_facts/puzzle.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/padded_line_dataset.h"
#include "src/dataset/tokenizer.h"
#include "src/llm/adamw_optimizer.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/memorize_general_facts/puzzle_readout.h"
#include "src/llm/experiments/memorize_general_facts/puzzle_report.h"
#include "src/llm/extract_top1_ids.h"
#include "src/llm/layers/cross_entropy_loss.h"
#include "src/util/status_macros.h"

namespace pluto::llm::memorize_general_facts {
namespace {

constexpr int kFactCount = 1024;
constexpr int kPromptTokens = 5;

absl::Status Copy(cuda::Executor& executor, void* destination,
                  const void* source, size_t bytes, cudaMemcpyKind kind) {
  return cuda::CudaStatus(
      cudaMemcpyAsync(destination, source, bytes, kind, executor.stream()),
      "copy puzzle data");
}

// Parameters are small here. Snapshot every tensor, including aliases, to
// prove that capture, fitting, and greedy verification never alter the source.
absl::StatusOr<cuda::PageLockedHostArray<uint8_t>> SnapshotWeights(
    cuda::Executor& executor, const Layer& layer) {
  size_t bytes = 0;
  for (const auto& weight : layer.weights()) {
    if (weight.size_bytes() > std::numeric_limits<size_t>::max() - bytes)
      return absl::InvalidArgumentError("weight snapshot size overflows");
    bytes += weight.size_bytes();
  }
  ASSIGN_OR_RETURN(auto result, cuda::PageLockedHostArray<uint8_t>::Allocate(
                                    executor, bytes));
  size_t offset = 0;
  for (const auto& weight : layer.weights()) {
    RETURN_IF_ERROR(Copy(executor, result.data() + offset, weight.data(),
                         weight.size_bytes(), cudaMemcpyDeviceToHost));
    offset += weight.size_bytes();
  }
  RETURN_IF_ERROR(executor.Synchronize());
  return result;
}

absl::Status CheckUnchanged(cuda::Executor& executor, const Layer& layer,
                            const cuda::PageLockedHostArray<uint8_t>& before) {
  ASSIGN_OR_RETURN(auto after, SnapshotWeights(executor, layer));
  if (before.size() != after.size() ||
      !std::equal(before.begin(), before.end(), after.begin()))
    return absl::InternalError("a frozen puzzle parameter changed");
  return absl::OkStatus();
}

// Exact BF16 rows are retained on-device for every training/evaluation pass.
// Host coordinates are only for the exact-collision audit and standalone plot.
struct CapturedCorpus {
  Buffer hidden;
  Buffer targets;
  cuda::PageLockedHostArray<int> host_targets;
  PuzzleReportData report;
};

absl::StatusOr<CapturedCorpus> CaptureCorpus(cuda::Executor& executor,
                                             const Layer& source,
                                             const Gpt2Config& config,
                                             PaddedLineDataSetIterator& data,
                                             absl::string_view text, int eos,
                                             std::ostream& output) {
  const int samples = data.sample_count();
  const int sequence = config.context_length;
  const size_t rows = static_cast<size_t>(samples) * sequence;
  const size_t row_bytes = config.model_width * sizeof(uint16_t);
  ASSIGN_OR_RETURN(auto hidden, Buffer::Allocate(executor, rows * row_bytes));
  ASSIGN_OR_RETURN(auto targets, Buffer::Allocate(executor, rows * sizeof(int)));
  ASSIGN_OR_RETURN(auto host_targets,
                   cuda::PageLockedHostArray<int>::Allocate(executor, rows));
  ASSIGN_OR_RETURN(auto predictions,
                   cuda::PageLockedHostArray<int>::Allocate(executor, rows));
  RETURN_IF_ERROR(data.Reset());
  size_t offset_rows = 0;
  for (size_t i = 0; i < data.batches_per_epoch(); ++i) {
    ASSIGN_OR_RETURN(auto batch, data.Next());
    ASSIGN_OR_RETURN(auto captured,
                     CaptureThirdAttention(executor, source, batch.inputs));
    const size_t batch_rows = static_cast<size_t>(batch.batch_size) * sequence;
    if (captured.hidden.size_bytes() != batch_rows * row_bytes)
      return absl::DataLossError("unexpected third-attention capture shape");
    RETURN_IF_ERROR(Copy(
        executor, static_cast<char*>(hidden.data()) + offset_rows * row_bytes,
        captured.hidden.data(), captured.hidden.size_bytes(),
        cudaMemcpyDeviceToDevice));
    RETURN_IF_ERROR(Copy(executor,
                         static_cast<int*>(targets.data()) + offset_rows,
                         batch.targets.data(), batch_rows * sizeof(int),
                         cudaMemcpyDeviceToDevice));
    RETURN_IF_ERROR(Copy(executor, host_targets.data() + offset_rows,
                         batch.targets.data(), batch_rows * sizeof(int),
                         cudaMemcpyDeviceToHost));
    ASSIGN_OR_RETURN(auto ids,
                     ExtractTop1Ids(executor, captured.logits, batch.targets,
                                    config.vocabulary_size));
    RETURN_IF_ERROR(Copy(executor, predictions.data() + offset_rows, ids.data(),
                         batch_rows * sizeof(int), cudaMemcpyDeviceToHost));
    offset_rows += batch_rows;
  }
  ASSIGN_OR_RETURN(auto host_hidden,
                   cuda::PageLockedHostArray<uint16_t>::Allocate(
                       executor, rows * config.model_width));
  RETURN_IF_ERROR(Copy(executor, host_hidden.data(), hidden.data(),
                       hidden.size_bytes(), cudaMemcpyDeviceToHost));
  RETURN_IF_ERROR(executor.Synchronize());
  if (offset_rows != rows)
    return absl::InternalError("puzzle capture did not visit every sample");

  int scored = 0;
  int wrong = 0;
  for (size_t row = 0; row < rows; ++row)
    if (host_targets[row] >= 0) {
      ++scored;
      wrong += predictions[row] != host_targets[row];
    }
  output << "Original model: correct=" << scored - wrong << "/" << scored
         << " scored next tokens\n"
         << std::flush;
  if (wrong != 0)
    return absl::FailedPreconditionError(
        "puzzle requires a checkpoint that memorizes all corpus completions");

  PuzzleReportData report;
  report.model_width = config.model_width;
  report.context_length = sequence;
  if (!text.empty() && text.back() == '\n')
    text.remove_suffix(1);
  report.facts = absl::StrSplit(text, '\n');
  for (auto& fact : report.facts)
    if (!fact.empty() && fact.back() == '\r')
      fact.pop_back();
  report.points.reserve(rows);
  for (int sample = 0; sample < samples; ++sample) {
    const auto tokens = data.sample_tokens(sample);
    for (int position = 0; position < sequence; ++position) {
      const size_t row = static_cast<size_t>(sample) * sequence + position;
      const bool padding = static_cast<size_t>(position) >= tokens.size();
      PuzzlePoint point;
      point.fact_index = sample;
      point.position = position;
      point.input_token = padding ? eos : tokens[position];
      point.target_token = host_targets[row];
      point.padding = padding;
      point.coordinates.reserve(config.model_width);
      for (int column = 0; column < config.model_width; ++column) {
        const uint32_t bits =
            uint32_t{host_hidden[row * config.model_width + column]} << 16;
        point.coordinates.push_back(std::bit_cast<float>(bits));
      }
      report.points.push_back(std::move(point));
    }
  }
  return CapturedCorpus{std::move(hidden), std::move(targets),
                        std::move(host_targets), std::move(report)};
}

// Allocate only the full and possible final partial minibatch once. Loading a
// batch then performs D2D copies, never host round trips or fresh allocations.
struct Batch {
  Buffer hidden;
  Buffer targets;
  int samples;
};

absl::StatusOr<Batch> AllocateBatch(cuda::Executor& executor,
                                    const Gpt2Config& config, int samples) {
  const size_t rows = static_cast<size_t>(samples) * config.context_length;
  ASSIGN_OR_RETURN(auto hidden,
                   Buffer::Allocate(executor, rows * config.model_width * 2));
  ASSIGN_OR_RETURN(auto targets, Buffer::Allocate(executor, rows * sizeof(int)));
  return Batch{std::move(hidden), std::move(targets), samples};
}

absl::Status LoadBatch(cuda::Executor& executor, const CapturedCorpus& cache,
                       const Gpt2Config& config, absl::Span<const int> indices,
                       Batch& batch) {
  if (indices.size() != static_cast<size_t>(batch.samples))
    return absl::InternalError("puzzle batch size mismatch");
  const size_t hidden_bytes = static_cast<size_t>(config.context_length) *
                              config.model_width * sizeof(uint16_t);
  const size_t target_bytes = config.context_length * sizeof(int);
  for (size_t sample = 0; sample < indices.size(); ++sample) {
    const size_t source = indices[sample];
    RETURN_IF_ERROR(Copy(
        executor,
        static_cast<char*>(batch.hidden.data()) + sample * hidden_bytes,
        static_cast<const char*>(cache.hidden.data()) + source * hidden_bytes,
        hidden_bytes, cudaMemcpyDeviceToDevice));
    RETURN_IF_ERROR(Copy(
        executor,
        static_cast<char*>(batch.targets.data()) + sample * target_bytes,
        static_cast<const char*>(cache.targets.data()) + source * target_bytes,
        target_bytes, cudaMemcpyDeviceToDevice));
  }
  return absl::OkStatus();
}

struct Metrics {
  int wrong = 0;        // Errors on scored suffix/EOS targets only.
  int complete = 0;     // Facts with all teacher-forced targets correct.
  int scored = 0;       // Denominator excluding prompt and padding.
  int eos_correct = 0;  // Correct EOS targets, not EOS-padding rows.
  int eos_scored = 0;   // One real terminal target per fact.
  double mean_ce = 0;   // Mean standard cross entropy over scored rows.
};

absl::StatusOr<Metrics> EvaluateReadout(cuda::Executor& executor,
                                        const Layer& readout, const Layer& loss,
                                        const CapturedCorpus& cache,
                                        const Gpt2Config& config, Batch& full,
                                        std::optional<Batch>& partial,
                                        int eos) {
  const int samples = cache.report.facts.size();
  const int sequence = config.context_length;
  const int max_rows = full.samples * sequence;
  ASSIGN_OR_RETURN(auto host_ids,
                   cuda::PageLockedHostArray<int>::Allocate(executor, max_rows));
  ASSIGN_OR_RETURN(auto host_loss, cuda::PageLockedHostArray<float>::Allocate(
                                       executor, max_rows));
  std::vector<int> indices(full.samples);
  Metrics metrics;
  for (int start = 0; start < samples; start += full.samples) {
    const int count = std::min(full.samples, samples - start);
    Batch& batch = count == full.samples ? full : *partial;
    std::iota(indices.begin(), indices.begin() + count, start);
    RETURN_IF_ERROR(LoadBatch(executor, cache, config,
                              absl::MakeConstSpan(indices).first(count),
                              batch));
    ASSIGN_OR_RETURN(auto forward, readout.fwd(executor, {batch.hidden}));
    ASSIGN_OR_RETURN(auto predictions,
                     ExtractTop1Ids(executor, forward.outputs[0], batch.targets,
                                    config.vocabulary_size));
    ASSIGN_OR_RETURN(auto losses,
                     loss.fwd(executor, {forward.outputs[0], batch.targets}));
    const size_t rows = static_cast<size_t>(count) * sequence;
    RETURN_IF_ERROR(Copy(executor, host_ids.data(), predictions.data(),
                         rows * sizeof(int), cudaMemcpyDeviceToHost));
    RETURN_IF_ERROR(Copy(executor, host_loss.data(), losses.outputs[0].data(),
                         rows * sizeof(float), cudaMemcpyDeviceToHost));
    RETURN_IF_ERROR(executor.Synchronize());
    for (int sample = 0; sample < count; ++sample) {
      bool complete = true;
      for (int position = 0; position < sequence; ++position) {
        const int local = sample * sequence + position;
        const int target =
            cache.host_targets[(start + sample) * sequence + position];
        if (target < 0)
          continue;  // Supplied prompt and EOS-padding rows are not targets.
        if (!std::isfinite(host_loss[local]))
          return absl::DataLossError("nonfinite puzzle cross entropy");
        ++metrics.scored;
        metrics.mean_ce += host_loss[local];
        metrics.wrong += host_ids[local] != target;
        metrics.eos_scored += target == eos;
        metrics.eos_correct += target == eos && host_ids[local] == target;
        complete &= host_ids[local] == target;
      }
      metrics.complete += complete;
    }
  }
  if (metrics.scored == 0)
    return absl::DataLossError("puzzle evaluation has no scored targets");
  metrics.mean_ce /= metrics.scored;
  return metrics;
}

// Actual prefixes, not cached future-token sequences, establish exact greedy
// completion. A first mistake is sufficient to disprove a fact's completion.
absl::StatusOr<int> VerifyGreedy(cuda::Executor& executor, const Layer& source,
                                 const Layer& readout, const Gpt2Config& config,
                                 const PaddedLineDataSetIterator& data,
                                 int eos) {
  const int sequence = config.context_length;
  ASSIGN_OR_RETURN(auto host_tokens,
                   cuda::PageLockedHostArray<int>::Allocate(executor, sequence));
  ASSIGN_OR_RETURN(auto host_ids,
                   cuda::PageLockedHostArray<int>::Allocate(executor, sequence));
  ASSIGN_OR_RETURN(auto tokens,
                   Buffer::Allocate(executor, host_tokens.size_bytes()));
  ASSIGN_OR_RETURN(auto mask,
                   Buffer::Allocate(executor, host_tokens.size_bytes()));
  RETURN_IF_ERROR(cuda::CudaStatus(
      cudaMemsetAsync(mask.data(), 0, mask.size_bytes(), executor.stream()),
      "initialize greedy mask"));
  int complete = 0;
  for (size_t sample = 0; sample < data.sample_count(); ++sample) {
    const auto expected = data.sample_tokens(sample);
    std::fill(host_tokens.begin(), host_tokens.end(), eos);
    std::copy_n(expected.begin(), kPromptTokens, host_tokens.begin());
    bool correct = true;
    for (size_t row = kPromptTokens - 1; row < expected.size(); ++row) {
      RETURN_IF_ERROR(Copy(executor, tokens.data(), host_tokens.data(),
                           tokens.size_bytes(), cudaMemcpyHostToDevice));
      ASSIGN_OR_RETURN(auto captured,
                       CaptureThirdAttention(executor, source, tokens));
      ASSIGN_OR_RETURN(auto forward, readout.fwd(executor, {captured.hidden}));
      ASSIGN_OR_RETURN(auto ids, ExtractTop1Ids(executor, forward.outputs[0],
                                                mask, config.vocabulary_size));
      RETURN_IF_ERROR(Copy(executor, host_ids.data(), ids.data(),
                           ids.size_bytes(), cudaMemcpyDeviceToHost));
      RETURN_IF_ERROR(executor.Synchronize());
      const int target = row + 1 < expected.size() ? expected[row + 1] : eos;
      if (host_ids[row] != target) {
        correct = false;
        break;
      }
      if (row + 1 < expected.size())
        host_tokens[row + 1] = host_ids[row];
    }
    complete += correct;
  }
  return complete;
}

absl::Status FitReadout(cuda::Executor& executor, const Layer& source,
                        const Gpt2Config& config, const CapturedCorpus& cache,
                        const PaddedLineDataSetIterator& data, int eos,
                        const PuzzleOptions& options, std::ostream& output) {
  ASSIGN_OR_RETURN(
      auto readout,
      CreatePuzzleReadout(executor, source, config, options.mlp_width,
                          options.seed, options.mlp_depth,
                          options.match_mlp_parameter_budget));
  ASSIGN_OR_RETURN(auto frozen_head,
                   SnapshotWeights(executor, *readout.embedding));
  ASSIGN_OR_RETURN(auto loss, CrossEntropyLossLayer::Create(
                                  executor, config.vocabulary_size,
                                  DataType::BF16, config.context_length));
  ASSIGN_OR_RETURN(auto optimizer, AdamWOptimizer::Create(
                                       executor, *readout.trainable,
                                       {.learning_rate = options.learning_rate,
                                        .beta1 = 0.9f,
                                        .beta2 = 0.999f,
                                        .epsilon = 1e-8f,
                                        .weight_decay = 0}));
  RETURN_IF_ERROR(optimizer->ZeroGrad());
  const int samples = cache.report.facts.size();
  const int batch_size = std::min(options.batch_size, samples);
  ASSIGN_OR_RETURN(auto full, AllocateBatch(executor, config, batch_size));
  std::optional<Batch> partial;
  if (samples % batch_size != 0) {
    ASSIGN_OR_RETURN(partial,
                     AllocateBatch(executor, config, samples % batch_size));
  }
  const auto best_path =
      std::filesystem::path(options.output_directory) / "best_mlp";
  std::ofstream statistics(std::filesystem::path(options.output_directory) /
                           "training.tsv");
  if (!statistics)
    return absl::InternalError("cannot create puzzle training statistics");
  statistics << "step\tseconds\tmean_ce\tcorrect\tscored\tcomplete_facts"
                "\teos_correct\teos_scored\n";
  size_t parameters = 0;
  for (const auto& weight : readout.trainable->weights())
    parameters += weight.size_bytes() / sizeof(float);
  const auto& budget = readout.parameter_budget;
  output << "Parameter budget: original post-A3 suffix="
         << budget.source_tail_parameters
         << "; replacement affine MLP=" << budget.mlp_parameters
         << "; requested_mlp_width=" << options.mlp_width
         << "; minimum_mlp_width=" << budget.minimum_mlp_width
         << "; resolved_mlp_width=" << budget.mlp_width
         << "; mlp_depth=" << options.mlp_depth
         << "; match_mlp_parameter_budget="
         << options.match_mlp_parameter_budget
         << " (shared frozen embedding/head excluded)\n"
         << "Trainable: " << options.mlp_depth << " residual "
         << config.model_width << " -> " << budget.mlp_width << " -> "
         << config.model_width
         << " MLP(s), each with input LN, one final LN; parameters="
         << parameters
         << "; frozen: original transformer and tied embedding head\n"
         << "Adam: learning_rate=" << options.learning_rate
         << ", cosine final ratio=0.1, no clipping or weight decay\n";
  const auto start = std::chrono::steady_clock::now();
  auto elapsed = [&] {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start)
        .count();
  };
  Metrics best;
  best.wrong = std::numeric_limits<int>::max();
  int best_step = 0;
  auto evaluate = [&](int step) -> absl::Status {
    ASSIGN_OR_RETURN(auto metrics,
                     EvaluateReadout(executor, *readout.model, *loss, cache,
                                     config, full, partial, eos));
    const double seconds = elapsed();
    output << absl::StrFormat(
                  "step=%d seconds=%.2f mean_ce=%.8f correct=%d/%d "
                  "token_accuracy=%.4f%% teacher_forced_complete=%d/%d\n",
                  step, seconds, metrics.mean_ce,
                  metrics.scored - metrics.wrong, metrics.scored,
                  100.0 * (metrics.scored - metrics.wrong) / metrics.scored,
                  metrics.complete, samples)
           << std::flush;
    statistics << absl::StrFormat("%d\t%.2f\t%.8f\t%d\t%d\t%d\t%d\t%d\n", step,
                                  seconds, metrics.mean_ce,
                                  metrics.scored - metrics.wrong,
                                  metrics.scored, metrics.complete,
                                  metrics.eos_correct, metrics.eos_scored)
               << std::flush;
    if (!statistics || !output)
      return absl::InternalError("writing puzzle statistics failed");
    if (metrics.wrong < best.wrong ||
        (metrics.wrong == best.wrong && metrics.mean_ce < best.mean_ce)) {
      best = metrics;
      best_step = step;
      RETURN_IF_ERROR(
          WriteToDirectory(executor, *readout.trainable, best_path));
    }
    return absl::OkStatus();
  };
  RETURN_IF_ERROR(evaluate(0));
  std::vector<int> order(samples);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 random(options.seed);
  int cursor = samples;
  for (int64_t update = 1; update <= options.steps; ++update) {
    const int step = static_cast<int>(update);
    if (cursor == samples) {
      std::shuffle(order.begin(), order.end(), random);
      cursor = 0;
    }
    const int count = std::min(batch_size, samples - cursor);
    Batch& batch = count == batch_size ? full : *partial;
    RETURN_IF_ERROR(LoadBatch(executor, cache, config,
                              absl::MakeConstSpan(order).subspan(cursor, count),
                              batch));
    cursor += count;
    const float progress = static_cast<float>(step - 1) / options.steps;
    RETURN_IF_ERROR(optimizer->SetLearningRate(
        options.learning_rate *
        (0.1f +
         0.9f * 0.5f * (1 + std::cos(3.14159265358979323846f * progress)))));
    ASSIGN_OR_RETURN(auto forward, readout.model->fwd(executor, {batch.hidden}));
    ASSIGN_OR_RETURN(auto loss_forward,
                     loss->fwd(executor, {forward.outputs[0], batch.targets}));
    ASSIGN_OR_RETURN(auto gradients,
                     loss->bwd(executor, {}, std::move(loss_forward.state)));
    // The tied head propagates activation gradients but is outside Adam.
    // Its parameter accumulators are not allowed to grow across updates.
    for (const auto& gradient : readout.model->gradients().subspan(
             readout.trainable->weights().size()))
      RETURN_IF_ERROR(cuda::CudaStatus(
          cudaMemsetAsync(gradient.data(), 0, gradient.size_bytes(),
                          executor.stream()),
          "clear frozen head gradients"));
    RETURN_IF_ERROR(
        readout.model->bwd(executor, gradients, std::move(forward.state))
            .status());
    RETURN_IF_ERROR(optimizer->ApplyStep());
    if (step % options.eval_every == 0)
      RETURN_IF_ERROR(evaluate(step));
    if (step % 32 == 0)
      RETURN_IF_ERROR(
          executor.Synchronize());  // Bound the asynchronous backlog.
  }
  if (options.steps % options.eval_every != 0)
    RETURN_IF_ERROR(evaluate(options.steps));
  RETURN_IF_ERROR(
      ReadFromDirectory(executor, *readout.trainable, best_path, false));
  ASSIGN_OR_RETURN(
      const int complete,
      VerifyGreedy(executor, source, *readout.model, config, data, eos));
  RETURN_IF_ERROR(CheckUnchanged(executor, *readout.embedding, frozen_head));
  output
      << absl::StrFormat(
             "BEST step=%d updates=%d seconds=%.2f mean_ce=%.8f correct=%d/%d "
             "token_accuracy=%.4f%% greedy_complete=%d/%d "
             "frozen_head_unchanged=true\n",
             best_step, options.steps, elapsed(), best.mean_ce,
             best.scored - best.wrong, best.scored,
             100.0 * (best.scored - best.wrong) / best.scored, complete,
             samples)
      << "EOS breakdown: eos_correct=" << best.eos_correct
      << " eos_scored=" << best.eos_scored
      << " non_eos_correct=" << best.scored - best.wrong - best.eos_correct
      << " non_eos_scored=" << best.scored - best.eos_scored << "\n"
      << "Best readout: " << best_path.string() << "\n"
      << std::flush;
  return output ? absl::OkStatus()
                : absl::InternalError("writing final puzzle statistics failed");
}

}  // namespace

absl::Status RunPuzzle(cuda::Executor& executor,
                       const tokenizer::Gpt2Tokenizer& base_tokenizer,
                       const PuzzleOptions& options, std::ostream& output) {
  if (options.checkpoint.empty() || options.corpus.empty() ||
      options.output_directory.empty() || options.batch_size <= 0 ||
      options.model_config.transformer_block_count < 3 ||
      (options.train_mlp &&
       (options.steps < 0 || options.eval_every <= 0 ||
        options.mlp_width <= 0 || options.mlp_depth <= 0 || options.seed < 0 ||
        !std::isfinite(options.learning_rate) || options.learning_rate <= 0)))
    return absl::InvalidArgumentError("invalid puzzle options");
  std::error_code error;
  if (std::filesystem::exists(options.output_directory, error))
    return absl::AlreadyExistsError("puzzle output directory must be new");
  if (error)
    return absl::InternalError(
        absl::StrCat("checking puzzle output: ", error.message()));
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(options.corpus));
  std::unique_ptr<tokenizer::CompactVocabularyTokenizer> compact;
  const tokenizer::Tokenizer* tokenizer = &base_tokenizer;
  int eos = base_tokenizer.eos_token_id();
  if (options.compact_vocabulary) {
    ASSIGN_OR_RETURN(
        compact, tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                     base_tokenizer, std::filesystem::path(options.checkpoint) /
                                         "compact_vocabulary.tsv"));
    tokenizer = compact.get();
    eos = compact->eos_token_id();
  }
  Gpt2Config config = options.model_config;
  config.vocabulary_size = tokenizer->vocab_size();
  config.pad_vocabulary = !options.compact_vocabulary;
  RETURN_IF_ERROR(config.Validate());
  // Resolve capacity before allocating/capturing the corpus. Snapshot-only
  // mode ignores readout settings, just as its CLI policy requires.
  std::optional<PuzzleReadoutParameterBudget> budget;
  if (options.train_mlp) {
    ASSIGN_OR_RETURN(budget, ResolvePuzzleReadoutParameterBudget(
                                 config, options.mlp_width, options.mlp_depth,
                                 options.match_mlp_parameter_budget));
  }
  ASSIGN_OR_RETURN(auto data, PaddedLineDataSetIterator::Create(
                                  executor, corpus.text(), *tokenizer,
                                  {.batch_size = options.batch_size,
                                   .context_length = config.context_length,
                                   .prompt_tokens = kPromptTokens,
                                   .eos_token = eos}));
  if (data->sample_count() != kFactCount)
    return absl::InvalidArgumentError("puzzle requires exactly 1024 facts");
  ASSIGN_OR_RETURN(auto source, CreateGpt2(executor, DataType::BF16, 0, config));
  RETURN_IF_ERROR(
      ReadFromDirectory(executor, *source, options.checkpoint, false));
  ASSIGN_OR_RETURN(auto frozen_source, SnapshotWeights(executor, *source));
  ASSIGN_OR_RETURN(auto captured,
                   CaptureCorpus(executor, *source, config, *data,
                                 corpus.text(), eos, output));
  ASSIGN_OR_RETURN(auto separation, VerifyPuzzleSeparation(captured.report));
  output
      << "Verified A3 separation: no identical hidden vectors have different "
         "scored next-token targets; scored="
      << separation.scored_points
      << " unique_vectors=" << separation.unique_vectors
      << " distinct_targets=" << separation.distinct_targets << "\n"
      << "This exact finite-corpus check covers suffix+EOS after five-token "
         "prompts, not prompt-only/padding rows or linear separability.\n";
  const bool created =
      std::filesystem::create_directories(options.output_directory, error);
  if (error)
    return absl::InternalError(
        absl::StrCat("creating puzzle output: ", error.message()));
  if (!created)
    return absl::AlreadyExistsError(
        "puzzle output directory was created by another run");
  const auto html =
      std::filesystem::path(options.output_directory) / "puzzle.html";
  RETURN_IF_ERROR(WritePuzzleHtml(captured.report, separation, html.string()));
  std::ofstream provenance(std::filesystem::path(options.output_directory) /
                           "run.txt");
  provenance << "checkpoint=" << options.checkpoint
             << "\ncorpus=" << options.corpus
             << "\nsource_layers=" << config.transformer_block_count
             << "\nmodel_width=" << config.model_width
             << "\nsource_mlp_width=" << config.feed_forward_width
             << "\nattention_heads=" << config.attention_heads
             << "\ncontext_length=" << config.context_length
             << "\nvocabulary_size=" << config.vocabulary_size
             << "\ntrain_mlp=" << options.train_mlp
             << "\nrequested_mlp_width=" << options.mlp_width << "\nmlp_width="
             << (budget ? budget->mlp_width : options.mlp_width)
             << "\nmlp_depth=" << options.mlp_depth
             << "\nmatch_mlp_parameter_budget="
             << options.match_mlp_parameter_budget
             << "\nsteps=" << options.steps
             << "\neval_every=" << options.eval_every
             << "\nbatch_size=" << options.batch_size
             << "\nseed=" << options.seed
             << "\nlearning_rate=" << options.learning_rate << "\n";
  if (budget)
    provenance << "minimum_mlp_width=" << budget->minimum_mlp_width
               << "\nsource_tail_parameters=" << budget->source_tail_parameters
               << "\nmlp_parameters=" << budget->mlp_parameters
               << "\ntrainable_parameters=" << budget->trainable_parameters
               << "\n";
  provenance.close();
  if (!provenance)
    return absl::InternalError("writing puzzle provenance failed");
  output << "Plots: " << html.string() << "\n" << std::flush;
  if (options.train_mlp)
    RETURN_IF_ERROR(FitReadout(executor, *source, config, captured, *data, eos,
                               options, output));
  RETURN_IF_ERROR(CheckUnchanged(executor, *source, frozen_source));
  output << "Frozen source parameters unchanged.\n" << std::flush;
  return output ? absl::OkStatus()
                : absl::InternalError("writing puzzle summary failed");
}

}  // namespace pluto::llm::memorize_general_facts
