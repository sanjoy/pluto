// Tests whether a single-fact embedding update acts mainly through a shared
// row shift, row-specific changes, position embeddings, or the learned blocks.
// These are interventions on two supplied checkpoints, not fact-ownership or
// information-entropy measurements. Every condition starts from trained bytes.
#include <cuda_runtime_api.h>

#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/padded_line_dataset.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/one_shot_memorizer/mlp_probe.h"
#include "src/llm/experiments/one_shot_memorizer/sentence_ablation.h"
#include "src/llm/gpt2.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, initial_checkpoint, "",
          "Original initialization checkpoint");
ABSL_FLAG(std::string, trained_checkpoint, "",
          "Single-fact trained checkpoint");
ABSL_FLAG(std::string, tokenizer, "", "Base GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt",
          "Full corpus");
ABSL_FLAG(int, selected_line, 631, "One-based corpus line to probe");
ABSL_FLAG(std::string, output_dir, "",
          "Fresh directory for intervention reports");
ABSL_FLAG(
    bool, evaluate_full_corpus, false,
    "Also evaluate every sentence for each condition; substantially slower");
ABSL_FLAG(int, full_corpus_batch_size, 32,
          "Batch size for optional full-corpus evaluation");
ABSL_FLAG(int, prompt_tokens, 5, "Prefix tokens supplied to each completion");
ABSL_FLAG(int, layers, 8, "Transformer block count");
ABSL_FLAG(int, model_width, 16, "Residual width");
ABSL_FLAG(int, attention_heads, 1, "Attention head count");
ABSL_FLAG(int, feed_forward_width, 64, "MLP expansion width");

namespace pluto::llm::one_shot_memorizer {
namespace {

namespace fs = std::filesystem;
using HostTensor = cuda::PageLockedHostArray<float>;

struct Evaluation {
  MlpEvaluation teacher;
  MlpGreedyEvaluation greedy;
};

absl::StatusOr<std::vector<Buffer>> UniqueWeights(
    cuda::Executor& executor, Layer& model,
    absl::Span<const TensorSpec> layout) {
  absl::flat_hash_set<const void*> seen;
  std::vector<Buffer> weights;
  for (const Buffer& weight : model.weights()) {
    if (&weight.executor() != &executor)
      return absl::InvalidArgumentError("weight executor mismatch");
    // The language-modeling head aliases the input embedding. Count it once,
    // preserving the first-occurrence order used by checkpoint I/O.
    if (seen.insert(weight.data()).second)
      weights.push_back(weight);
  }
  if (weights.size() != layout.size())
    return absl::InvalidArgumentError(
        "checkpoint tensor count differs from layout");
  for (size_t i = 0; i < weights.size(); ++i)
    if (weights[i].size_bytes() != layout[i].element_count * sizeof(float))
      return absl::InvalidArgumentError(
          absl::StrCat("tensor shape mismatch: ", i));
  return weights;
}

absl::StatusOr<std::vector<HostTensor>> CopyD2H(
    cuda::Executor& executor, absl::Span<const Buffer> weights) {
  std::vector<HostTensor> result;
  result.reserve(weights.size());
  for (const auto& weight : weights) {
    ASSIGN_OR_RETURN(
        auto host,
        HostTensor::Allocate(executor, weight.size_bytes() / sizeof(float)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), weight.data(), host.size_bytes(),
                        cudaMemcpyDeviceToHost, executor.stream()),
        "snapshot single-fact probe weights"));
    result.push_back(std::move(host));
  }
  RETURN_IF_ERROR(executor.Synchronize());
  return result;
}

absl::Status Upload(cuda::Executor& executor, absl::Span<const float> values,
                    Buffer& weight) {
  if (values.size() * sizeof(float) != weight.size_bytes())
    return absl::InternalError("intervention changed tensor shape");
  ASSIGN_OR_RETURN(auto host, HostTensor::CopyFrom(executor, values));
  // Staging storage is pinned; its release is ordered after this stream copy.
  return cuda::CudaStatus(
      cudaMemcpyAsync(weight.data(), host.data(), host.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "upload single-fact intervention");
}

absl::Status Restore(cuda::Executor& executor,
                     absl::Span<const HostTensor> originals,
                     absl::Span<Buffer> weights) {
  if (originals.size() != weights.size())
    return absl::InternalError("restore changed tensor count");
  for (size_t i = 0; i < weights.size(); ++i) {
    if (originals[i].size_bytes() != weights[i].size_bytes())
      return absl::InternalError("restore changed tensor shape");
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(weights[i].data(), originals[i].data(),
                        originals[i].size_bytes(), cudaMemcpyHostToDevice,
                        executor.stream()),
        "restore all trained single-fact weights"));
  }
  return absl::OkStatus();
}

absl::Status ValidateSnapshot(absl::Span<const TensorSpec> layout,
                              absl::Span<const HostTensor> snapshot) {
  std::vector<absl::Span<const float>> views;
  for (const auto& tensor : snapshot)
    views.push_back(tensor.span());
  return ValidateParameterValues(layout, views);
}

// Broadcast a double-precision column shift, then round once to the FP32 master
// representation. GPU inference subsequently performs its normal BF16 casts.
absl::StatusOr<std::vector<float>> ShiftRows(absl::Span<const float> source,
                                             absl::Span<const double> shift,
                                             double sign) {
  if (shift.empty() || source.size() % shift.size() != 0)
    return absl::InternalError("column shift does not fit its matrix");
  std::vector<float> values(source.size());
  for (size_t i = 0; i < values.size(); ++i) {
    const double value = source[i] + sign * shift[i % shift.size()];
    if (!std::isfinite(value) ||
        std::abs(value) > std::numeric_limits<float>::max())
      return absl::OutOfRangeError("embedding intervention overflows FP32");
    values[i] = static_cast<float>(value);
  }
  return values;
}

absl::StatusOr<Evaluation> Evaluate(cuda::Executor& executor,
                                    const Layer& model,
                                    PaddedLineDataSetIterator& dataset,
                                    int vocabulary_size) {
  Evaluation result;
  ASSIGN_OR_RETURN(
      result.teacher,
      EvaluateMlpReplacements(executor, model, dataset, {}, vocabulary_size));
  ASSIGN_OR_RETURN(
      result.greedy,
      VerifyMlpGreedyCompletions(executor, model, dataset, {}, vocabulary_size));
  if (result.teacher.sentences !=
          static_cast<int64_t>(dataset.sample_count()) ||
      result.greedy.sentences != result.teacher.sentences ||
      result.teacher.targets != dataset.supervised_row_count() ||
      result.teacher.targets_per_sentence.size() != dataset.sample_count() ||
      result.teacher.correct_per_sentence.size() != dataset.sample_count() ||
      result.greedy.exact_per_sentence.size() != dataset.sample_count())
    return absl::InternalError("single-fact evaluation cardinalities changed");
  for (size_t i = 0; i < dataset.sample_count(); ++i)
    if ((result.teacher.targets_per_sentence[i] ==
         result.teacher.correct_per_sentence[i]) !=
        result.greedy.exact_per_sentence[i])
      return absl::InternalError(
          "teacher and greedy sentence exactness disagree");
  return result;
}

bool SameResults(const Evaluation& a, const Evaluation& b) {
  return a.teacher.targets_per_sentence == b.teacher.targets_per_sentence &&
         a.teacher.correct_per_sentence == b.teacher.correct_per_sentence &&
         a.greedy.exact_per_sentence == b.greedy.exact_per_sentence &&
         a.greedy.generated_targets == b.greedy.generated_targets;
}

absl::Status Run() {
  const fs::path initial_path = absl::GetFlag(FLAGS_initial_checkpoint);
  const fs::path trained_path = absl::GetFlag(FLAGS_trained_checkpoint);
  const fs::path output_dir = absl::GetFlag(FLAGS_output_dir);
  const int selected_line = absl::GetFlag(FLAGS_selected_line);
  if (initial_path.empty() || trained_path.empty() || output_dir.empty() ||
      absl::GetFlag(FLAGS_tokenizer).empty() || selected_line <= 0 ||
      absl::GetFlag(FLAGS_full_corpus_batch_size) <= 0)
    return absl::InvalidArgumentError(
        "initial_checkpoint, trained_checkpoint, tokenizer, fresh output_dir, "
        "positive selected_line and full_corpus_batch_size are required");

  ASSIGN_OR_RETURN(
      auto base, tokenizer::Gpt2Tokenizer::Load(absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto tokenizer,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *base, trained_path / "compact_vocabulary.tsv"));
  RETURN_IF_ERROR(
      tokenizer->ValidateFile(initial_path / "compact_vocabulary.tsv"));
  if (tokenizer->original_eos_token_id() != base->eos_token_id())
    return absl::InvalidArgumentError(
        "base tokenizer EOS differs from checkpoint");
  const Gpt2Config config{
      .transformer_block_count = absl::GetFlag(FLAGS_layers),
      .model_width = absl::GetFlag(FLAGS_model_width),
      .attention_heads = absl::GetFlag(FLAGS_attention_heads),
      .feed_forward_width = absl::GetFlag(FLAGS_feed_forward_width),
      .vocabulary_size = tokenizer->vocab_size(),
      .pad_vocabulary = false};
  RETURN_IF_ERROR(config.Validate());
  ASSIGN_OR_RETURN(
      auto layout,
      BuildGpt2ParameterLayout(
          {.vocabulary_size = config.vocabulary_size,
           .model_width = config.model_width,
           .feed_forward_width = config.feed_forward_width,
           .context_length = kGpt2ContextLength,
           .transformer_block_count = config.transformer_block_count}));
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  absl::string_view text = corpus.text();
  if (!text.empty() && text.back() == '\n')
    text.remove_suffix(1);
  std::vector<absl::string_view> lines = absl::StrSplit(text, '\n');
  for (auto& line : lines) {
    if (!line.empty() && line.back() == '\r')
      line.remove_suffix(1);
    if (absl::StripAsciiWhitespace(line).empty())
      return absl::InvalidArgumentError("corpus contains an empty/blank line");
  }
  if (static_cast<size_t>(selected_line) > lines.size())
    return absl::InvalidArgumentError("selected_line is outside the corpus");
  const std::string selected_text(lines[selected_line - 1]);
  PaddedLineDataSetOptions options{
      .batch_size = 1,
      .context_length = kGpt2ContextLength,
      .prompt_tokens = absl::GetFlag(FLAGS_prompt_tokens),
      .eos_token = tokenizer->eos_token_id(),
      .shuffle = false};
  ASSIGN_OR_RETURN(auto selected,
                   PaddedLineDataSetIterator::Create(*executor, selected_text,
                                                     *tokenizer, options));
  std::unique_ptr<PaddedLineDataSetIterator> full;
  if (absl::GetFlag(FLAGS_evaluate_full_corpus)) {
    options.batch_size = absl::GetFlag(FLAGS_full_corpus_batch_size);
    ASSIGN_OR_RETURN(full, PaddedLineDataSetIterator::Create(
                               *executor, corpus.text(), *tokenizer, options));
    if (full->sample_count() != lines.size())
      return absl::InternalError("full dataset changed corpus line numbering");
  }

  ASSIGN_OR_RETURN(auto model, CreateGpt2(*executor, DataType::BF16, 0, config));
  RETURN_IF_ERROR(
      ReadFromDirectory(*executor, *model, initial_path.string(), false));
  ASSIGN_OR_RETURN(auto weights, UniqueWeights(*executor, *model, layout));
  ASSIGN_OR_RETURN(auto initial, CopyD2H(*executor, weights));
  RETURN_IF_ERROR(ValidateSnapshot(layout, initial));
  RETURN_IF_ERROR(
      ReadFromDirectory(*executor, *model, trained_path.string(), false));
  ASSIGN_OR_RETURN(auto trained, CopyD2H(*executor, weights));
  RETURN_IF_ERROR(ValidateSnapshot(layout, trained));
  if (layout.size() < 2 || !layout[0].token_embedding ||
      layout[1].name != "position_embedding.weight")
    return absl::InternalError("embedding layout does not match GPT-2");

  const size_t width = static_cast<size_t>(config.model_width);
  const size_t vocabulary_size = static_cast<size_t>(config.vocabulary_size);
  std::vector<double> mean(width, 0);
  double total_squared = 0;
  size_t parameters = 0;
  for (const auto& tensor : trained)
    parameters += tensor.size();
  for (size_t i = 0; i < trained[0].size(); ++i) {
    const double delta = static_cast<double>(trained[0][i]) - initial[0][i];
    mean[i % width] += delta;
    total_squared += delta * delta;
  }
  double common_squared = 0;
  for (double& value : mean) {
    value /= vocabulary_size;
    common_squared += vocabulary_size * value * value;
  }
  double residual_squared = 0;
  for (size_t i = 0; i < trained[0].size(); ++i) {
    const double residual =
        static_cast<double>(trained[0][i]) - initial[0][i] - mean[i % width];
    residual_squared += residual * residual;
  }
  ASSIGN_OR_RETURN(auto no_common, ShiftRows(trained[0].span(), mean, -1));
  ASSIGN_OR_RETURN(auto compensated_positions,
                   ShiftRows(trained[1].span(), mean, 1));
  ASSIGN_OR_RETURN(auto only_common, ShiftRows(initial[0].span(), mean, 1));

  std::error_code error;
  if (!fs::create_directory(output_dir, error))
    return absl::InvalidArgumentError(
        absl::StrCat("output_dir must be fresh: ", error.message()));
  std::ofstream manifest(output_dir / "manifest.txt");
  std::ofstream shifts(output_dir / "embedding_common_shift.tsv");
  std::ofstream conditions(output_dir / "conditions.tsv");
  std::ofstream per_sentence(output_dir / "per_sentence.tsv");
  if (!manifest || !shifts || !conditions || !per_sentence)
    return absl::UnknownError("cannot create single-fact reports");
  for (auto* stream : {&manifest, &shifts, &conditions, &per_sentence})
    *stream << std::setprecision(std::numeric_limits<double>::max_digits10);
  manifest << "initial_checkpoint=" << initial_path.string()
           << "\ntrained_checkpoint=" << trained_path.string()
           << "\ncorpus=" << absl::GetFlag(FLAGS_corpus)
           << "\ntokenizer=" << absl::GetFlag(FLAGS_tokenizer)
           << "\nselected_line=" << selected_line
           << "\nselected_text=" << selected_text
           << "\ncorpus_lines=" << lines.size()
           << "\nprompt_tokens=" << options.prompt_tokens
           << "\ncontext_length=" << kGpt2ContextLength
           << "\nselected_batch_size=1"
           << "\nevaluate_full_corpus=" << static_cast<bool>(full)
           << "\nfull_corpus_batch_size="
           << absl::GetFlag(FLAGS_full_corpus_batch_size)
           << "\nvocabulary_size=" << vocabulary_size
           << "\nmodel_width=" << width
           << "\nlayers=" << config.transformer_block_count
           << "\nattention_heads=" << config.attention_heads
           << "\nfeed_forward_width=" << config.feed_forward_width
           << "\nunique_tensors=" << trained.size()
           << "\nparameters=" << parameters
           << "\nembedding_parameters=" << trained[0].size()
           << "\nposition_parameters=" << trained[1].size()
           << "\ncommon_shift_parameters=" << mean.size()
           << "\nembedding_delta_squared_norm=" << total_squared
           << "\ncommon_shift_squared_norm=" << common_squared
           << "\nrow_residual_squared_norm=" << residual_squared
           << "\ncommon_shift_squared_norm_fraction=";
  if (total_squared != 0)
    manifest << common_squared / total_squared;
  else
    manifest << "undefined (zero embedding delta)";
  manifest
      << "\n\nDelta = trained - initial. Mean is over all vocabulary rows.\n"
         "E_initial + row_residual equals E_trained - mean, so there is no "
         "duplicate condition.\n"
         "A common embedding shift adds one identical scalar to every head "
         "logit for FIXED hidden input,\n"
         "which cancels from exact-real softmax. Tied input embeddings change "
         "hidden states;\n"
         "BF16 operand/output rounding also prevents assuming full-model "
         "invariance.\n"
         "Position compensation restores E+P only in exact arithmetic, not "
         "necessarily in the GPU kernel.\n"
         "All conditions restore every trained parameter before intervention; "
         "no training occurs.\n"
         "The supplied initial checkpoint and single-fact training provenance "
         "are caller assertions.\n"
         "The shared coordinates describe an update, not the entire model "
         "or its information entropy.\n"
         "Greedy uses only the prefix plus its own predictions; "
         "generated_targets stops on first error.\n";
  shifts << "column\tmean_trained_minus_initial\n";
  for (size_t column = 0; column < mean.size(); ++column)
    shifts << column << '\t' << mean[column] << '\n';
  std::cout << "Embedding delta squared norm=" << total_squared
            << " shared-row-shift squared norm=" << common_squared
            << " residual squared norm=" << residual_squared << std::endl;
  if (total_squared != 0)
    std::cout << "Shared-row-shift fraction=" << common_squared / total_squared
              << std::endl;
  manifest.flush();
  shifts.flush();
  conditions << "condition\tscope\tchanged_tensors\tchanged_master_"
                "values\ttargets\tcorrect_targets"
                "\tsentences\tteacher_exact\tgreedy_exact\tgreedy_generated_"
                "targets\tseconds\n";
  per_sentence << "condition\tscope\tcorpus_line\ttargets\tcorrect_"
                  "targets\tteacher_exact\tgreedy_exact\n";

  Evaluation selected_baseline;
  Evaluation full_baseline;
  const std::vector<std::string> names{
      "trained_baseline",
      "initial_baseline",
      "embedding_common_shift_removed",
      "embedding_common_shift_removed_positions_compensated",
      "embedding_initial_plus_common_shift",
      "embedding_restored_initial",
      "positions_restored_initial",
      "only_embedding_trained",
      "embedding_and_positions_restored_initial",
      "restored_trained"};
  for (size_t condition = 0; condition < names.size(); ++condition) {
    RETURN_IF_ERROR(Restore(*executor, trained, absl::MakeSpan(weights)));
    switch (condition) {
      case 1:
        RETURN_IF_ERROR(Restore(*executor, initial, absl::MakeSpan(weights)));
        break;
      case 2:
      case 3:
        RETURN_IF_ERROR(Upload(*executor, no_common, weights[0]));
        if (condition == 3)
          RETURN_IF_ERROR(Upload(*executor, compensated_positions, weights[1]));
        break;
      case 4:
        RETURN_IF_ERROR(Upload(*executor, only_common, weights[0]));
        break;
      case 5:
        RETURN_IF_ERROR(Upload(*executor, initial[0].span(), weights[0]));
        break;
      case 6:
        RETURN_IF_ERROR(Upload(*executor, initial[1].span(), weights[1]));
        break;
      case 7:
        for (size_t tensor = 1; tensor < weights.size(); ++tensor)
          RETURN_IF_ERROR(
              Upload(*executor, initial[tensor].span(), weights[tensor]));
        break;
      case 8:
        RETURN_IF_ERROR(Upload(*executor, initial[0].span(), weights[0]));
        RETURN_IF_ERROR(Upload(*executor, initial[1].span(), weights[1]));
        break;
      default:
        break;
    }
    ASSIGN_OR_RETURN(auto current, CopyD2H(*executor, weights));
    size_t changed_tensors = 0;
    size_t changed_values = 0;
    for (size_t tensor = 0; tensor < current.size(); ++tensor) {
      size_t changed = 0;
      for (size_t i = 0; i < current[tensor].size(); ++i)
        changed += std::bit_cast<uint32_t>(current[tensor][i]) !=
                   std::bit_cast<uint32_t>(trained[tensor][i]);
      changed_tensors += changed != 0;
      changed_values += changed;
    }
    if ((condition == 0 || condition + 1 == names.size()) &&
        changed_values != 0)
      return absl::InternalError(
          "trained parameter restoration was not bit-identical");

    auto evaluate = [&](PaddedLineDataSetIterator& dataset, const char* scope,
                        Evaluation& baseline) -> absl::Status {
      const auto started = std::chrono::steady_clock::now();
      ASSIGN_OR_RETURN(auto result, Evaluate(*executor, *model, dataset,
                                             config.vocabulary_size));
      const double seconds = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();
      conditions << names[condition] << '\t' << scope << '\t' << changed_tensors
                 << '\t' << changed_values << '\t' << result.teacher.targets
                 << '\t' << result.teacher.correct_targets << '\t'
                 << result.teacher.sentences << '\t'
                 << result.teacher.exact_sentences << '\t'
                 << result.greedy.exact_sentences << '\t'
                 << result.greedy.generated_targets << '\t' << seconds << '\n';
      for (size_t i = 0; i < dataset.sample_count(); ++i)
        per_sentence << names[condition] << '\t' << scope << '\t'
                     << (&dataset == selected.get()
                             ? static_cast<size_t>(selected_line)
                             : i + 1)
                     << '\t' << result.teacher.targets_per_sentence[i] << '\t'
                     << result.teacher.correct_per_sentence[i] << '\t'
                     << (result.teacher.targets_per_sentence[i] ==
                         result.teacher.correct_per_sentence[i])
                     << '\t' << result.greedy.exact_per_sentence[i] << '\n';
      conditions.flush();
      per_sentence.flush();
      if (!conditions || !per_sentence)
        return absl::UnknownError("cannot write single-fact evaluation");
      std::cout << names[condition] << " [" << scope
                << "] correct=" << result.teacher.correct_targets << '/'
                << result.teacher.targets
                << " greedy_exact=" << result.greedy.exact_sentences << '/'
                << result.greedy.sentences << " seconds=" << seconds
                << std::endl;
      if (condition == 0)
        baseline = result;
      if (condition + 1 == names.size() && !SameResults(result, baseline))
        return absl::InternalError(
            "restored trained evaluation differs from baseline");
      return absl::OkStatus();
    };
    RETURN_IF_ERROR(evaluate(*selected, "selected", selected_baseline));
    if (full)
      RETURN_IF_ERROR(evaluate(*full, "full_corpus", full_baseline));
  }
  manifest
      << "complete=true\nrestored_trained_bytes_and_results_verified=true\n";
  for (auto* stream : {&manifest, &shifts, &conditions, &per_sentence}) {
    stream->close();
    if (!*stream)
      return absl::UnknownError("cannot close single-fact reports");
  }
  return absl::OkStatus();
}

}  // namespace
}  // namespace pluto::llm::one_shot_memorizer

int main(int argc, char** argv) {
  if (absl::ParseCommandLine(argc, argv).size() != 1) {
    std::cerr << "unexpected positional arguments\n";
    return 1;
  }
  const auto status = pluto::llm::one_shot_memorizer::Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
