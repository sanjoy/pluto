// Swaps matched checkpoint components in both directions to test whether the
// largest sentence-removal embedding deltas affect completion behavior. Row
// ranking uses only parameter deltas, never token labels or evaluation results.
// These post-hoc mixed models may be off distribution: effects do not establish
// exclusive fact ownership, and neither supplied baseline must be perfect.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
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
#include "src/cuda/buffer.h"
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

ABSL_FLAG(std::string, baseline_checkpoint, "", "Full-corpus checkpoint");
ABSL_FLAG(std::string, omitted_checkpoint, "", "Sentence-removed checkpoint");
ABSL_FLAG(std::string, tokenizer, "", "Base GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt",
          "Full corpus");
ABSL_FLAG(int, selected_line, 631, "One-based omitted sentence, for reporting");
ABSL_FLAG(std::string, output_dir, "",
          "Fresh directory for intervention reports");
ABSL_FLAG(int, batch_size, 32, "Samples per full-corpus evaluation batch");
ABSL_FLAG(int, prompt_tokens, 5, "Prefix tokens supplied to each completion");
ABSL_FLAG(int, layers, 8, "Transformer block count");
ABSL_FLAG(int, model_width, 16, "Residual width");
ABSL_FLAG(int, attention_heads, 1, "Attention head count");
ABSL_FLAG(int, feed_forward_width, 64, "MLP expansion width");
ABSL_FLAG(bool, self_test_only, false,
          "Run CPU-only ranking tests without creating a CUDA executor");

namespace pluto::llm::one_shot_memorizer {
namespace {

namespace fs = std::filesystem;
using HostTensor = cuda::PageLockedHostArray<float>;

// A row's rank is independent of the direction in which it is transplanted.
struct RowDelta {
  size_t row;
  double squared_l2;
};

absl::StatusOr<std::vector<RowDelta>> RankEmbeddingRows(
    const TensorDelta& embedding) {
  if (!embedding.tensor.token_embedding || embedding.tensor.shape.size() != 2 ||
      embedding.tensor.shape[1] == 0 ||
      embedding.deltas.size() / embedding.tensor.shape[1] !=
          embedding.tensor.shape[0] ||
      embedding.deltas.size() % embedding.tensor.shape[1] != 0)
    return absl::InvalidArgumentError("invalid embedding delta shape");
  const size_t width = embedding.tensor.shape[1];
  std::vector<RowDelta> rows;
  for (size_t row = 0; row < embedding.tensor.shape[0]; ++row) {
    double squared_l2 = 0;
    for (size_t column = 0; column < width; ++column)
      squared_l2 += std::pow(embedding.deltas[row * width + column], 2);
    if (!std::isfinite(squared_l2))
      return absl::InvalidArgumentError("nonfinite embedding row delta norm");
    rows.push_back({row, squared_l2});
  }
  std::sort(rows.begin(), rows.end(), [](const RowDelta& a, const RowDelta& b) {
    return a.squared_l2 != b.squared_l2 ? a.squared_l2 > b.squared_l2
                                        : a.row < b.row;
  });
  return rows;
}

// The ranking is deliberately tiny and testable without checkpoints or CUDA.
absl::Status TestRanking() {
  TensorDelta delta;
  delta.tensor.token_embedding = true;
  delta.tensor.shape = {5, 2};
  delta.deltas = {3, 4, 0, -5, 6, 0, 0, -0.0, -3, -4};
  ASSIGN_OR_RETURN(auto rows, RankEmbeddingRows(delta));
  const std::vector<size_t> expected{2, 0, 1, 4, 3};
  for (size_t i = 0; i < expected.size(); ++i)
    if (rows[i].row != expected[i])
      return absl::InternalError("L2 ranking or compact-ID tie-break failed");
  for (double& value : delta.deltas)
    value = -value;
  ASSIGN_OR_RETURN(auto reverse, RankEmbeddingRows(delta));
  for (size_t i = 0; i < rows.size(); ++i)
    if (rows[i].row != reverse[i].row ||
        rows[i].squared_l2 != reverse[i].squared_l2)
      return absl::InternalError("ranking changes with transplant direction");
  delta.deltas.pop_back();
  if (RankEmbeddingRows(delta).ok())
    return absl::InternalError("malformed row shape was accepted");
  delta.deltas.push_back(std::numeric_limits<double>::quiet_NaN());
  if (RankEmbeddingRows(delta).ok())
    return absl::InternalError("nonfinite row delta was accepted");
  std::cout << "CPU ranking tests passed; no CUDA executor was created.\n";
  return absl::OkStatus();
}

absl::StatusOr<std::vector<Buffer>> UniqueWeights(
    cuda::Executor& executor, Layer& model,
    absl::Span<const TensorSpec> layout) {
  absl::flat_hash_set<const void*> seen;
  std::vector<Buffer> weights;
  for (const auto& weight : model.weights()) {
    if (&weight.executor() != &executor)
      return absl::InvalidArgumentError("weight executor mismatch");
    // The head aliases the token embedding: transplanting E changes both uses.
    if (seen.insert(weight.data()).second)
      weights.push_back(weight);
  }
  if (weights.size() != layout.size())
    return absl::InvalidArgumentError(
        "unique weight count differs from layout");
  for (size_t i = 0; i < weights.size(); ++i)
    if (weights[i].size_bytes() != layout[i].element_count * sizeof(float))
      return absl::InvalidArgumentError(
          absl::StrCat("tensor shape mismatch: ", i));
  return weights;
}

absl::StatusOr<std::vector<HostTensor>> CopyD2H(
    cuda::Executor& executor, absl::Span<const Buffer> weights) {
  std::vector<HostTensor> result;
  for (const auto& weight : weights) {
    ASSIGN_OR_RETURN(
        auto host,
        HostTensor::Allocate(executor, weight.size_bytes() / sizeof(float)));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), weight.data(), host.size_bytes(),
                        cudaMemcpyDeviceToHost, executor.stream()),
        "snapshot paired-checkpoint weights"));
    result.push_back(std::move(host));
  }
  RETURN_IF_ERROR(executor.Synchronize());
  return result;
}

std::vector<absl::Span<const float>> Views(
    absl::Span<const HostTensor> snapshot) {
  std::vector<absl::Span<const float>> result;
  for (const auto& tensor : snapshot)
    result.push_back(tensor.span());
  return result;
}

absl::Status CopyTensor(cuda::Executor& executor, const HostTensor& source,
                        Buffer& target) {
  if (source.size_bytes() != target.size_bytes())
    return absl::InternalError("transplant tensor size changed");
  return cuda::CudaStatus(
      cudaMemcpyAsync(target.data(), source.data(), source.size_bytes(),
                      cudaMemcpyHostToDevice, executor.stream()),
      "transplant pinned checkpoint tensor");
}

absl::Status Restore(cuda::Executor& executor,
                     absl::Span<const HostTensor> origin,
                     absl::Span<Buffer> weights) {
  if (origin.size() != weights.size())
    return absl::InternalError("restore tensor count changed");
  for (size_t tensor = 0; tensor < weights.size(); ++tensor)
    RETURN_IF_ERROR(CopyTensor(executor, origin[tensor], weights[tensor]));
  return absl::OkStatus();
}

struct Evaluation {
  MlpEvaluation teacher;
  MlpGreedyEvaluation greedy;
};

absl::StatusOr<Evaluation> Evaluate(cuda::Executor& executor,
                                    const Layer& model,
                                    PaddedLineDataSetIterator& data,
                                    int vocab) {
  Evaluation result;
  ASSIGN_OR_RETURN(result.teacher,
                   EvaluateMlpReplacements(executor, model, data, {}, vocab));
  ASSIGN_OR_RETURN(result.greedy,
                   VerifyMlpGreedyCompletions(executor, model, data, {}, vocab));
  if (result.teacher.sentences != static_cast<int64_t>(data.sample_count()) ||
      result.greedy.sentences != result.teacher.sentences ||
      result.teacher.targets != data.supervised_row_count() ||
      result.teacher.targets_per_sentence.size() != data.sample_count() ||
      result.teacher.correct_per_sentence.size() != data.sample_count() ||
      result.greedy.exact_per_sentence.size() != data.sample_count())
    return absl::InternalError("evaluation cardinalities changed");
  for (size_t i = 0; i < data.sample_count(); ++i)
    if ((result.teacher.targets_per_sentence[i] ==
         result.teacher.correct_per_sentence[i]) !=
        result.greedy.exact_per_sentence[i])
      return absl::InternalError("teacher/greedy exactness disagree");
  return result;
}

bool SameResults(const Evaluation& a, const Evaluation& b) {
  return a.teacher.targets == b.teacher.targets &&
         a.teacher.correct_targets == b.teacher.correct_targets &&
         a.teacher.sentences == b.teacher.sentences &&
         a.teacher.exact_sentences == b.teacher.exact_sentences &&
         a.teacher.targets_per_sentence == b.teacher.targets_per_sentence &&
         a.teacher.correct_per_sentence == b.teacher.correct_per_sentence &&
         a.greedy.sentences == b.greedy.sentences &&
         a.greedy.exact_sentences == b.greedy.exact_sentences &&
         a.greedy.exact_per_sentence == b.greedy.exact_per_sentence &&
         a.greedy.generated_targets == b.greedy.generated_targets;
}

absl::Status Run() {
  if (absl::GetFlag(FLAGS_self_test_only))
    return TestRanking();
  const fs::path baseline_path = absl::GetFlag(FLAGS_baseline_checkpoint);
  const fs::path omitted_path = absl::GetFlag(FLAGS_omitted_checkpoint);
  const fs::path output_dir = absl::GetFlag(FLAGS_output_dir);
  const int selected_line = absl::GetFlag(FLAGS_selected_line);
  if (baseline_path.empty() || omitted_path.empty() || output_dir.empty() ||
      absl::GetFlag(FLAGS_tokenizer).empty() || selected_line <= 0 ||
      absl::GetFlag(FLAGS_batch_size) <= 0 ||
      absl::GetFlag(FLAGS_prompt_tokens) <= 0)
    return absl::InvalidArgumentError(
        "checkpoint paths, tokenizer, fresh "
        "output_dir and positive sizes required");
  std::error_code error;
  if (fs::exists(output_dir, error) || error)
    return absl::InvalidArgumentError("output_dir must be fresh");
  ASSIGN_OR_RETURN(
      auto base, tokenizer::Gpt2Tokenizer::Load(absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto tokenizer,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *base, baseline_path / "compact_vocabulary.tsv"));
  RETURN_IF_ERROR(
      tokenizer->ValidateFile(omitted_path / "compact_vocabulary.tsv"));
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
  if (layout.size() < 4 || !layout[0].token_embedding ||
      layout[1].name != "position_embedding.weight" ||
      layout[layout.size() - 2].name != "final_layer_norm.gamma" ||
      layout.back().name != "final_layer_norm.beta")
    return absl::InternalError("GPT-2 transplant groups do not match layout");
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  absl::string_view text = corpus.text();
  if (!text.empty() && text.back() == '\n')
    text.remove_suffix(1);
  std::vector<absl::string_view> lines = absl::StrSplit(text, '\n');
  for (auto& line : lines) {
    if (!line.empty() && line.back() == '\r')
      line.remove_suffix(1);
    if (absl::StripAsciiWhitespace(line).empty())
      return absl::InvalidArgumentError("corpus has an empty/blank line");
  }
  if (static_cast<size_t>(selected_line) > lines.size())
    return absl::InvalidArgumentError("selected_line is outside the corpus");

  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto data,
                   PaddedLineDataSetIterator::Create(
                       *executor, corpus.text(), *tokenizer,
                       {.batch_size = absl::GetFlag(FLAGS_batch_size),
                        .context_length = kGpt2ContextLength,
                        .prompt_tokens = absl::GetFlag(FLAGS_prompt_tokens),
                        .eos_token = tokenizer->eos_token_id(),
                        .shuffle = false}));
  if (data->sample_count() != lines.size())
    return absl::InternalError("dataset changed corpus line numbering");
  ASSIGN_OR_RETURN(auto model, CreateGpt2(*executor, DataType::BF16, 0, config));
  ASSIGN_OR_RETURN(auto weights, UniqueWeights(*executor, *model, layout));
  RETURN_IF_ERROR(
      ReadFromDirectory(*executor, *model, baseline_path.string(), false));
  ASSIGN_OR_RETURN(auto baseline, CopyD2H(*executor, weights));
  RETURN_IF_ERROR(
      ReadFromDirectory(*executor, *model, omitted_path.string(), false));
  ASSIGN_OR_RETURN(auto omitted, CopyD2H(*executor, weights));
  ASSIGN_OR_RETURN(auto delta, CompareParameterValues(layout, Views(baseline),
                                                      Views(omitted), 0));
  ASSIGN_OR_RETURN(auto ranked, RankEmbeddingRows(delta.tensors[0]));

  if (!fs::create_directory(output_dir, error))
    return absl::InvalidArgumentError(
        absl::StrCat("cannot create fresh output_dir: ", error.message()));
  std::ofstream manifest(output_dir / "manifest.txt");
  std::ofstream rows(output_dir / "embedding_rows.tsv");
  std::ofstream conditions(output_dir / "conditions.tsv");
  std::ofstream per_line(output_dir / "per_sentence.tsv");
  std::ofstream tensors(output_dir / "tensor_deltas.tsv");
  for (auto* stream : {&manifest, &rows, &conditions, &per_line, &tensors}) {
    if (!*stream)
      return absl::UnknownError("cannot create paired-checkpoint reports");
    *stream << std::setprecision(std::numeric_limits<double>::max_digits10);
  }
  manifest
      << "baseline_checkpoint=" << baseline_path.string()
      << "\nomitted_checkpoint=" << omitted_path.string()
      << "\ncorpus=" << absl::GetFlag(FLAGS_corpus)
      << "\ntokenizer=" << absl::GetFlag(FLAGS_tokenizer)
      << "\nselected_line=" << selected_line
      << "\nselected_text=" << lines[selected_line - 1]
      << "\nprompt_tokens=" << absl::GetFlag(FLAGS_prompt_tokens)
      << "\ncontext_length=" << kGpt2ContextLength
      << "\nbatch_size=" << absl::GetFlag(FLAGS_batch_size)
      << "\nvocabulary_size=" << config.vocabulary_size
      << "\nmodel_width=" << config.model_width
      << "\nlayers=" << config.transformer_block_count
      << "\nattention_heads=" << config.attention_heads
      << "\nfeed_forward_width=" << config.feed_forward_width
      << "\nunique_tensors=" << weights.size()
      << "\ncorpus_lines=" << lines.size()
      << "\n\nRanking: descending FP32-master row L2 difference; compact-ID "
         "tie-break.\n"
         "Ranking precedes ID labeling and completion evaluation.\n"
         "Embedding transplants modify tied input embeddings AND output head.\n"
         "Every condition restores all recipient master bytes first. No "
         "training or checkpoint writes.\n"
         "All transformer parameters exclude positions, token embeddings and "
         "final norm.\n"
         "Checkpoint matching, initialization, schedule, and omitted-line "
         "provenance are caller assertions.\n"
         "Post-hoc mixed models can be off distribution; effects are not "
         "exclusive fact ownership.\n"
         "No baseline-perfectness assumption. Greedy starts from the supplied "
         "prefix only and stops at first error.\n";
  rows << "rank\tcompact_token_id\toriginal_token_id\tdelta_"
          "l2\ttop1\ttop4\ttop16\n";
  for (size_t rank = 0; rank < ranked.size(); ++rank) {
    const auto& row = ranked[rank];
    rows << rank + 1 << '\t' << row.row << '\t'
         << tokenizer->original_token_ids()[row.row] << '\t'
         << std::sqrt(row.squared_l2) << '\t' << (rank < 1) << '\t'
         << (rank < 4) << '\t' << (rank < 16) << '\n';
  }
  tensors << "tensor_index\tname\tvalues\tchanged_values\tdelta_l2\n";
  for (const auto& tensor : delta.tensors)
    tensors << tensor.tensor.checkpoint_index << '\t' << tensor.tensor.name
            << '\t' << tensor.summary.element_count << '\t'
            << tensor.summary.bitwise_changed_count << '\t'
            << tensor.summary.delta_l2 << '\n';
  conditions
      << "recipient\tcondition\tchanged_tensors\tchanged_master_"
         "values\ttargets\tcorrect_targets\tsentences\tteacher_exact\tgreedy_"
         "exact\tgreedy_generated_targets\tselected_correct_targets\tselected_"
         "targets\tselected_greedy_exact\tseconds\n";
  per_line << "recipient\tcondition\tcorpus_line\ttargets\tcorrect_"
              "targets\tteacher_exact\tgreedy_exact\n";
  manifest.flush();
  rows.flush();
  tensors.flush();

  const std::vector<std::string> names{
      "origin",          "embedding_top1", "embedding_top4",
      "embedding_top16", "embedding_all",  "transformers_all",
      "positions",       "final_norm",     "restored"};
  for (bool recipient_is_baseline : {true, false}) {
    const auto& origin = recipient_is_baseline ? baseline : omitted;
    const auto& donor = recipient_is_baseline ? omitted : baseline;
    const char* recipient = recipient_is_baseline ? "baseline" : "omitted";
    Evaluation origin_results;
    for (size_t condition = 0; condition < names.size(); ++condition) {
      RETURN_IF_ERROR(Restore(*executor, origin, absl::MakeSpan(weights)));
      if (condition >= 1 && condition <= 3) {
        const size_t count =
            std::min(ranked.size(), size_t{1} << (2 * (condition - 1)));
        const size_t width = config.model_width;
        for (size_t rank = 0; rank < count; ++rank) {
          const size_t offset = ranked[rank].row * width;
          RETURN_IF_ERROR(cuda::CudaStatus(
              cudaMemcpyAsync(static_cast<float*>(weights[0].data()) + offset,
                              donor[0].data() + offset, width * sizeof(float),
                              cudaMemcpyHostToDevice, executor->stream()),
              "transplant pinned ranked embedding row"));
        }
      } else if (condition == 4) {
        RETURN_IF_ERROR(CopyTensor(*executor, donor[0], weights[0]));
      } else if (condition == 5) {
        for (size_t tensor = 2; tensor + 2 < weights.size(); ++tensor)
          RETURN_IF_ERROR(
              CopyTensor(*executor, donor[tensor], weights[tensor]));
      } else if (condition == 6) {
        RETURN_IF_ERROR(CopyTensor(*executor, donor[1], weights[1]));
      } else if (condition == 7) {
        for (size_t tensor = weights.size() - 2; tensor < weights.size();
             ++tensor)
          RETURN_IF_ERROR(
              CopyTensor(*executor, donor[tensor], weights[tensor]));
      }
      ASSIGN_OR_RETURN(auto current, CopyD2H(*executor, weights));
      ASSIGN_OR_RETURN(
          auto changed,
          CompareParameterValues(layout, Views(origin), Views(current), 0));
      const bool is_control = condition == 0 || condition + 1 == names.size();
      if (is_control && changed.total.bitwise_changed_count != 0)
        return absl::InternalError(
            "recipient restoration was not byte-identical");
      const size_t changed_tensors =
          std::count_if(changed.tensors.begin(), changed.tensors.end(),
                        [](const TensorDelta& tensor) {
                          return tensor.summary.bitwise_changed_count != 0;
                        });
      const auto started = std::chrono::steady_clock::now();
      ASSIGN_OR_RETURN(auto result, Evaluate(*executor, *model, *data,
                                             config.vocabulary_size));
      const double seconds = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();
      const size_t selected = selected_line - 1;
      conditions << recipient << '\t' << names[condition] << '\t'
                 << changed_tensors << '\t'
                 << changed.total.bitwise_changed_count << '\t'
                 << result.teacher.targets << '\t'
                 << result.teacher.correct_targets << '\t'
                 << result.teacher.sentences << '\t'
                 << result.teacher.exact_sentences << '\t'
                 << result.greedy.exact_sentences << '\t'
                 << result.greedy.generated_targets << '\t'
                 << result.teacher.correct_per_sentence[selected] << '\t'
                 << result.teacher.targets_per_sentence[selected] << '\t'
                 << result.greedy.exact_per_sentence[selected] << '\t'
                 << seconds << '\n';
      for (size_t line = 0; line < lines.size(); ++line)
        per_line << recipient << '\t' << names[condition] << '\t' << line + 1
                 << '\t' << result.teacher.targets_per_sentence[line] << '\t'
                 << result.teacher.correct_per_sentence[line] << '\t'
                 << (result.teacher.targets_per_sentence[line] ==
                     result.teacher.correct_per_sentence[line])
                 << '\t' << result.greedy.exact_per_sentence[line] << '\n';
      conditions.flush();
      per_line.flush();
      if (!conditions || !per_line)
        return absl::UnknownError("cannot write intervention evaluation");
      std::cout << recipient << " <- donor " << names[condition]
                << ": correct=" << result.teacher.correct_targets << '/'
                << result.teacher.targets
                << " exact=" << result.greedy.exact_sentences << '/'
                << result.greedy.sentences << " selected_exact="
                << result.greedy.exact_per_sentence[selected]
                << " seconds=" << seconds << std::endl;
      if (condition == 0)
        origin_results = result;
      if (condition + 1 == names.size()) {
        if (!SameResults(origin_results, result))
          return absl::InternalError(
              "restored results differ from recipient origin");
        ASSIGN_OR_RETURN(auto after, CopyD2H(*executor, weights));
        ASSIGN_OR_RETURN(
            auto after_delta,
            CompareParameterValues(layout, Views(origin), Views(after), 0));
        if (after_delta.total.bitwise_changed_count != 0)
          return absl::InternalError(
              "evaluation changed restored master weights");
      }
    }
  }
  manifest << "complete=true\nboth_restored_bytes_and_results_verified=true\n";
  for (auto* stream : {&manifest, &rows, &conditions, &per_line, &tensors}) {
    stream->close();
    if (!*stream)
      return absl::UnknownError("cannot close paired-checkpoint reports");
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
