// Tests two predeclared whole-model update compositions. Component checkpoints
// must come from matched-clock, loss-masked training, not shortened independent
// training runs. Labels enter only evaluation, never parameter construction.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/padded_line_dataset.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/one_shot_memorizer/fact_superposition.h"
#include "src/llm/experiments/one_shot_memorizer/mlp_probe.h"
#include "src/llm/experiments/one_shot_memorizer/sentence_ablation.h"
#include "src/llm/gpt2.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, initial_checkpoint, "", "Common step_0 checkpoint");
ABSL_FLAG(std::string, joint_checkpoint, "",
          "Jointly trained step_1024 checkpoint");
ABSL_FLAG(std::string, omit_line_1_checkpoint, "",
          "Greece-contribution step_1024");
ABSL_FLAG(std::string, omit_line_2_checkpoint, "",
          "France-contribution step_1024");
ABSL_FLAG(std::string, tokenizer, "",
          "Original base GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "",
          "Two-line corpus: original France80 then Greece406");
ABSL_FLAG(std::string, output_dir, "",
          "Fresh directory for reports and predicted checkpoints");
ABSL_FLAG(
    std::string, control_step1_dir, "",
    "Optional trainer output with baseline, baseline_repeat, omit_line_1, "
    "omit_line_2 subdirectories; verifies common step0, endpoint repeat, "
    "and exact step1 SUM before endpoint inference");

namespace pluto::llm::one_shot_memorizer {
namespace {
namespace fs = std::filesystem;
using Snapshot = std::vector<float>;
constexpr int kVocabulary = 4475;
constexpr int kSteps = 1024;

std::vector<absl::Span<const float>> Views(absl::Span<const TensorSpec> layout,
                                           const Snapshot& snapshot) {
  std::vector<absl::Span<const float>> result;
  for (const auto& tensor : layout)
    result.push_back(absl::MakeConstSpan(snapshot).subspan(
        tensor.flat_offset, tensor.element_count));
  return result;
}

bool Identical(const Snapshot& a, const Snapshot& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

absl::StatusOr<std::vector<Buffer>> UniqueWeights(
    Layer& model, absl::Span<const TensorSpec> layout) {
  absl::flat_hash_set<const void*> seen;
  std::vector<Buffer> result;
  for (const auto& weight : model.weights())
    if (seen.insert(weight.data()).second)
      result.push_back(weight);
  if (result.size() != layout.size())
    return absl::InvalidArgumentError(
        "unique weight inventory differs from GPT-2 layout");
  for (size_t i = 0; i < result.size(); ++i)
    if (result[i].size_bytes() != layout[i].element_count * sizeof(float))
      return absl::InvalidArgumentError(
          absl::StrCat("weight shape differs: ", i));
  return result;
}

absl::StatusOr<Snapshot> CopyD2H(cuda::Executor& executor,
                                 absl::Span<const Buffer> weights,
                                 absl::Span<const TensorSpec> layout) {
  const size_t count = layout.back().flat_offset + layout.back().element_count;
  ASSIGN_OR_RETURN(auto staging,
                   cuda::PageLockedHostArray<float>::Allocate(executor, count));
  for (size_t i = 0; i < weights.size(); ++i)
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(staging.data() + layout[i].flat_offset,
                        weights[i].data(), weights[i].size_bytes(),
                        cudaMemcpyDeviceToHost, executor.stream()),
        "snapshot superposition weights"));
  RETURN_IF_ERROR(executor.Synchronize());
  Snapshot result(staging.begin(), staging.end());
  RETURN_IF_ERROR(ValidateParameterValues(layout, Views(layout, result)));
  return result;
}

absl::Status CopyH2D(cuda::Executor& executor, const Snapshot& values,
                     absl::Span<const TensorSpec> layout,
                     absl::Span<const Buffer> weights) {
  RETURN_IF_ERROR(ValidateParameterValues(layout, Views(layout, values)));
  ASSIGN_OR_RETURN(auto staging, cuda::PageLockedHostArray<float>::Allocate(
                                     executor, values.size()));
  std::copy(values.begin(), values.end(), staging.begin());
  for (size_t i = 0; i < weights.size(); ++i)
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(
            weights[i].data(), staging.data() + layout[i].flat_offset,
            weights[i].size_bytes(), cudaMemcpyHostToDevice, executor.stream()),
        "upload predicted superposition weights"));
  return executor.Synchronize();
}

// Retain the file values so a final strict reload checks source immutability.
struct Source {
  fs::path path;
  Snapshot weights;
};

absl::StatusOr<Snapshot> Load(
    cuda::Executor& executor, Layer& scratch, absl::Span<const Buffer> weights,
    absl::Span<const TensorSpec> layout,
    const tokenizer::CompactVocabularyTokenizer& vocabulary,
    const fs::path& path, int expected_step) {
  ASSIGN_OR_RETURN(const auto info, InspectCheckpointDirectory(path));
  if (info.step != expected_step)
    return absl::InvalidArgumentError(
        "checkpoint step does not match frozen protocol");
  RETURN_IF_ERROR(vocabulary.ValidateFile(path / "compact_vocabulary.tsv"));
  RETURN_IF_ERROR(ReadFromDirectory(executor, scratch, path, false));
  return CopyD2H(executor, weights, layout);
}

void WriteDelta(std::ostream& stream, absl::string_view condition,
                absl::string_view tensor, const DeltaSummary& delta) {
  stream << condition << '\t' << tensor << '\t' << delta.element_count << '\t'
         << delta.bitwise_changed_count << '\t' << delta.delta_l2 << '\t'
         << delta.maximum_absolute_delta << '\t';
  if (delta.relative_l2)
    stream << *delta.relative_l2;
  else
    stream << "NA";
  stream << '\n';
}

absl::Status Run() {
  const fs::path initial_path = absl::GetFlag(FLAGS_initial_checkpoint);
  const fs::path joint_path = absl::GetFlag(FLAGS_joint_checkpoint);
  const fs::path a_path = absl::GetFlag(FLAGS_omit_line_2_checkpoint);
  const fs::path b_path = absl::GetFlag(FLAGS_omit_line_1_checkpoint);
  const fs::path output = absl::GetFlag(FLAGS_output_dir);
  const fs::path control = absl::GetFlag(FLAGS_control_step1_dir);
  if (initial_path.empty() || joint_path.empty() || a_path.empty() ||
      b_path.empty() || output.empty() ||
      absl::GetFlag(FLAGS_tokenizer).empty() ||
      absl::GetFlag(FLAGS_corpus).empty())
    return absl::InvalidArgumentError(
        "all four checkpoints, tokenizer, corpus and fresh output_dir are "
        "required");
  std::error_code error;
  if (fs::exists(output, error) || error)
    return absl::AlreadyExistsError("output_dir must be fresh and accessible");
  ASSIGN_OR_RETURN(
      auto base, tokenizer::Gpt2Tokenizer::Load(absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto vocabulary,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *base, initial_path / "compact_vocabulary.tsv"));
  if (vocabulary->vocab_size() != kVocabulary ||
      vocabulary->eos_token_id() != 4474 ||
      vocabulary->original_eos_token_id() != base->eos_token_id())
    return absl::InvalidArgumentError(
        "expected original 4475-token compact vocabulary");
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  // Freeze the relation family and line order as well as the two targets. The
  // strings validate the protocol; parameter construction never reads them.
  absl::string_view text = corpus.text();
  if (!text.empty() && text.back() == '\n')
    text.remove_suffix(1);
  if (text !=
      "The capital of France is Paris, a city on the Seine.\n"
      "The capital of Greece is Athens, an ancient Mediterranean city.")
    return absl::InvalidArgumentError(
        "corpus differs from the fixed France/Greece pair");
  const Gpt2Config config{.transformer_block_count = 8,
                          .model_width = 16,
                          .attention_heads = 1,
                          .feed_forward_width = 64,
                          .vocabulary_size = kVocabulary,
                          .pad_vocabulary = false};
  ASSIGN_OR_RETURN(auto layout, BuildGpt2ParameterLayout(
                                    {.vocabulary_size = kVocabulary,
                                     .model_width = 16,
                                     .feed_forward_width = 64,
                                     .context_length = kGpt2ContextLength,
                                     .transformer_block_count = 8}));
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto data, PaddedLineDataSetIterator::Create(
                                  *executor, corpus.text(), *vocabulary,
                                  {.batch_size = 2,
                                   .context_length = kGpt2ContextLength,
                                   .prompt_tokens = 5,
                                   .eos_token = vocabulary->eos_token_id(),
                                   .shuffle = false}));
  if (data->sample_count() != 2 || data->supervised_row_count() != 18)
    return absl::FailedPreconditionError(
        "expected two sentences and 18 suffix/EOS targets");
  ASSIGN_OR_RETURN(auto scratch,
                   CreateGpt2(*executor, DataType::BF16, 0, config));
  ASSIGN_OR_RETURN(auto scratch_weights, UniqueWeights(*scratch, layout));
  std::vector<Source> sources;
  auto load = [&](const fs::path& path, int step) -> absl::StatusOr<Snapshot> {
    ASSIGN_OR_RETURN(auto values, Load(*executor, *scratch, scratch_weights,
                                       layout, *vocabulary, path, step));
    sources.push_back({path, values});
    return values;
  };
  ASSIGN_OR_RETURN(const auto initial, load(initial_path, 0));
  ASSIGN_OR_RETURN(const auto joint, load(joint_path, kSteps));
  ASSIGN_OR_RETURN(const auto only_a, load(a_path, kSteps));
  ASSIGN_OR_RETURN(const auto only_b, load(b_path, kSteps));
  ASSIGN_OR_RETURN(
      const auto sum,
      SuperposeFactParameters(initial, only_a, only_b, FactSuperposition::kSum));
  ASSIGN_OR_RETURN(const auto mean,
                   SuperposeFactParameters(initial, only_a, only_b,
                                           FactSuperposition::kMean));
  if (!fs::create_directory(output, error) || error)
    return absl::UnknownError("cannot create fresh output directory");
  std::ofstream manifest(output / "manifest.tsv"),
      summary(output / "conditions.tsv"), per_line(output / "per_sentence.tsv"),
      deltas(output / "parameter_errors.tsv"),
      controls(output / "controls.tsv");
  for (auto* stream : {&manifest, &summary, &per_line, &deltas, &controls}) {
    if (!*stream)
      return absl::UnknownError("cannot create superposition reports");
    *stream << std::setprecision(17);
  }
  manifest << "key\tvalue\ncompleted\tfalse\ninitial_checkpoint\t"
           << initial_path.string() << "\njoint_checkpoint\t"
           << joint_path.string() << "\nonly_a_checkpoint\t" << a_path.string()
           << "\nonly_b_checkpoint\t" << b_path.string() << "\ncorpus\t"
           << absl::GetFlag(FLAGS_corpus) << "\ntokenizer\t"
           << absl::GetFlag(FLAGS_tokenizer) << "\ncontrol_step1_dir\t"
           << control.string()
           << "\nprotocol\tFrance80/Greece406; 1024 matched-clock steps; full "
              "vocabulary"
              "\nformula_sum\tW0+((WA-W0)+(WB-W0))"
              "\nformula_mean\tW0+0.5*((WA-W0)+(WB-W0))"
              "\narithmetic\tFP64 deltas and sum, one final FP32 cast; no "
              "fitted coefficients"
              "\nparameters\tall unique masters including norms, positions and "
              "tied embedding"
              "\nevaluation\tfresh BF16 models; pair only; five-token prompt; "
              "suffix plus EOS"
              "\nprovenance\ttraining schedule/mask clocks are caller "
              "assertions from trainer reports"
              "\ncaveat\tconstructed weights may be off distribution; not "
              "exclusive fact ownership\n";
  manifest.flush();
  controls << "control\tresult\n";
  if (!control.empty()) {
    for (const char* name :
         {"baseline", "baseline_repeat", "omit_line_1", "omit_line_2"}) {
      ASSIGN_OR_RETURN(auto start, load(control / name / "step_0", 0));
      if (!Identical(initial, start))
        return absl::FailedPreconditionError(
            "training initialization differs from common step0");
      controls << name << "_initial_identical\tPASS\n";
    }
    ASSIGN_OR_RETURN(auto repeat,
                     load(control / "baseline_repeat" / "step_1024", kSteps));
    if (!Identical(joint, repeat))
      return absl::FailedPreconditionError(
          "joint/repeat endpoint bytes differ");
    controls << "joint_repeat_endpoint_identical\tPASS\n";
    ASSIGN_OR_RETURN(auto first_joint, load(control / "baseline" / "step_1", 1));
    ASSIGN_OR_RETURN(auto first_a, load(control / "omit_line_2" / "step_1", 1));
    ASSIGN_OR_RETURN(auto first_b, load(control / "omit_line_1" / "step_1", 1));
    const bool a_unchanged = Identical(initial, first_a);
    const bool b_unchanged = Identical(initial, first_b);
    if (a_unchanged == b_unchanged ||
        !Identical(first_joint, a_unchanged ? first_b : first_a))
      return absl::FailedPreconditionError(
          "step1 must have exactly one unchanged component and the other equal "
          "joint");
    ASSIGN_OR_RETURN(auto first_sum,
                     SuperposeFactParameters(initial, first_a, first_b,
                                             FactSuperposition::kSum));
    if (!Identical(first_sum, first_joint))
      return absl::FailedPreconditionError(
          "step1 SUM does not equal joint bytes");
    controls << "step1_one_component_unchanged\tPASS\nstep1_sum_equals_"
                "joint\tPASS\n";
    manifest << "step1_updated_fact\t" << (a_unchanged ? "Greece" : "France")
             << '\n';
  } else
    controls << "training_initial_repeat_and_step1_checks\tNOT_REQUESTED\n";
  summary << "condition\tcorrect_targets\ttargets\tgreedy_exact\tsentences\n";
  per_line << "condition\tpair_line\toriginal_line\tcorrect_"
              "targets\ttargets\tgreedy_exact\tfirst_mismatch_"
              "position\tpredicted_token\texpected_token\n";
  deltas << "condition\ttensor\tcoordinates\tbitwise_changed_vs_joint\tl2_"
            "error\tmax_abs_error\trelative_l2_vs_joint\n";
  const std::pair<const char*, const Snapshot*> conditions[] = {
      {"joint", &joint},
      {"France_only", &only_a},
      {"Greece_only", &only_b},
      {"sum", &sum},
      {"mean", &mean}};
  for (const auto& [name, values] : conditions) {
    ASSIGN_OR_RETURN(auto difference,
                     CompareParameterValues(layout, Views(layout, joint),
                                            Views(layout, *values), 0));
    WriteDelta(deltas, name, "ALL", difference.total);
    for (const auto& tensor : difference.tensors)
      WriteDelta(deltas, name, tensor.tensor.name, tensor.summary);
    // Every condition receives a newly constructed model. The source model is
    // only a strict-loading/snapshot scratchpad and is never evaluated here.
    ASSIGN_OR_RETURN(auto model,
                     CreateGpt2(*executor, DataType::BF16, 0, config));
    ASSIGN_OR_RETURN(auto weights, UniqueWeights(*model, layout));
    RETURN_IF_ERROR(CopyH2D(*executor, *values, layout, weights));
    ASSIGN_OR_RETURN(auto before, CopyD2H(*executor, weights, layout));
    if (!Identical(before, *values))
      return absl::DataLossError(
          "uploaded construction does not match requested bytes");
    ASSIGN_OR_RETURN(
        auto teacher,
        EvaluateMlpReplacements(*executor, *model, *data, {}, kVocabulary));
    ASSIGN_OR_RETURN(
        auto greedy,
        VerifyMlpGreedyCompletions(*executor, *model, *data, {}, kVocabulary));
    if (teacher.targets != 18 || teacher.sentences != 2 ||
        greedy.sentences != 2 || teacher.targets_per_sentence.size() != 2 ||
        teacher.correct_per_sentence.size() != 2 ||
        greedy.exact_per_sentence.size() != 2 ||
        greedy.first_mismatch_per_sentence.size() != 2)
      return absl::InternalError("pair evaluation cardinality changed");
    for (size_t line = 0; line < 2; ++line) {
      if ((teacher.targets_per_sentence[line] ==
           teacher.correct_per_sentence[line]) !=
          greedy.exact_per_sentence[line])
        return absl::InternalError("teacher and greedy exactness disagree");
      per_line << name << '\t' << line + 1 << '\t' << (line == 0 ? 80 : 406)
               << '\t' << teacher.correct_per_sentence[line] << '\t'
               << teacher.targets_per_sentence[line] << '\t'
               << greedy.exact_per_sentence[line];
      if (const auto& failed = greedy.first_mismatch_per_sentence[line])
        per_line << '\t' << failed->target_position << '\t'
                 << failed->predicted_token << '\t' << failed->expected_token;
      else
        per_line << "\tNA\tNA\tNA";
      per_line << '\n';
    }
    summary << name << '\t' << teacher.correct_targets << '\t'
            << teacher.targets << '\t' << greedy.exact_sentences << '\t'
            << greedy.sentences << '\n';
    if ((std::string(name) == "France_only" && !greedy.exact_per_sentence[0]) ||
        (std::string(name) == "Greece_only" && !greedy.exact_per_sentence[1])) {
      std::cerr << "WARNING: " << name
                << " has not memorized its retained fact\n";
      manifest << "warning\t" << name
               << " has not memorized its retained fact\n";
    }
    if (std::string(name) == "sum" || std::string(name) == "mean") {
      const fs::path predicted = output / name / "step_1024";
      RETURN_IF_ERROR(WriteToDirectory(*executor, *model, predicted));
      RETURN_IF_ERROR(
          vocabulary->SaveToFile(predicted / "compact_vocabulary.tsv"));
    }
    ASSIGN_OR_RETURN(auto after, CopyD2H(*executor, weights, layout));
    if (!Identical(before, after))
      return absl::DataLossError(
          "inference or saving changed construction weights");
    controls << name << "_weights_unchanged\tPASS\n";
    std::cout << name << ": " << teacher.correct_targets << "/18 targets, "
              << greedy.exact_sentences << "/2 exact completions\n";
  }
  // All source reads are repeated after every output has been produced. No
  // source path is ever passed to WriteToDirectory or SaveToFile.
  for (const auto& source : sources) {
    ASSIGN_OR_RETURN(auto info, InspectCheckpointDirectory(source.path));
    ASSIGN_OR_RETURN(auto after,
                     Load(*executor, *scratch, scratch_weights, layout,
                          *vocabulary, source.path, info.step));
    if (!Identical(source.weights, after))
      return absl::DataLossError(
          "a source checkpoint changed during the probe");
  }
  controls << "all_source_weight_bytes_unchanged\tPASS\n";
  manifest << "source_checkpoints_checked\t" << sources.size()
           << "\ncompleted\ttrue\n";
  for (auto* stream : {&manifest, &summary, &per_line, &deltas, &controls}) {
    stream->flush();
    if (!*stream)
      return absl::UnknownError("superposition report write failed");
  }
  return absl::OkStatus();
}
}  // namespace
}  // namespace pluto::llm::one_shot_memorizer

int main(int argc, char** argv) {
  if (absl::ParseCommandLine(argc, argv).size() != 1) {
    std::cerr << "Unexpected positional arguments\n";
    return 1;
  }
  const auto status = pluto::llm::one_shot_memorizer::Run();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
