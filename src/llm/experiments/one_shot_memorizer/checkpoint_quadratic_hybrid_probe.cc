// A fixed, post-hoc hybrid experiment: retain preselected original MLPs and
// replace the rest with previously fitted quadratic maps. No fitting or
// outcome-dependent subset selection occurs in this program.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "src/cuda/executor.h"
#include "src/cuda/page_locked_host_array.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/padded_line_dataset.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/one_shot_memorizer/mlp_probe.h"
#include "src/llm/experiments/one_shot_memorizer/quadratic_features.h"
#include "src/llm/experiments/one_shot_memorizer/quadratic_probe_artifacts.h"
#include "src/llm/gpt2.h"
#include "src/llm/layers/combinators.h"
#include "src/llm/layers/fully_connected.h"
#include "src/util/status_macros.h"

ABSL_FLAG(std::string, checkpoint, "", "Original memorized checkpoint");
ABSL_FLAG(std::string, tokenizer, "", "Base GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt",
          "Original 1,024-fact corpus");
ABSL_FLAG(std::string, construction_dir, "",
          "Saved quadratic construction containing coefficients.tsv");
ABSL_FLAG(std::string, output_dir, "", "Fresh experiment output directory");

namespace pluto::llm::one_shot_memorizer {
namespace {

constexpr int kBlocks = 8;
constexpr int kWidth = 16;
constexpr int kVocabulary = 4475;
constexpr int kPrompt = 5;
constexpr int kBatch = 32;

struct Condition {
  const char* name;           // Exact retained-original set, not replaced set.
  std::vector<int> retained;  // Fixed before any outcome is observed.
};

absl::StatusOr<std::vector<std::vector<uint8_t>>> Snapshot(
    cuda::Executor& executor, const Layer& layer) {
  std::vector<std::vector<uint8_t>> result;
  for (const auto& weight : layer.weights()) {
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<uint8_t>::Allocate(
                                    executor, weight.size_bytes()));
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(host.data(), weight.data(), host.size_bytes(),
                        cudaMemcpyDeviceToHost, executor.stream()),
        "snapshot hybrid model weights"));
    RETURN_IF_ERROR(executor.Synchronize());
    result.emplace_back(host.begin(), host.end());
  }
  return result;
}

absl::StatusOr<std::unique_ptr<ComposedLayer>> MakeReplacement(
    cuda::Executor& executor, const QuadraticCoefficients& coefficients) {
  ASSIGN_OR_RETURN(auto projection, FullyConnectedLayer::Create(
                                        executor, 152, kWidth, DataType::BF16,
                                        kGpt2ContextLength));
  const absl::Span<const float> parts[] = {coefficients.weights,
                                           coefficients.bias};
  for (int part = 0; part < 2; ++part) {
    ASSIGN_OR_RETURN(auto host, cuda::PageLockedHostArray<float>::CopyFrom(
                                    executor, parts[part]));
    if (host.size_bytes() != projection->weights()[part].size_bytes())
      return absl::InvalidArgumentError(
          "saved quadratic tensor shape mismatch");
    RETURN_IF_ERROR(cuda::CudaStatus(
        cudaMemcpyAsync(projection->weights()[part].data(), host.data(),
                        host.size_bytes(), cudaMemcpyHostToDevice,
                        executor.stream()),
        "upload saved hybrid quadratic coefficients"));
  }
  ComposedLayerBuilder builder;
  RETURN_IF_ERROR(builder.add(
      QuadraticFeaturesLayer::Create(executor, kGpt2ContextLength)));
  RETURN_IF_ERROR(builder.add(std::move(projection)));
  return builder.create("saved_quadratic_mlp");
}

absl::Status Run() {
  namespace fs = std::filesystem;
  const fs::path checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const fs::path construction = absl::GetFlag(FLAGS_construction_dir);
  const fs::path output = absl::GetFlag(FLAGS_output_dir);
  if (checkpoint.empty() || construction.empty() || output.empty() ||
      absl::GetFlag(FLAGS_tokenizer).empty())
    return absl::InvalidArgumentError(
        "checkpoint, construction_dir, tokenizer and fresh output_dir "
        "required");
  std::error_code error;
  if (fs::exists(output, error))
    return absl::AlreadyExistsError("output_dir must be fresh");
  if (error)
    return absl::UnknownError(error.message());
  std::ifstream saved(construction / "coefficients.tsv");
  if (!saved)
    return absl::NotFoundError("cannot open saved coefficients.tsv");
  ASSIGN_OR_RETURN(auto coefficients, ReadQuadraticCoefficients(saved));
  ASSIGN_OR_RETURN(
      auto base, tokenizer::Gpt2Tokenizer::Load(absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto compact,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *base, checkpoint / "compact_vocabulary.tsv"));
  if (compact->vocab_size() != kVocabulary ||
      compact->original_eos_token_id() != base->eos_token_id())
    return absl::InvalidArgumentError("unexpected compact vocabulary");
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto dataset, PaddedLineDataSetIterator::Create(
                                     *executor, corpus.text(), *compact,
                                     {.batch_size = kBatch,
                                      .context_length = kGpt2ContextLength,
                                      .prompt_tokens = kPrompt,
                                      .eos_token = compact->eos_token_id()}));
  size_t real_rows = 0;
  for (size_t sentence = 0; sentence < dataset->sample_count(); ++sentence)
    real_rows += dataset->sample_tokens(sentence).size();
  if (dataset->sample_count() != 1024 ||
      dataset->supervised_row_count() != 10002 || real_rows != 14098)
    return absl::InvalidArgumentError("corpus differs from fixed experiment");
  ASSIGN_OR_RETURN(auto model, CreateGpt2(*executor, DataType::BF16, 0,
                                          {.transformer_block_count = kBlocks,
                                           .model_width = kWidth,
                                           .attention_heads = 1,
                                           .feed_forward_width = 64,
                                           .vocabulary_size = kVocabulary,
                                           .pad_vocabulary = false}));
  RETURN_IF_ERROR(
      ReadFromDirectory(*executor, *model, checkpoint.string(), false));
  ASSIGN_OR_RETURN(auto original_before, Snapshot(*executor, *model));
  std::vector<std::unique_ptr<ComposedLayer>> owners;
  std::vector<std::vector<std::vector<uint8_t>>> replacements_before;
  for (int block = 0; block < kBlocks; ++block) {
    ASSIGN_OR_RETURN(auto layer,
                     MakeReplacement(*executor, coefficients[block]));
    ASSIGN_OR_RETURN(auto snapshot, Snapshot(*executor, *layer));
    replacements_before.push_back(std::move(snapshot));
    owners.push_back(std::move(layer));
  }
  if (!fs::create_directory(output, error) || error)
    return absl::UnknownError(
        absl::StrCat("cannot create output: ", error.message()));
  std::ofstream manifest(output / "manifest.tsv"),
      scores(output / "evaluations.tsv"), cases(output / "sentences.tsv"),
      failures(output / "first_failures.tsv");
  for (auto* stream : {&manifest, &scores, &cases, &failures}) {
    if (!*stream)
      return absl::UnknownError("cannot create hybrid report");
    *stream << std::setprecision(17);
  }
  manifest
      << "key\tvalue\ncompleted\tfalse\ncheckpoint\t" << checkpoint.string()
      << "\nconstruction_dir\t" << construction.string() << "\ntokenizer\t"
      << absl::GetFlag(FLAGS_tokenizer) << "\ncorpus\t"
      << absl::GetFlag(FLAGS_corpus)
      << "\nretained_original_sets\tall;none;0;7;0,7;0,1;0,1,7"
         "\nselection\tpreselected fixed list; no adaptive search"
         "\nfitting\tnone; reuse saved FP32 quadratic coefficients verbatim"
         "\nreplacement\tfresh same-forward MLP LayerNorm input at all "
         "positions"
         "\nquadratic_basis\tlinear16 plus raw upper-triangle136; BF16 products"
         "\nbackbone\tall attention, LayerNorm, residuals and head remain "
         "learned"
         "\nbatch_size\t32\ncontext_length\t1024\nprompt_tokens\t5"
         "\ncorpus_sentences\t1024\nreal_rows\t14098\ntargets\t10002"
         "\nfit_sentences\t819\nheld_sentences\t205"
         "\nheld_definition\tzero-based corpus index divisible by5; excluded "
         "only"
         " from original quadratic regression, NOT backbone training"
         "\nfirst_failure_position\tabsolute zero-based target; query=target-1"
         "\nfirst_failure_prefix\treport-only gold IDs equal actual generated"
         " prefix before first mismatch\n";
  manifest.flush();
  scores
      << "retained_original\tretained_count\tgroup\tcorrect_targets\ttargets\t"
         "teacher_exact\tgreedy_exact\tsentences\tgenerated_targets\tseconds\n";
  cases << "retained_original\tline_1based\tgroup\tcorrect_targets\ttargets\t"
           "greedy_exact\tgenerated_targets\n";
  failures << "retained_original\tline_1based\tgroup\ttarget_position\tquery_"
              "position\t"
              "predicted_token\texpected_token\tgold_prefix_ids\n";
  const std::vector<Condition> conditions{{"all", {0, 1, 2, 3, 4, 5, 6, 7}},
                                          {"none", {}},
                                          {"0", {0}},
                                          {"7", {7}},
                                          {"0,7", {0, 7}},
                                          {"0,1", {0, 1}},
                                          {"0,1,7", {0, 1, 7}}};
  for (const auto& condition : conditions) {
    std::vector<MlpReplacement> replacements;
    for (int block = 0; block < kBlocks; ++block)
      if (std::find(condition.retained.begin(), condition.retained.end(),
                    block) == condition.retained.end())
        replacements.push_back(
            {block, MlpSource::kLayerNorm, owners[block].get()});
    const auto started = std::chrono::steady_clock::now();
    ASSIGN_OR_RETURN(auto teacher,
                     EvaluateMlpReplacements(*executor, *model, *dataset,
                                             replacements, kVocabulary));
    ASSIGN_OR_RETURN(auto greedy,
                     VerifyMlpGreedyCompletions(*executor, *model, *dataset,
                                                replacements, kVocabulary));
    if (teacher.targets != 10002 || teacher.sentences != 1024 ||
        greedy.sentences != 1024 ||
        teacher.correct_per_sentence.size() != 1024 ||
        teacher.targets_per_sentence.size() != 1024 ||
        greedy.exact_per_sentence.size() != 1024 ||
        greedy.first_mismatch_per_sentence.size() != 1024)
      return absl::InternalError("incomplete hybrid corpus evaluation");
    int64_t correct[2]{}, targets[2]{}, exact[2]{}, sentences[2]{},
        generated[2]{};
    for (size_t sentence = 0; sentence < 1024; ++sentence) {
      const bool held = sentence % 5 == 0;
      const bool matches = teacher.correct_per_sentence[sentence] ==
                           teacher.targets_per_sentence[sentence];
      const auto& mismatch = greedy.first_mismatch_per_sentence[sentence];
      if (matches != greedy.exact_per_sentence[sentence] ||
          matches == mismatch.has_value())
        return absl::InternalError(
            "hybrid teacher/greedy/mismatch disagreement");
      int64_t generated_count = teacher.targets_per_sentence[sentence];
      if (mismatch) {
        const auto tokens = dataset->sample_tokens(sentence);
        if (mismatch->target_position < kPrompt ||
            static_cast<size_t>(mismatch->target_position) > tokens.size() ||
            mismatch->predicted_token == mismatch->expected_token)
          return absl::InternalError("malformed hybrid first mismatch");
        const int expected =
            static_cast<size_t>(mismatch->target_position) == tokens.size()
                ? compact->eos_token_id()
                : tokens[mismatch->target_position];
        if (mismatch->expected_token != expected)
          return absl::InternalError(
              "hybrid failure target disagrees with corpus");
        generated_count = mismatch->target_position - kPrompt + 1;
        failures << condition.name << '\t' << sentence + 1 << '\t'
                 << (held ? "held" : "fit") << '\t' << mismatch->target_position
                 << '\t' << mismatch->target_position - 1 << '\t'
                 << mismatch->predicted_token << '\t'
                 << mismatch->expected_token << '\t';
        for (int position = 0; position < mismatch->target_position; ++position)
          failures << (position == 0 ? "" : ",") << tokens[position];
        failures << '\n';
      }
      correct[held] += teacher.correct_per_sentence[sentence];
      targets[held] += teacher.targets_per_sentence[sentence];
      exact[held] += matches;
      generated[held] += generated_count;
      ++sentences[held];
      cases << condition.name << '\t' << sentence + 1 << '\t'
            << (held ? "held" : "fit") << '\t'
            << teacher.correct_per_sentence[sentence] << '\t'
            << teacher.targets_per_sentence[sentence] << '\t' << matches << '\t'
            << generated_count << '\n';
    }
    if (correct[0] + correct[1] != teacher.correct_targets ||
        exact[0] + exact[1] != teacher.exact_sentences ||
        teacher.exact_sentences != greedy.exact_sentences ||
        generated[0] + generated[1] != greedy.generated_targets)
      return absl::InternalError("hybrid aggregate accounting mismatch");
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - started)
                               .count();
    for (int group = 0; group < 3; ++group) {
      const bool all = group == 2;
      scores << condition.name << '\t' << condition.retained.size() << '\t'
             << (all          ? "all"
                 : group == 0 ? "fit"
                              : "held")
             << '\t' << (all ? teacher.correct_targets : correct[group]) << '\t'
             << (all ? teacher.targets : targets[group]) << '\t'
             << (all ? teacher.exact_sentences : exact[group]) << '\t'
             << (all ? greedy.exact_sentences : exact[group]) << '\t'
             << (all ? teacher.sentences : sentences[group]) << '\t'
             << (all ? greedy.generated_targets : generated[group]) << '\t'
             << seconds << '\n';
    }
    for (auto* stream : {&scores, &cases, &failures}) {
      stream->flush();
      if (!*stream)
        return absl::UnknownError("cannot write hybrid evaluation");
    }
    std::cout << "Retained original {" << condition.name
              << "}: " << teacher.correct_targets << "/10002 targets, "
              << greedy.exact_sentences << "/1024 exact greedy; " << seconds
              << " s" << std::endl;
    if (std::string(condition.name) == "all" &&
        (teacher.correct_targets != 10002 || greedy.exact_sentences != 1024))
      return absl::FailedPreconditionError(
          "original-model positive control failed");
    if (std::string(condition.name) == "none" &&
        (teacher.correct_targets != 9968 || greedy.exact_sentences != 996))
      return absl::FailedPreconditionError(
          "all-quadratic reproduction control failed");
  }
  ASSIGN_OR_RETURN(auto original_after, Snapshot(*executor, *model));
  if (original_before != original_after)
    return absl::FailedPreconditionError(
        "original model master weights changed");
  for (int block = 0; block < kBlocks; ++block) {
    ASSIGN_OR_RETURN(auto after, Snapshot(*executor, *owners[block]));
    if (after != replacements_before[block])
      return absl::FailedPreconditionError(
          "saved quadratic coefficients changed");
  }
  manifest << "original_positive_control\tPASS\n"
              "all_quadratic_reproduction_control\tPASS\n"
              "original_master_bytes_unchanged\tPASS\n"
              "replacement_master_bytes_unchanged\tPASS\ncompleted\ttrue\n";
  manifest.flush();
  return manifest ? absl::OkStatus()
                  : absl::UnknownError("cannot finish manifest");
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
