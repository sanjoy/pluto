// Tests a conditional moment-based encoding hypothesis for an actual trained
// checkpoint. The backbone remains learned; none of these readouts is claimed
// to reconstruct its features, original parameters, or arbitrary-prompt
// behavior.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_cat.h"
#include "src/cuda/executor.h"
#include "src/dataset/compact_vocabulary.h"
#include "src/dataset/dataset.h"
#include "src/dataset/gpt2_tokenizer.h"
#include "src/dataset/padded_line_dataset.h"
#include "src/llm/checkpoint.h"
#include "src/llm/experiments/one_shot_memorizer/associative_readout.h"
#include "src/llm/experiments/one_shot_memorizer/feature_capture.h"
#include "src/llm/gpt2.h"
#include "src/util/status_macros.h"
#include "src/util/tee_stream.h"

ABSL_FLAG(std::string, checkpoint, "", "Exact memorized checkpoint directory");
ABSL_FLAG(std::string, tokenizer, "", "Base GPT-2 tokenizer directory");
ABSL_FLAG(std::string, corpus, "testdata/general_facts_dataset.txt",
          "Evaluation corpus");
ABSL_FLAG(std::string, output_dir, "", "Fresh report directory");
ABSL_FLAG(int, batch_size, 32, "Corpus sentences per capture batch");
ABSL_FLAG(int, prompt_tokens, 5, "Supplied prefix length");
ABSL_FLAG(int, layers, 8, "Transformer block count");
ABSL_FLAG(int, model_width, 16, "Residual width");
ABSL_FLAG(int, attention_heads, 1, "Attention heads");
ABSL_FLAG(int, feed_forward_width, 64, "Inner MLP width");

namespace pluto::llm::one_shot_memorizer {
namespace {

// The checkpoint has trained rows even for tokens never used as supervised
// targets. Unlike a fitted prototype decoder, those rows must still compete in
// the full-vocabulary control. Do not invent observations for those classes.
ReadoutEvaluation EvaluateTiedHead(const AssociativeReadout& head,
                                   absl::Span<const float> features,
                                   absl::Span<const int> labels) {
  ReadoutEvaluation evaluation;
  evaluation.sample_count = labels.size();
  for (size_t i = 0; i < labels.size(); ++i) {
    double best = -std::numeric_limits<double>::infinity();
    int prediction = -1;
    for (int token : head.seen_classes) {
      double dot = 0;
      for (int d = 0; d < head.width; ++d)
        dot += static_cast<double>(features[i * head.width + d]) *
               head.weights[static_cast<size_t>(token) * head.width + d];
      if (dot > best) {
        best = dot;
        prediction = token;
      }
    }
    evaluation.predictions.push_back(prediction);
    evaluation.correct_count += prediction == labels[i];
  }
  evaluation.accuracy =
      static_cast<double>(evaluation.correct_count) / labels.size();
  return evaluation;
}

// Centering rows removes the common weight vector that adds the same logit
// to all competing classes. One fitted global scale compares geometry without
// silently granting one independent fit per class or coordinate.
struct Alignment {
  double scale;
  double relative_error;
  double mean_cosine;
};
Alignment CompareGeometry(const AssociativeReadout& reference,
                          const AssociativeReadout& candidate) {
  const int width = reference.width;
  std::vector<double> a_mean(width), b_mean(width);
  for (int token : candidate.seen_classes)
    for (int d = 0; d < width; ++d) {
      a_mean[d] += reference.weights[static_cast<size_t>(token) * width + d];
      b_mean[d] += candidate.weights[static_cast<size_t>(token) * width + d];
    }
  for (int d = 0; d < width; ++d) {
    a_mean[d] /= candidate.seen_classes.size();
    b_mean[d] /= candidate.seen_classes.size();
  }
  double aa = 0, bb = 0, ab = 0, cosines = 0;
  for (int token : candidate.seen_classes) {
    double row_aa = 0, row_bb = 0, row_ab = 0;
    for (int d = 0; d < width; ++d) {
      const double a =
          reference.weights[static_cast<size_t>(token) * width + d] - a_mean[d];
      const double b =
          candidate.weights[static_cast<size_t>(token) * width + d] - b_mean[d];
      row_aa += a * a;
      row_bb += b * b;
      row_ab += a * b;
    }
    aa += row_aa;
    bb += row_bb;
    ab += row_ab;
    if (row_aa > 0 && row_bb > 0)
      cosines += row_ab / std::sqrt(row_aa * row_bb);
  }
  // A negative scale reverses the classifier's preferences, so cannot count
  // as equivalent geometry even when it gives a small unconstrained error.
  const double scale = bb > 0 ? std::max(0.0, ab / bb) : 0;
  return {
      scale,
      aa > 0 ? std::sqrt(
                   std::max(0.0, aa - 2 * scale * ab + scale * scale * bb) / aa)
             : 0,
      cosines / candidate.seen_classes.size()};
}

absl::Status RunProbe() {
  namespace fs = std::filesystem;
  const fs::path checkpoint = absl::GetFlag(FLAGS_checkpoint);
  const fs::path output_dir = absl::GetFlag(FLAGS_output_dir);
  const int width = absl::GetFlag(FLAGS_model_width);
  if (checkpoint.empty() || output_dir.empty() ||
      absl::GetFlag(FLAGS_tokenizer).empty())
    return absl::InvalidArgumentError(
        "checkpoint, output_dir and tokenizer are required");
  ASSIGN_OR_RETURN(
      auto base, tokenizer::Gpt2Tokenizer::Load(absl::GetFlag(FLAGS_tokenizer)));
  ASSIGN_OR_RETURN(auto tokenizer,
                   tokenizer::CompactVocabularyTokenizer::LoadFromFile(
                       *base, checkpoint / "compact_vocabulary.tsv"));
  if (tokenizer->original_eos_token_id() != base->eos_token_id())
    return absl::InvalidArgumentError("tokenizer EOS differs from checkpoint");
  ASSIGN_OR_RETURN(auto executor, cuda::Executor::Create());
  ASSIGN_OR_RETURN(auto corpus, LoadTextCorpus(absl::GetFlag(FLAGS_corpus)));
  ASSIGN_OR_RETURN(auto dataset,
                   PaddedLineDataSetIterator::Create(
                       *executor, corpus.text(), *tokenizer,
                       {.batch_size = absl::GetFlag(FLAGS_batch_size),
                        .context_length = kGpt2ContextLength,
                        .prompt_tokens = absl::GetFlag(FLAGS_prompt_tokens),
                        .eos_token = tokenizer->eos_token_id()}));
  const int vocab = tokenizer->vocab_size();
  const Gpt2Config config{
      .transformer_block_count = absl::GetFlag(FLAGS_layers),
      .model_width = width,
      .attention_heads = absl::GetFlag(FLAGS_attention_heads),
      .feed_forward_width = absl::GetFlag(FLAGS_feed_forward_width),
      .vocabulary_size = vocab,
      .pad_vocabulary = false};
  ASSIGN_OR_RETURN(auto model, CreateGpt2(*executor, DataType::BF16, 0, config));
  RETURN_IF_ERROR(
      ReadFromDirectory(*executor, *model, checkpoint.string(), false));
  std::error_code error;
  const bool created = fs::create_directory(output_dir, error);
  if (error)
    return absl::UnknownError(
        absl::StrCat("cannot create output_dir: ", error.message()));
  if (!created)
    return absl::AlreadyExistsError("output_dir must be fresh");
  std::ofstream file(output_dir / "readout_probe.tsv");
  if (!file)
    return absl::UnknownError("cannot create readout report");
  util::TeeStream report(std::cout, file);
  report << "# checkpoint=" << checkpoint.string()
         << "\n# corpus=" << absl::GetFlag(FLAGS_corpus)
         << "\n# tokenizer=" << absl::GetFlag(FLAGS_tokenizer)
         << "\n# model_width=" << width
         << "\n# layers=" << config.transformer_block_count
         << "\n# attention_heads=" << config.attention_heads
         << "\n# feed_forward_width=" << config.feed_forward_width
         << "\n# batch_size=" << absl::GetFlag(FLAGS_batch_size)
         << "\n# prompt_tokens=" << absl::GetFlag(FLAGS_prompt_tokens)
         << "\n# Conditional readout reconstruction on LEARNED final-norm "
            "features.\n"
         << "# No input embedding is replaced; these are independent CPU "
            "output heads.\n";
  std::vector<float> features;
  std::vector<int> labels, original_predictions;
  std::vector<size_t> sentence_ids;
  size_t first_sentence = 0;
  for (size_t batch_index = 0; batch_index < dataset->batches_per_epoch();
       ++batch_index) {
    ASSIGN_OR_RETURN(auto batch, dataset->Next());
    ASSIGN_OR_RETURN(auto captured, CaptureGpt2ReadoutBatch(
                                        *executor, *model, batch, width, vocab));
    features.insert(features.end(), captured.values.begin(),
                    captured.values.end());
    labels.insert(labels.end(), captured.labels.begin(), captured.labels.end());
    original_predictions.insert(original_predictions.end(),
                                captured.original_predictions.begin(),
                                captured.original_predictions.end());
    for (int sample : captured.sample_indices)
      sentence_ids.push_back(first_sentence + sample);
    first_sentence += batch.batch_size;
  }
  if (labels.empty() || first_sentence != dataset->sample_count() ||
      labels.size() != static_cast<size_t>(dataset->supervised_row_count()))
    return absl::InternalError("feature capture omitted samples or targets");
  if (original_predictions != labels)
    return absl::FailedPreconditionError(
        "unaltered checkpoint is not fully memorized");
  if (dataset->sample_count() >
      static_cast<size_t>(std::numeric_limits<int>::max()))
    return absl::OutOfRangeError("too many sentences for group-held-out IDs");
  const std::vector<int> groups(sentence_ids.begin(), sentence_ids.end());
  ASSIGN_OR_RETURN(auto means, BuildReadout(features, labels, width, vocab));
  AssociativeReadout original = means;
  ASSIGN_OR_RETURN(
      original.weights,
      CopyEffectiveBf16Readout(*executor, model->weights()[0], width, vocab));
  original.biases.assign(vocab, 0);
  original.seen_classes.resize(vocab);
  std::iota(original.seen_classes.begin(), original.seen_classes.end(), 0);
  const auto cpu_baseline = EvaluateTiedHead(original, features, labels);
  if (cpu_baseline.predictions != original_predictions)
    return absl::FailedPreconditionError(
        "CPU effective-BF16 head does not reproduce GPU top-1");
  report
      << "# GPU and independent CPU tied-head controls agree on all "
      << labels.size() << " targets.\n"
      << "# seen_target_classes=" << means.seen_classes.size()
      << "\n# unsupported_target_classes=" << vocab - means.seen_classes.size()
      << "\n# singleton_classes=" << means.singleton_class_count
      << "\n# LOO excludes one query row, NOT its whole sentence or backbone "
         "training.\n"
      << "# Leave-one-sentence-out excludes other rows of the same sentence, "
         "but\n"
         "# still uses the globally trained backbone. Unsupported targets "
         "count as wrong.\n"
      << "# exact_sentences is a teacher-forced all-rows-correct count. With a "
         "fixed replacement head,\n"
      << "# deterministic causal greedy decoding has the same exact count by "
         "induction.\n"
      << "# LOO uses a different head for each query: its exact_sentences "
         "field is not applicable.\n"
      << "readout\tprotocol\tcorrect\ttargets\texact_"
         "sentences\tsentences\tsingleton_correct\tsingleton_targets\trepeated_"
         "correct\trepeated_targets\tunseen_targets\tseconds\n";
  auto print = [&](const std::string& name, const std::string& protocol,
                   const ReadoutEvaluation& evaluation,
                   absl::Span<const int> targets, double seconds) {
    size_t singleton_correct = 0, singleton_targets = 0, repeated_correct = 0,
           repeated_targets = 0;
    std::vector<bool> exact(dataset->sample_count(), true);
    for (size_t i = 0; i < targets.size(); ++i) {
      const bool correct = evaluation.predictions[i] == targets[i];
      if (means.class_counts[targets[i]] == 1) {
        ++singleton_targets;
        singleton_correct += correct;
      } else {
        ++repeated_targets;
        repeated_correct += correct;
      }
      if (!correct)
        exact[sentence_ids[i]] = false;
    }
    report << name << '\t' << protocol << '\t' << evaluation.correct_count
           << '\t' << targets.size() << '\t';
    if (protocol == "self")
      report << std::count(exact.begin(), exact.end(), true);
    else
      report << "NA";
    report << '\t' << dataset->sample_count() << '\t' << singleton_correct
           << '\t' << singleton_targets << '\t' << repeated_correct << '\t'
           << repeated_targets << '\t' << evaluation.unseen_target_count << '\t'
           << seconds << '\n';
  };
  print("original_full", "self", cpu_baseline, labels, 0);
  original.seen_classes = means.seen_classes;
  const auto supported_control = EvaluateTiedHead(original, features, labels);
  print("original_supported", "self", supported_control, labels, 0);
  std::ofstream geometry(output_dir / "readout_geometry.tsv");
  geometry << "readout\tglobal_scale\tcentered_relative_frobenius_error\tmean_"
              "centered_row_cosine\n";
  struct Recipe {
    const char* name;
    ReadoutOptions options;
    bool loo;
  };
  const std::vector<Recipe> recipes{
      {"mean_dot", {ReadoutMode::kMeanDot}, true},
      {"mean_dot_normalized", {ReadoutMode::kMeanDot, 1e-3, true}, true},
      {"nearest_centroid", {ReadoutMode::kNearestCentroid}, true},
      {"nearest_centroid_normalized",
       {ReadoutMode::kNearestCentroid, 1e-3, true},
       true},
      {"lda_ridge_0.1", {ReadoutMode::kSharedCovarianceLda, 0.1}, false},
      {"lda_ridge_0.01", {ReadoutMode::kSharedCovarianceLda, 0.01}, false},
      {"lda_ridge_0.001", {ReadoutMode::kSharedCovarianceLda, 0.001}, true}};
  for (const auto& recipe : recipes) {
    auto start = std::chrono::steady_clock::now();
    ASSIGN_OR_RETURN(auto readout, BuildReadout(features, labels, width, vocab,
                                                recipe.options));
    ASSIGN_OR_RETURN(auto evaluation,
                     EvaluateReadout(readout, features, labels));
    print(
        recipe.name, "self", evaluation, labels,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count());
    const auto alignment = CompareGeometry(original, readout);
    geometry << recipe.name << '\t' << alignment.scale << '\t'
             << alignment.relative_error << '\t' << alignment.mean_cosine
             << '\n';
    if (recipe.loo) {
      start = std::chrono::steady_clock::now();
      ASSIGN_OR_RETURN(
          auto loo, EvaluateReadoutLeaveOneOut(features, labels, width, vocab,
                                               recipe.options));
      print(recipe.name, "leave_one_row_out", loo, labels,
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          start)
                .count());
    }
    if (std::string(recipe.name) == "nearest_centroid" ||
        std::string(recipe.name) == "lda_ridge_0.001") {
      start = std::chrono::steady_clock::now();
      ASSIGN_OR_RETURN(auto held_out, EvaluateReadoutLeaveGroupOut(
                                          features, labels, groups, width,
                                          vocab, recipe.options));
      print(recipe.name, "leave_one_sentence_out", held_out, labels,
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          start)
                .count());
    }
  }
  // A global renaming of classes changes nothing. Shuffling labels BETWEEN
  // examples preserves class frequencies but destroys their feature
  // association. Singleton prototypes still self-match: report them separately,
  // not as evidence.
  auto shuffled = labels;
  std::mt19937 random(0);
  std::shuffle(shuffled.begin(), shuffled.end(), random);
  const auto start = std::chrono::steady_clock::now();
  const ReadoutOptions null_options{ReadoutMode::kNearestCentroid};
  ASSIGN_OR_RETURN(auto null_model,
                   BuildReadout(features, shuffled, width, vocab, null_options));
  ASSIGN_OR_RETURN(auto null_evaluation,
                   EvaluateReadout(null_model, features, shuffled));
  print("shuffled_nearest_centroid", "shuffled_self", null_evaluation, shuffled,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count());
  if (!file || !geometry)
    return absl::UnknownError("failed writing readout reports");
  report << "# Complete\n";
  return absl::OkStatus();
}
}  // namespace
}  // namespace pluto::llm::one_shot_memorizer

int main(int argc, char** argv) {
  if (absl::ParseCommandLine(argc, argv).size() != 1) {
    std::cerr << "unexpected positional arguments\n";
    return 1;
  }
  const auto status = pluto::llm::one_shot_memorizer::RunProbe();
  if (!status.ok()) {
    std::cerr << status << '\n';
    return 1;
  }
  return 0;
}
